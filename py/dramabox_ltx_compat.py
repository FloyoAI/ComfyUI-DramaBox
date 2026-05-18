from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import TypeVar

import torch
from tqdm import tqdm

from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.transformer import Modality, X0Model
from ltx_core.types import Audio, LatentState
from ltx_core.utils import to_denoised, to_velocity
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    generate_enhanced_prompt,
    post_process_latent,
)
from ltx_pipelines.utils.model_ledger import ModelLedger

_M = TypeVar("_M", bound=torch.nn.Module)
_T = TypeVar("_T")


@contextmanager
def gpu_model(model: _M) -> Iterator[_M]:
    """Yield a model and free its parameter storage on exit."""
    try:
        yield model
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model.to("meta")
        cleanup_memory()


class AudioDecoder:
    """Compatibility wrapper mirroring legacy AudioDecoder behavior."""

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        warm: bool = False,
    ) -> None:
        self._warm = warm
        self._ledger = ModelLedger(
            dtype=dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            registry=registry or DummyRegistry(),
        )
        self._warm_decoder = None
        self._warm_vocoder = None
        if warm:
            self._warm_decoder = self._ledger.audio_decoder()
            self._warm_vocoder = self._ledger.vocoder()

    def __call__(self, latent: torch.Tensor) -> Audio:
        if self._warm and self._warm_decoder is not None and self._warm_vocoder is not None:
            return vae_decode_audio(latent, self._warm_decoder, self._warm_vocoder)

        with gpu_model(self._ledger.audio_decoder()) as decoder, gpu_model(self._ledger.vocoder()) as vocoder:
            return vae_decode_audio(latent, decoder, vocoder)


class AudioConditioner:
    """Compatibility wrapper mirroring legacy AudioConditioner behavior."""

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        warm: bool = False,
    ) -> None:
        self._warm = warm
        self._ledger = ModelLedger(
            dtype=dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            registry=registry or DummyRegistry(),
        )
        self._warm_encoder = self._ledger.audio_encoder() if warm else None

    def __call__(self, fn: Callable[[torch.nn.Module], _T]) -> _T:
        if self._warm and self._warm_encoder is not None:
            return fn(self._warm_encoder)
        with gpu_model(self._ledger.audio_encoder()) as encoder:
            return fn(encoder)


class PromptEncoder:
    """Minimal PromptEncoder compatibility shim for legacy scripts."""

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        warm: bool = False,
        use_bnb_4bit: bool = False,
        audio_only: bool = False,
    ) -> None:
        self._warm = warm
        self._ledger = ModelLedger(
            dtype=dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            registry=registry or DummyRegistry(),
        )
        self._warm_text_encoder = None
        self._warm_embeddings_processor = None
        self._use_bnb_4bit = use_bnb_4bit
        self._audio_only = audio_only

        if warm:
            self._warm_text_encoder = self._ledger.text_encoder()
            self._warm_embeddings_processor = self._ledger.gemma_embeddings_processor()

    def __call__(
        self,
        prompts: list[str],
        *,
        enhance_first_prompt: bool = False,
        enhance_prompt_image: str | None = None,
        enhance_prompt_seed: int = 42,
        streaming_prefetch_count: int | None = None,
    ):
        del streaming_prefetch_count
        del self._use_bnb_4bit
        del self._audio_only

        if self._warm and self._warm_text_encoder is not None and self._warm_embeddings_processor is not None:
            return self._encode(prompts, self._warm_text_encoder, self._warm_embeddings_processor,
                                enhance_first_prompt, enhance_prompt_image, enhance_prompt_seed)

        with gpu_model(self._ledger.text_encoder()) as text_encoder, gpu_model(
            self._ledger.gemma_embeddings_processor()
        ) as embeddings_processor:
            return self._encode(
                prompts,
                text_encoder,
                embeddings_processor,
                enhance_first_prompt,
                enhance_prompt_image,
                enhance_prompt_seed,
            )

    @staticmethod
    def _encode(prompts, text_encoder, embeddings_processor, enhance_first_prompt, enhance_prompt_image, enhance_prompt_seed):
        prompts_local = list(prompts)
        if enhance_first_prompt and prompts_local:
            prompts_local[0] = generate_enhanced_prompt(
                text_encoder, prompts_local[0], enhance_prompt_image, seed=enhance_prompt_seed
            )
        raw_outputs = [text_encoder.encode(p) for p in prompts_local]
        return [embeddings_processor.process_hidden_states(hs, mask) for hs, mask in raw_outputs]


_POSITIVE_ONLY_GUIDER = MultiModalGuider(
    params=MultiModalGuiderParams(cfg_scale=1.0, stg_scale=0.0, modality_scale=1.0),
)


def _ensure_guider(guider: MultiModalGuider | None) -> MultiModalGuider:
    return guider if guider is not None else _POSITIVE_ONLY_GUIDER


def _repeat_state(state: LatentState, n: int) -> LatentState:
    def _repeat(t: torch.Tensor) -> torch.Tensor:
        repeats = [1] * t.dim()
        repeats[0] = n
        return t.repeat(repeats)

    return LatentState(
        latent=_repeat(state.latent),
        denoise_mask=_repeat(state.denoise_mask),
        positions=_repeat(state.positions),
        clean_latent=_repeat(state.clean_latent),
        attention_mask=_repeat(state.attention_mask) if state.attention_mask is not None else None,
    )


def _timesteps_from_mask(denoise_mask: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """Match vendored ltx2 behavior for batched sigma tensors.

    When sigma is shaped (B,), reshape to (B, 1, ...) so broadcasting aligns
    with denoise_mask's batch dimension instead of token dimension.
    """
    if isinstance(sigma, torch.Tensor) and sigma.dim() == 1:
        sigma = sigma.view(-1, *([1] * (denoise_mask.dim() - 1)))
    return denoise_mask * sigma


def _modality_from_latent_state(
    state: LatentState,
    context: torch.Tensor,
    sigma: torch.Tensor,
    enabled: bool = True,
) -> Modality:
    return Modality(
        enabled=enabled,
        latent=state.latent,
        sigma=sigma,
        timesteps=_timesteps_from_mask(state.denoise_mask, sigma),
        positions=state.positions,
        context=context,
        context_mask=None,
        attention_mask=state.attention_mask,
    )


def _guided_denoise(  # noqa: PLR0913
    transformer: X0Model,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    sigma: torch.Tensor,
    video_guider: MultiModalGuider,
    audio_guider: MultiModalGuider,
    v_context: torch.Tensor | None,
    a_context: torch.Tensor | None,
    *,
    last_denoised_video: torch.Tensor | None,
    last_denoised_audio: torch.Tensor | None,
    step_index: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    v_skip = video_guider.should_skip_step(step_index)
    a_skip = audio_guider.should_skip_step(step_index)

    if v_skip and a_skip:
        return last_denoised_video, last_denoised_audio

    if video_state is not None and v_context is None:
        raise ValueError("v_context is required when video_state is provided")
    if audio_state is not None and a_context is None:
        raise ValueError("a_context is required when audio_state is provided")

    _pass = tuple[str, torch.Tensor | None, torch.Tensor | None, PerturbationConfig]
    passes: list[_pass] = [("cond", v_context, a_context, PerturbationConfig.empty())]

    if video_guider.do_unconditional_generation() or audio_guider.do_unconditional_generation():
        if video_guider.do_unconditional_generation() and video_guider.negative_context is None:
            raise ValueError("Negative context is required for unconditioned denoising")
        if audio_guider.do_unconditional_generation() and audio_guider.negative_context is None:
            raise ValueError("Negative context is required for unconditioned denoising")
        v_neg = video_guider.negative_context if video_guider.negative_context is not None else v_context
        a_neg = audio_guider.negative_context if audio_guider.negative_context is not None else a_context
        passes.append(("uncond", v_neg, a_neg, PerturbationConfig.empty()))

    stg_perturbations: list[Perturbation] = []
    if video_guider.do_perturbed_generation():
        stg_perturbations.append(
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=video_guider.params.stg_blocks)
        )
    if audio_guider.do_perturbed_generation():
        stg_perturbations.append(
            Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=audio_guider.params.stg_blocks)
        )
    if stg_perturbations:
        passes.append(("ptb", v_context, a_context, PerturbationConfig(stg_perturbations)))

    if video_guider.do_isolated_modality_generation() or audio_guider.do_isolated_modality_generation():
        passes.append(
            (
                "mod",
                v_context,
                a_context,
                PerturbationConfig(
                    [
                        Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                        Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
                    ]
                ),
            )
        )

    pass_names = [name for name, _, _, _ in passes]
    ptb_configs = [ptb for _, _, _, ptb in passes]
    n = len(passes)

    def _batched_sigma(state: LatentState) -> torch.Tensor:
        return sigma.expand(state.latent.shape[0] * n)

    batched_video = None
    if video_state is not None:
        v_context = torch.cat([vc for _, vc, _, _ in passes], dim=0)
        batched_video = _modality_from_latent_state(
            _repeat_state(video_state, n),
            v_context,
            _batched_sigma(video_state),
            enabled=not v_skip,
        )

    batched_audio = None
    if audio_state is not None:
        a_context = torch.cat([ac for _, _, ac, _ in passes], dim=0)
        batched_audio = _modality_from_latent_state(
            _repeat_state(audio_state, n),
            a_context,
            _batched_sigma(audio_state),
            enabled=not a_skip,
        )

    all_v, all_a = transformer(
        video=batched_video,
        audio=batched_audio,
        perturbations=BatchedPerturbationConfig(ptb_configs),
    )

    splits_v = list(all_v.chunk(n)) if all_v is not None else [0.0] * n
    splits_a = list(all_a.chunk(n)) if all_a is not None else [0.0] * n
    r = dict(zip(pass_names, zip(splits_v, splits_a, strict=True), strict=True))

    cond_v, cond_a = r["cond"]
    uncond_v, uncond_a = r.get("uncond", (0.0, 0.0))
    ptb_v, ptb_a = r.get("ptb", (0.0, 0.0))
    mod_v, mod_a = r.get("mod", (0.0, 0.0))

    denoised_video = last_denoised_video if v_skip else video_guider.calculate(cond_v, uncond_v, ptb_v, mod_v)
    denoised_audio = last_denoised_audio if a_skip else audio_guider.calculate(cond_a, uncond_a, ptb_a, mod_a)
    return denoised_video, denoised_audio


class SimpleDenoiser:
    def __init__(self, v_context: torch.Tensor | None, a_context: torch.Tensor | None) -> None:
        self.v_context = v_context
        self.a_context = a_context

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        sigma = sigmas[step_index]
        pos_video = _modality_from_latent_state(video_state, self.v_context, sigma) if video_state is not None else None
        pos_audio = _modality_from_latent_state(audio_state, self.a_context, sigma) if audio_state is not None else None
        return transformer(video=pos_video, audio=pos_audio, perturbations=None)


class GuidedDenoiser:
    def __init__(
        self,
        v_context: torch.Tensor | None,
        a_context: torch.Tensor | None,
        video_guider: MultiModalGuider | None = None,
        audio_guider: MultiModalGuider | None = None,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context
        self.video_guider = video_guider
        self.audio_guider = audio_guider
        self._last_denoised_video: torch.Tensor | None = None
        self._last_denoised_audio: torch.Tensor | None = None

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        denoised_video, denoised_audio = _guided_denoise(
            transformer=transformer,
            video_state=video_state,
            audio_state=audio_state,
            sigma=sigmas[step_index],
            video_guider=_ensure_guider(self.video_guider),
            audio_guider=_ensure_guider(self.audio_guider),
            v_context=self.v_context,
            a_context=self.a_context,
            last_denoised_video=self._last_denoised_video,
            last_denoised_audio=self._last_denoised_audio,
            step_index=step_index,
        )
        self._last_denoised_video = denoised_video
        self._last_denoised_audio = denoised_audio
        return denoised_video, denoised_audio


def _step_state(
    state: LatentState | None,
    denoised: torch.Tensor | None,
    stepper: DiffusionStepProtocol,
    sigmas: torch.Tensor,
    step_idx: int,
) -> LatentState | None:
    if state is None or denoised is None:
        return state
    denoised = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
    return replace(state, latent=stepper.step(state.latent, denoised, sigmas, step_idx))


def euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    stepper: DiffusionStepProtocol,
    transformer: X0Model,
    denoiser,
) -> tuple[LatentState | None, LatentState | None]:
    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        denoised_video, denoised_audio = denoiser(transformer, video_state, audio_state, sigmas, step_idx)
        video_state = _step_state(video_state, denoised_video, stepper, sigmas, step_idx)
        audio_state = _step_state(audio_state, denoised_audio, stepper, sigmas, step_idx)
    return (video_state, audio_state)


def heun_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    stepper: DiffusionStepProtocol,
    transformer: X0Model,
    denoiser,
) -> tuple[LatentState | None, LatentState | None]:
    n = len(sigmas) - 1
    for step_idx in tqdm(range(n)):
        sigma_curr = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]
        dt = sigma_next - sigma_curr

        denoised_video_1, denoised_audio_1 = denoiser(transformer, video_state, audio_state, sigmas, step_idx)
        video_pred = _step_state(video_state, denoised_video_1, stepper, sigmas, step_idx)
        audio_pred = _step_state(audio_state, denoised_audio_1, stepper, sigmas, step_idx)

        if step_idx == n - 1 or float(sigma_next) == 0.0:
            video_state, audio_state = video_pred, audio_pred
            continue

        denoised_video_2, denoised_audio_2 = denoiser(transformer, video_pred, audio_pred, sigmas, step_idx + 1)

        def _heun_step(state, state_pred, denoised_1, denoised_2):
            if state is None or denoised_1 is None or denoised_2 is None:
                return state
            d1 = post_process_latent(denoised_1, state.denoise_mask, state.clean_latent)
            d2 = post_process_latent(denoised_2, state.denoise_mask, state.clean_latent)
            v1 = to_velocity(state.latent, sigma_curr, d1)
            v2 = to_velocity(state_pred.latent, sigma_next, d2)
            v_avg = 0.5 * (v1 + v2)
            new_lat = (state.latent.to(torch.float32) + v_avg.to(torch.float32) * dt).to(state.latent.dtype)
            return replace(state, latent=new_lat)

        video_state = _heun_step(video_state, video_pred, denoised_video_1, denoised_video_2)
        audio_state = _heun_step(audio_state, audio_pred, denoised_audio_1, denoised_audio_2)

    return (video_state, audio_state)
