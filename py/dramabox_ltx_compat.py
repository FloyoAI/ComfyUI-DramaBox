from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
import gc
import logging
import types
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
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    module_ops_from_gemma_root,
)
from ltx_core.model.transformer import Modality, X0Model
from ltx_core.types import Audio, LatentState
from ltx_core.utils import find_matching_file, to_denoised, to_velocity
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    generate_enhanced_prompt,
    post_process_latent,
)
from ltx_pipelines.utils.model_ledger import ModelLedger

_M = TypeVar("_M", bound=torch.nn.Module)
_T = TypeVar("_T")
logger = logging.getLogger(__name__)


def _ensure_rope_factor_compat() -> None:
    """Install a vendored-style Gemma encoder mutator that avoids rope factor crashes."""
    try:
        from ltx_core.text_encoders.gemma.encoders import encoder_configurator as _enc_cfg
    except Exception:
        return

    if getattr(_enc_cfg, "_dramabox_vendored_gemma_compat", False):
        return

    def _resolve_rope_meta(config):
        rope_params = getattr(config, "rope_parameters", {}) or {}
        full = rope_params.get("full_attention", {}) if isinstance(rope_params, dict) else {}
        if not isinstance(full, dict):
            full = {}

        default_theta = getattr(config, "default_theta", {}) or {}
        rope_theta = full.get("rope_theta")
        if rope_theta is None and isinstance(rope_params, dict):
            rope_theta = rope_params.get("rope_theta")
        if rope_theta is None:
            rope_theta = default_theta.get("global", default_theta.get("local", 10000.0))

        rope_type = full.get("rope_type")
        if rope_type is None and isinstance(rope_params, dict):
            rope_type = rope_params.get("rope_type")
        if rope_type is None:
            rope_scaling = getattr(config, "rope_scaling", None)
            if isinstance(rope_scaling, dict):
                rope_type = rope_scaling.get("rope_type")
        if rope_type is None:
            rope_type = "default"

        factor = full.get("factor")
        if factor is None and isinstance(rope_params, dict):
            factor = rope_params.get("factor")
        if factor is None:
            factor = 1.0

        partial_rotary_factor = full.get("partial_rotary_factor")
        if partial_rotary_factor is None and isinstance(rope_params, dict):
            partial_rotary_factor = rope_params.get("partial_rotary_factor")
        if partial_rotary_factor is None:
            partial_rotary_factor = 1.0

        return {
            "rope_type": str(rope_type),
            "rope_theta": float(rope_theta),
            "factor": float(factor),
            "partial_rotary_factor": float(partial_rotary_factor),
        }

    def _compute_inv_freq(config):
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads

        rope_meta = _resolve_rope_meta(config)
        dim = int(head_dim * rope_meta["partial_rotary_factor"])
        if dim <= 0:
            dim = int(head_dim)
        if dim % 2 != 0:
            dim -= 1
        if dim <= 0:
            dim = int(head_dim)

        inv_freq = 1.0 / (
            rope_meta["rope_theta"]
            ** (torch.arange(0, dim, 2, dtype=torch.int64, device="cpu").to(dtype=torch.float) / dim)
        )

        if rope_meta["rope_type"] == "linear":
            inv_freq = inv_freq / max(rope_meta["factor"], 1e-8)

        return inv_freq, rope_meta

    def _create_and_populate_vendored(module):
        model = module.model
        vision_tower = model.model.vision_tower
        v_model = getattr(vision_tower, "vision_model", vision_tower)
        l_model = model.model.language_model

        config = model.config.text_config
        dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        rope_params = getattr(config, "rope_parameters", {}) or {}
        sliding = rope_params.get("sliding_attention", {}) if isinstance(rope_params, dict) else {}
        if not isinstance(sliding, dict):
            sliding = {}

        default_theta = getattr(config, "default_theta", {"local": 10000.0})
        base = getattr(config, "rope_local_base_freq", None)
        if base is None:
            base = sliding.get("rope_theta", default_theta.get("local", 10000.0))

        local_rope_freqs = 1.0 / (
            float(base) ** (torch.arange(0, dim, 2, dtype=torch.int64, device="cpu").to(dtype=torch.float) / dim)
        )

        inv_freqs, rope_meta = _compute_inv_freq(config)

        try:
            merged_rope_scaling = dict(getattr(config, "rope_scaling", {}) or {})
            merged_rope_scaling.setdefault("rope_type", rope_meta["rope_type"])
            merged_rope_scaling.setdefault("factor", rope_meta["factor"])
            merged_rope_scaling.setdefault("rope_theta", rope_meta["rope_theta"])
            setattr(config, "rope_scaling", merged_rope_scaling)
        except Exception:
            pass

        positions_length = len(v_model.embeddings.position_ids[0])
        position_ids = torch.arange(positions_length, dtype=torch.long, device="cpu").unsqueeze(0)
        v_model.embeddings.register_buffer("position_ids", position_ids)
        embed_scale = torch.tensor(model.config.text_config.hidden_size**0.5, device="cpu")
        l_model.embed_tokens.register_buffer("embed_scale", embed_scale)

        if hasattr(l_model, "rotary_emb_local") and l_model.rotary_emb_local is not None:
            l_model.rotary_emb_local.register_buffer("inv_freq", local_rope_freqs)

        if hasattr(l_model, "rotary_emb") and l_model.rotary_emb is not None:
            l_model.rotary_emb.register_buffer("inv_freq", inv_freqs)

        return module

    _enc_cfg.create_and_populate = _create_and_populate_vendored

    try:
        from ltx_core.loader.module_ops import ModuleOps
        import ltx_core.text_encoders.gemma as _gemma_pkg
        import ltx_pipelines.utils.model_ledger as _ledger_mod

        old_ops = getattr(_ledger_mod, "GEMMA_MODEL_OPS", None)
        if old_ops is None:
            old_ops = getattr(_gemma_pkg, "GEMMA_MODEL_OPS", None)
        if old_ops is None:
            old_ops = getattr(_enc_cfg, "GEMMA_MODEL_OPS", None)

        if old_ops is not None:
            new_ops = ModuleOps(
                name=getattr(old_ops, "name", "GemmaModel"),
                matcher=getattr(old_ops, "matcher"),
                mutator=_create_and_populate_vendored,
            )

            # Patch every place that may have captured GEMMA_MODEL_OPS by value.
            _enc_cfg.GEMMA_MODEL_OPS = new_ops
            _gemma_pkg.GEMMA_MODEL_OPS = new_ops
            _ledger_mod.GEMMA_MODEL_OPS = new_ops
    except Exception:
        pass

    _enc_cfg._dramabox_vendored_gemma_compat = True


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

    @staticmethod
    def _prepare_decode_latent(latent: torch.Tensor) -> torch.Tensor:
        """Return a decode-safe latent tensor.

        Some vocoder kernels require tensors with version counters. Latents
        produced under torch.inference_mode() do not track version counters,
        so clone outside inference mode when needed.
        """
        try:
            is_inference_tensor = bool(latent.is_inference())
        except Exception:
            is_inference_tensor = False

        if not is_inference_tensor:
            return latent

        with torch.inference_mode(False):
            return latent.detach().clone()

    def __call__(self, latent: torch.Tensor) -> Audio:
        latent_for_decode = self._prepare_decode_latent(latent)

        if self._warm and self._warm_decoder is not None and self._warm_vocoder is not None:
            with torch.inference_mode(False), torch.no_grad():
                return vae_decode_audio(latent_for_decode, self._warm_decoder, self._warm_vocoder)

        with gpu_model(self._ledger.audio_decoder()) as decoder, gpu_model(self._ledger.vocoder()) as vocoder:
            with torch.inference_mode(False), torch.no_grad():
                return vae_decode_audio(latent_for_decode, decoder, vocoder)


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
        self._dtype = dtype
        self._device = device
        self._use_bnb_4bit = use_bnb_4bit
        self._audio_only = audio_only
        _ensure_rope_factor_compat()

        registry = registry or DummyRegistry()
        module_ops = module_ops_from_gemma_root(gemma_root)
        model_folder = find_matching_file(gemma_root, "model*.safetensors").parent
        weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]

        self._text_encoder_builder = Builder(
            model_path=tuple(weight_paths),
            model_class_configurator=GemmaTextEncoderConfigurator,
            model_sd_ops=GEMMA_LLM_KEY_OPS,
            module_ops=(GEMMA_MODEL_OPS, *module_ops),
            registry=registry,
        )
        self._embeddings_processor_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            registry=registry,
        )

        self._warm_text_encoder = None
        self._warm_embeddings_processor = None

        if warm:
            if use_bnb_4bit:
                self._warm_text_encoder = self._load_bnb_4bit_encoder(gemma_root)
            else:
                self._warm_text_encoder = self._text_encoder_builder.build(
                    device=self._device,
                    dtype=self._dtype,
                ).eval()

            built_ep = self._embeddings_processor_builder.build(
                device=torch.device("cpu"),
                dtype=self._dtype,
            )
            self._warm_embeddings_processor = self._prepare_embeddings_processor(built_ep)

    def _prepare_embeddings_processor(self, embeddings_processor):
        if embeddings_processor is None:
            return None

        if not self._audio_only:
            return embeddings_processor.to(device=self._device, dtype=self._dtype).eval()

        ep = embeddings_processor
        freed = 0

        if getattr(ep, "video_connector", None) is not None:
            try:
                freed += sum(
                    p.numel() * p.element_size()
                    for p in ep.video_connector.parameters()
                    if not p.is_meta
                )
            except Exception:
                pass
            del ep.video_connector
            ep.video_connector = None

        fe = getattr(ep, "feature_extractor", None)
        if fe is not None and getattr(fe, "video_aggregate_embed", None) is not None:
            try:
                freed += sum(
                    p.numel() * p.element_size()
                    for p in fe.video_aggregate_embed.parameters()
                    if not p.is_meta
                )
            except Exception:
                pass
            out_features = getattr(fe.video_aggregate_embed, "out_features", 1)
            del fe.video_aggregate_embed

            class _DummyVideoEmbed(torch.nn.Module):
                def __init__(self, out_f):
                    super().__init__()
                    self.out_features = out_f

                def forward(self, x):
                    return torch.zeros(
                        x.shape[0],
                        x.shape[1],
                        self.out_features,
                        device=x.device,
                        dtype=x.dtype,
                    )

            fe.video_aggregate_embed = _DummyVideoEmbed(out_features)

        if fe is not None and getattr(fe, "audio_aggregate_embed", None) is not None:
            def _forward_audio_only(self, hidden_states, attention_mask, padding_side="left"):
                from ltx_core.text_encoders.gemma.feature_extractor import (
                    _rescale_norm,
                    norm_and_concat_per_token_rms,
                )

                encoded = (
                    torch.stack(hidden_states, dim=-1)
                    if isinstance(hidden_states, (list, tuple))
                    else hidden_states
                )
                normed = norm_and_concat_per_token_rms(encoded, attention_mask).to(encoded.dtype)
                a_dim = self.audio_aggregate_embed.out_features
                audio = self.audio_aggregate_embed(_rescale_norm(normed, a_dim, self.embedding_dim))
                video = audio.new_zeros((audio.shape[0], audio.shape[1], 1))
                return video, audio

            fe.forward = types.MethodType(_forward_audio_only, fe)

        def _audio_only_create(video_features, audio_features, additive_attention_mask, _ep=ep):
            m = additive_attention_mask
            while m.dim() > 2:
                m = m[:, 0]

            binary_mask = (m >= -1.0).to(torch.int64)
            audio_encoded = None
            if _ep.audio_connector is not None:
                audio_encoded, _ = _ep.audio_connector(audio_features, additive_attention_mask)

            return video_features, audio_encoded, binary_mask

        ep.create_embeddings = _audio_only_create

        ep = ep.to(device=self._device, dtype=self._dtype).eval()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if freed:
            logger.info("[DramaBox] Audio-only mode: freed video components, saved %.2f GB VRAM", freed / 1e9)

        return ep

    def _load_bnb_4bit_encoder(self, gemma_root: str):
        import json
        import os
        from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration
        from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
        from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer

        try:
            import bitsandbytes as _bnb

            _bnb.functional.get_4bit_type("nf4")
        except Exception as exc:
            raise RuntimeError(f"bitsandbytes not functional: {exc}") from exc

        prequantized = False
        cfg_path = os.path.join(gemma_root, "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                prequantized = "quantization_config" in cfg
            except Exception:
                prequantized = False

        from_kwargs = {
            "device_map": str(self._device),
            "torch_dtype": self._dtype,
        }
        if not prequantized:
            from_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self._dtype,
            )

        hf_model = Gemma3ForConditionalGeneration.from_pretrained(gemma_root, **from_kwargs)
        tokenizer = LTXVGemmaTokenizer(
            str(find_matching_file(gemma_root, "tokenizer.model").parent),
            1024,
        )
        return GemmaTextEncoder(model=hf_model, tokenizer=tokenizer, dtype=self._dtype)

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

        _ensure_rope_factor_compat()

        if self._warm and self._warm_text_encoder is not None and self._warm_embeddings_processor is not None:
            return self._encode(prompts, self._warm_text_encoder, self._warm_embeddings_processor,
                                enhance_first_prompt, enhance_prompt_image, enhance_prompt_seed)

        with gpu_model(self._text_encoder_builder.build(device=self._device, dtype=self._dtype).eval()) as text_encoder, gpu_model(
            self._prepare_embeddings_processor(
                self._embeddings_processor_builder.build(device=torch.device("cpu"), dtype=self._dtype)
            )
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
    separate_passes: bool = False,
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

    def _contexts_batch_compatible() -> bool:
        if video_state is not None:
            v_contexts = [vc for _, vc, _, _ in passes]
            if any(vc is None for vc in v_contexts):
                return False
            ref_shape = v_contexts[0].shape[1:]
            if any(vc.shape[1:] != ref_shape for vc in v_contexts[1:]):
                return False
        if audio_state is not None:
            a_contexts = [ac for _, _, ac, _ in passes]
            if any(ac is None for ac in a_contexts):
                return False
            ref_shape = a_contexts[0].shape[1:]
            if any(ac.shape[1:] != ref_shape for ac in a_contexts[1:]):
                return False
        return True

    use_separate_passes = separate_passes or not _contexts_batch_compatible()

    if use_separate_passes:
        r: dict[str, tuple[torch.Tensor | float, torch.Tensor | float]] = {}
        for pass_name, pass_v_context, pass_a_context, pass_ptb in passes:
            pass_video = None
            if video_state is not None:
                pass_video = _modality_from_latent_state(
                    video_state,
                    pass_v_context,
                    sigma,
                    enabled=not v_skip,
                )

            pass_audio = None
            if audio_state is not None:
                pass_audio = _modality_from_latent_state(
                    audio_state,
                    pass_a_context,
                    sigma,
                    enabled=not a_skip,
                )

            out_v, out_a = transformer(
                video=pass_video,
                audio=pass_audio,
                perturbations=BatchedPerturbationConfig([pass_ptb]),
            )
            r[pass_name] = (out_v if out_v is not None else 0.0, out_a if out_a is not None else 0.0)

        cond_v, cond_a = r["cond"]
        uncond_v, uncond_a = r.get("uncond", (0.0, 0.0))
        ptb_v, ptb_a = r.get("ptb", (0.0, 0.0))
        mod_v, mod_a = r.get("mod", (0.0, 0.0))

        denoised_video = last_denoised_video if v_skip else video_guider.calculate(cond_v, uncond_v, ptb_v, mod_v)
        denoised_audio = last_denoised_audio if a_skip else audio_guider.calculate(cond_a, uncond_a, ptb_a, mod_a)
        return denoised_video, denoised_audio

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
        separate_passes: bool = False,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context
        self.video_guider = video_guider
        self.audio_guider = audio_guider
        self.separate_passes = separate_passes
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
            separate_passes=self.separate_passes,
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
