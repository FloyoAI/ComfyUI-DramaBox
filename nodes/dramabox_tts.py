"""
DramaBox TTS — ComfyUI node that runs the full LTX audio diffusion pipeline.

DramaBox weights and Gemma text encoder auto-download from HuggingFace into
    ComfyUI/custom_nodes/ComfyUI-DramaBox/models/   (~30 GB total, gitignored)
on first use and are cached locally afterwards.

Inputs
------


prompt          Scene description with quoted dialogue
seed            Reproducibility seed
steps           Denoising steps (default 30)
cfg_scale       Classifier-free guidance (default 2.5)
stg_scale       Skip-token guidance (default 1.5)
voice_ref       Optional AUDIO — reference voice to clone
negative_prompt What the model should avoid
options         Optional DRAMABOX_OPTIONS from DramaBox Options node

Outputs
-------
audio           ComfyUI AUDIO: {"waveform": Tensor[B,C,S], "sample_rate": int}
info            Human-readable stats string
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# ── ensure bundled src/ and ltx2/ are importable ─────────────────────────────
_NODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("src", "ltx2"):
    _p = os.path.join(_NODE_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Node-local models folder (gitignored) — DramaBox weights land here
_MODELS_DIR = os.path.join(_NODE_DIR, "models")
os.makedirs(_MODELS_DIR, exist_ok=True)

_DEFAULT_NEG = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "robotic voice, echo, background noise, off-sync audio, repetitive speech"
)


# ─────────────────────────────────────────────────────────────────────────────
# Model cache — keyed by str(device)
# Models are wrapped in comfy.model_patcher.ModelPatcher so ComfyUI can evict
# them to CPU when other nodes need VRAM and reload them when we run.
# ─────────────────────────────────────────────────────────────────────────────

_LOADED_MODELS: dict[str, dict] = {}


def _estimate_bytes(module: torch.nn.Module) -> int:
    """Rough VRAM estimate for a module (parameters + buffers)."""
    total = 0
    for t in (*module.parameters(), *module.buffers()):
        if not getattr(t, 'is_meta', False):
            total += t.numel() * t.element_size()
    return max(total, 1)


class _HFModelWrapper(torch.nn.Module):
    """Thin nn.Module shell around a HuggingFace model.

    HF models (e.g. Gemma3ForConditionalGeneration) define ``device`` as a
    read-only property.  ComfyUI's ModelPatcher sets ``model.device = ...``
    during load/offload which raises AttributeError on such models.
    Wrapping the HF model as a registered sub-module solves this:
    - ModelPatcher sets .device on *this* wrapper (which has no such property)
    - .to(device) / parameter moves still cascade to the inner HF model
      because it is registered via self.inner
    """
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner  # registered submodule — moves cascade automatically

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


def _make_patcher(module: torch.nn.Module, load_device: torch.device, wrap_hf: bool = False):
    """Wrap *module* in a ComfyUI ModelPatcher so the memory manager tracks it.

    Set ``wrap_hf=True`` for HuggingFace models that have a read-only
    ``device`` property so ModelPatcher can do its device-assignment bookkeeping.
    """
    import comfy.model_management as _mm
    import comfy.model_patcher
    target = _HFModelWrapper(module) if wrap_hf else module
    return comfy.model_patcher.ModelPatcher(
        target,
        load_device=load_device,
        offload_device=_mm.unet_offload_device(),
        size=_estimate_bytes(module),
    )


def _load_models(device) -> dict:
    """Load (or return cached) all DramaBox model components as ModelPatchers."""
    cache_key = str(device)
    if cache_key in _LOADED_MODELS:
        logger.info("[DramaBox] Using cached models.")
        return _LOADED_MODELS[cache_key]

    torch_dtype = torch.bfloat16

    # -- DramaBox weights (auto-download into node models/) -------------------
    from model_downloader import get_model_path, get_gemma_path
    logger.info("[DramaBox] Resolving DramaBox weights (downloads on first use)…")
    ckpt_transformer = get_model_path("transformer",      cache_dir=_MODELS_DIR)
    ckpt_audio       = get_model_path("audio_components", cache_dir=_MODELS_DIR)
    logger.info("[DramaBox] transformer  : %s", ckpt_transformer)
    logger.info("[DramaBox] audio comps  : %s", ckpt_audio)

    # -- Gemma directory (auto-download unsloth/gemma-3-12b-it-bnb-4bit) ------
    gemma_root = get_gemma_path(cache_dir=_MODELS_DIR)
    logger.info("[DramaBox] gemma        : %s", gemma_root)

    from ltx_pipelines.utils.blocks import PromptEncoder, AudioConditioner, AudioDecoder

    # -- 1. Prompt encoder ----------------------------------------------------
    logger.info("[DramaBox] Loading PromptEncoder…")
    prompt_encoder = PromptEncoder(
        checkpoint_path=ckpt_audio,
        gemma_root=gemma_root,
        dtype=torch_dtype,
        device=device,
        warm=True,
        use_bnb_4bit=True,
        audio_only=True,
    )

    # -- 2. Audio VAE encoder (for voice reference conditioning) --------------
    logger.info("[DramaBox] Loading AudioConditioner…")
    audio_conditioner = AudioConditioner(
        checkpoint_path=ckpt_audio,
        dtype=torch_dtype,
        device=device,
        warm=True,
    )

    # -- 3. Transformer (LTX audio-only DiT) ----------------------------------
    logger.info("[DramaBox] Loading Transformer…")
    from safetensors import safe_open
    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.loader.sd_ops import SDOps
    from ltx_core.model.transformer.model import LTXModel, LTXModelType
    from ltx_core.model.transformer.rope import LTXRopeType
    from ltx_core.model.transformer.text_projection import create_caption_projection
    from ltx_core.model.transformer.attention import AttentionFunction
    from ltx_core.model.model_protocol import ModelConfigurator

    with safe_open(ckpt_transformer, framework="pt") as f:
        config = json.loads(f.metadata()["config"])

    class _AudioOnlyConfigurator(ModelConfigurator[LTXModel]):
        @classmethod
        def from_config(cls, cfg):
            _t = cfg.get("transformer", {})
            cp = None
            if not _t.get("caption_proj_before_connector", False):
                with torch.device("meta"):
                    cp = create_caption_projection(_t, audio=True)
            return LTXModel(
                model_type=LTXModelType.AudioOnly,
                audio_num_attention_heads=_t.get("audio_num_attention_heads", 32),
                audio_attention_head_dim=_t.get("audio_attention_head_dim", 64),
                audio_in_channels=_t.get("audio_in_channels", 128),
                audio_out_channels=_t.get("audio_out_channels", 128),
                num_layers=_t.get("num_layers", 48),
                audio_cross_attention_dim=_t.get("audio_cross_attention_dim", 2048),
                norm_eps=_t.get("norm_eps", 1e-6),
                attention_type=AttentionFunction(_t.get("attention_type", "default")),
                positional_embedding_theta=10000.0,
                audio_positional_embedding_max_pos=[20.0],
                timestep_scale_multiplier=_t.get("timestep_scale_multiplier", 1000),
                use_middle_indices_grid=_t.get("use_middle_indices_grid", True),
                rope_type=LTXRopeType(_t.get("rope_type", "interleaved")),
                double_precision_rope=_t.get("frequencies_precision", False) == "float64",
                apply_gated_attention=_t.get("apply_gated_attention", False),
                audio_caption_projection=cp,
                cross_attention_adaln=_t.get("cross_attention_adaln", False),
            )

    sd_ops = (
        SDOps("AO")
        .with_matching(prefix="model.diffusion_model.")
        .with_replacement("model.diffusion_model.", "")
    )
    transformer = (
        Builder(
            model_path=ckpt_transformer,
            model_class_configurator=_AudioOnlyConfigurator,
            model_sd_ops=sd_ops,
            registry=DummyRegistry(),
        )
        .build(device=device, dtype=torch_dtype)
        .to(device)
        .eval()
    )
    n_params = sum(p.numel() for p in transformer.parameters()) / 1e9
    logger.info("[DramaBox] Transformer: %.1fB params", n_params)

    # -- 4. Audio VAE decoder + vocoder ----------------------------------------
    logger.info("[DramaBox] Loading AudioDecoder…")
    audio_decoder = AudioDecoder(
        checkpoint_path=ckpt_audio,
        dtype=torch_dtype,
        device=device,
        warm=True,
    )

    # ── Register all nn.Modules with ComfyUI model management ────────────────
    # ComfyUI can now offload our models to CPU when other nodes need VRAM
    # and reload them via load_models_gpu() at the start of each generate().
    patchers = []

    # Gemma text encoder (HF model inside GemmaTextEncoder wrapper)
    # Uses wrap_hf=True because Gemma3ForConditionalGeneration.device is read-only.
    _gemma_nn = getattr(prompt_encoder._warm_text_encoder, "model", None)
    if _gemma_nn is not None:
        patchers.append(_make_patcher(_gemma_nn, device, wrap_hf=True))

    # Embeddings processor
    patchers.append(_make_patcher(prompt_encoder._warm_embeddings_processor, device))

    # Audio VAE encoder
    patchers.append(_make_patcher(audio_conditioner._warm_encoder, device))

    # Transformer
    patchers.append(_make_patcher(transformer, device))

    # Audio VAE decoder + vocoder
    patchers.append(_make_patcher(audio_decoder._warm_decoder, device))
    patchers.append(_make_patcher(audio_decoder._warm_vocoder, device))

    model = {
        "patchers":          patchers,
        "prompt_encoder":    prompt_encoder,
        "audio_conditioner": audio_conditioner,
        "transformer":       transformer,
        "audio_decoder":     audio_decoder,
        "device":            device,
        "dtype":             torch_dtype,
    }
    _LOADED_MODELS[cache_key] = model
    logger.info("[DramaBox] All components loaded and registered with ComfyUI model management.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _auto_rescale(cfg: float) -> float:
    """CFG-aware std-rescale schedule (prevents clipping at high CFG)."""
    if cfg <= 2.0:
        return 0.0
    if cfg <= 3.0:
        return 0.6 * (cfg - 2.0)
    if cfg <= 4.0:
        return 0.6 + 0.2 * (cfg - 3.0)
    if cfg <= 8.0:
        return 0.8
    return min(1.0, 0.8 + 0.1 * (cfg - 8.0))


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class DramaBoxTTS:
    """
    Generate expressive, voice-clonable speech using the DramaBox LTX diffusion model.

    All weights (DramaBox + Gemma) download automatically on first use
    and are cached in ComfyUI-DramaBox/models/.
    """

    CATEGORY = "DramaBox"
    DESCRIPTION = (
        "Generate expressive TTS with optional voice cloning.\n"
        "All weights (DramaBox + Gemma) auto-download on first use.\n"
        "Optionally connect any audio source as voice_ref to clone a voice."
    )
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": True},
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Enter prompt...",
                        "tooltip": (
                            "Scene description with quoted dialogue.\n"
                            "Example: 'A woman speaks warmly, \"Hello there.\"'"
                        ),
                    },
                ),
            },
            "optional": {
                "voice_ref": (
                    "AUDIO",
                    {"tooltip": "Reference voice for cloning. 5–15 s of clean speech works best."},
                ),
                "options": (
                    "DRAMABOX_OPTIONS",
                    {"tooltip": "Connect DramaBox Options for cfg, steps, duration, and more."},
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed

    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def generate(
        self,
        seed: int = 0,
        prompt: str = "",
        voice_ref=None,
        options: dict | None = None,
    ):
        import comfy.model_management as mm
        device = mm.get_torch_device()

        # ── Load (or retrieve cached) model components ───────────────────
        # _load_models() builds ModelPatchers on first call; subsequent calls
        # return the cache. load_models_gpu() then ensures all our modules are
        # on the target device — ComfyUI will offload other models if needed.
        model = _load_models(device)
        mm.load_models_gpu(model["patchers"])

        opts = options or {}
        ref_duration: float  = opts.get("ref_duration", 10.0)
        gen_duration: float  = opts.get("gen_duration", 0.0)
        duration_mult: float = opts.get("duration_multiplier", 1.1)
        steps: int           = int(opts.get("steps", 30))
        cfg_scale: float     = float(opts.get("cfg_scale", 2.5))
        stg_scale: float     = float(opts.get("stg_scale", 1.5))
        negative_prompt: str = opts.get("negative_prompt", _DEFAULT_NEG)
        rescale_raw          = opts.get("rescale_scale", "auto")
        rescale_scale: float = (
            _auto_rescale(cfg_scale) if rescale_raw == "auto" else float(rescale_raw)
        )

        prompt_enc  = model["prompt_encoder"]
        audio_cond  = model["audio_conditioner"]
        transformer = model["transformer"]
        decoder     = model["audio_decoder"]
        torch_dtype = model["dtype"]

        # ── imports from bundled ltx2 / src ─────────────────────────────
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.components.patchifiers import AudioPatchifier
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        from ltx_core.components.schedulers import LTX2Scheduler
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.model.transformer.model import X0Model
        from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
        from ltx_core.tools import AudioLatentTools
        from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
        from ltx_pipelines.utils.denoisers import GuidedDenoiser
        from ltx_pipelines.utils.samplers import euler_denoising_loop
        from audio_conditioning import AudioConditionByReferenceLatent
        from inference import estimate_speech_duration

        t_total = time.time()
        patchifier = AudioPatchifier(patch_size=1)

        # ── 1. Compute target shape ──────────────────────────────────────
        if gen_duration and gen_duration > 0:
            gen_dur = float(gen_duration)
        else:
            gen_dur = round(estimate_speech_duration(prompt) * duration_mult, 1)
        fps = 25.0
        n_frames = int(round(gen_dur * fps)) + 1
        n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1
        pixel_shape  = VideoPixelShape(batch=1, frames=n_frames, height=64, width=64, fps=fps)
        target_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        audio_tools  = AudioLatentTools(patchifier=patchifier, target_shape=target_shape)
        logger.info("[DramaBox] target: %.1fs → %d frames", gen_dur, n_frames)

        # ── 2. Initial latent state ──────────────────────────────────────
        state = audio_tools.create_initial_state(device=device, dtype=torch_dtype)

        # ── 3. Voice reference conditioning ─────────────────────────────
        if voice_ref is not None:
            try:
                from ltx_pipelines.utils.media_io import decode_audio_from_file
                import tempfile, soundfile as sf

                waveform = voice_ref["waveform"]          # [B, C, S]
                sr_in: int = voice_ref["sample_rate"]
                wav = waveform[0].cpu().float().numpy()   # [C, S]
                if wav.ndim == 2:
                    wav = wav.T                           # → [S, C]

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, wav, sr_in)
                    tmp_path = tmp.name

                try:
                    voice = decode_audio_from_file(tmp_path, device, 0.0, ref_duration)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

                if voice is not None:
                    w = voice.waveform
                    if w.dim() == 2:
                        w = w.repeat(2, 1) if w.shape[0] == 1 else w
                        w = w.unsqueeze(0)
                    elif w.dim() == 3 and w.shape[1] == 1:
                        w = w.repeat(1, 2, 1)
                    n_ref = int(ref_duration * voice.sampling_rate)
                    if w.shape[-1] < n_ref:
                        w = w.repeat(1, 1, (n_ref // w.shape[-1]) + 1)
                    w = w[..., :n_ref]
                    peak = w.abs().max()
                    if peak > 0:
                        w = w * (10 ** (-4.0 / 20) / peak)
                    voice = Audio(waveform=w, sampling_rate=voice.sampling_rate)
                    ref_latent = audio_cond(lambda enc: vae_encode_audio(voice, enc, None))
                    cond = AudioConditionByReferenceLatent(
                        latent=ref_latent.to(device, torch_dtype), strength=1.0
                    )
                    state = cond.apply_to(state, audio_tools)
                    logger.info("[DramaBox] Voice reference encoded.")
            except Exception as exc:
                logger.warning("[DramaBox] voice_ref failed (%s) — running without it", exc)

        # ── 4. Add noise ────────────────────────────────────────────────
        gen = torch.Generator(device=device).manual_seed(seed)
        state = GaussianNoiser(generator=gen)(state, noise_scale=1.0)

        # ── 5. Encode text prompts ───────────────────────────────────────
        logger.info("[DramaBox] Encoding prompts…")
        prompts = [prompt, negative_prompt] if cfg_scale > 1.0 else [prompt]
        ctx = prompt_enc(prompts, streaming_prefetch_count=None)
        a_ctx     = ctx[0].audio_encoding
        a_ctx_neg = ctx[1].audio_encoding if cfg_scale > 1.0 else None

        # ── 6. Build denoiser ────────────────────────────────────────────
        guider = MultiModalGuider(
            params=MultiModalGuiderParams(
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
                stg_blocks=[29],
                rescale_scale=rescale_scale,
                modality_scale=1.0,
            ),
            negative_context=a_ctx_neg,
        )
        denoiser = GuidedDenoiser(
            v_context=None,
            a_context=a_ctx,
            video_guider=None,
            audio_guider=guider,
        )

        # ── 7. Diffusion sampling ────────────────────────────────────────
        logger.info(
            "[DramaBox] Denoising (%d steps, cfg=%.1f, stg=%.1f)…",
            steps, cfg_scale, stg_scale,
        )
        sigmas = LTX2Scheduler().execute(steps=steps, latent=state.latent).to(device)
        x0 = X0Model(transformer)
        _, audio_state = euler_denoising_loop(
            sigmas=sigmas,
            video_state=None,
            audio_state=state,
            stepper=EulerDiffusionStep(),
            transformer=x0,
            denoiser=denoiser,
        )

        # ── 8. Unpatchify + end-of-clip silence-prior fix ─────────────────
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        latent = audio_state.latent
        # The DiT bakes a silence prior at frame 513; interpolate across it
        # for longer generations to avoid an audible glitch.
        if latent.shape[2] > 513:
            patched = latent.clone()
            for f in (512, 513):
                t = (f - 511) / 3
                patched[:, :, f, :] = (
                    (1 - t) * latent[:, :, 511, :] + t * latent[:, :, 514, :]
                )
            latent = patched

        # ── 9. Decode latents → waveform ────────────────────────────────
        logger.info("[DramaBox] Decoding…")
        decoded = decoder(latent)

        # ── 10. Return as ComfyUI AUDIO ──────────────────────────────────
        waveform = decoded.waveform.cpu().float()
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)   # [B=1, C, S]
        sample_rate = decoded.sampling_rate
        audio_dur = waveform.shape[-1] / sample_rate
        elapsed = time.time() - t_total

        info = (
            f"Generated {audio_dur:.1f}s of audio in {elapsed:.1f}s "
            f"| {sample_rate} Hz | {steps} steps "
            f"| cfg={cfg_scale} stg={stg_scale} | seed={seed}"
        )
        logger.info("[DramaBox] %s", info)

        return ({"waveform": waveform, "sample_rate": sample_rate}, info)


# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "DramaBoxTTS": DramaBoxTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DramaBoxTTS": "DramaBox TTS",
}
