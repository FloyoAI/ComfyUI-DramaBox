"""
DramaBox TTS — ComfyUI node that runs the full LTX audio diffusion pipeline.

DramaBox weights (~8.5 GB) download automatically from HuggingFace on first use into:

    ComfyUI/models/dramabox/
        dramabox-dit-v1.safetensors
        dramabox-audio-components.safetensors
        silence_latent_frame.pt

Gemma 3 12B weights are loaded from ComfyUI/models/text_encoders/ (fp8 safetensors)
or from a full HuggingFace snapshot in ComfyUI/models/dramabox/.
Existing weights in the old node-local models/ folder are migrated automatically.
Once present, files are loaded directly — no HuggingFace API calls on startup.

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

try:
    from server import PromptServer as _PromptServer
except Exception:
    _PromptServer = None

logger = logging.getLogger(__name__)


class _SuppressAudioOnlyInitNoise(logging.Filter):
    """Filter known harmless missing-init logs for stripped audio-only branches."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage().lower()
        except Exception:
            return True

        if "initialized parameters or buffers" not in msg:
            return True

        if "feature_extractor.video_aggregate_embed" in msg:
            return False
        if "video_connector." in msg:
            return False

        return True


def _install_noise_filters() -> None:
    filt = _SuppressAudioOnlyInitNoise()
    targets = (
        logging.getLogger(),
        logging.getLogger("ltx_core.loader.single_gpu_model_builder"),
        logging.getLogger("ltx_core.loader.module_ops"),
    )
    for target in targets:
        target.addFilter(filt)


_install_noise_filters()

# ── local import bootstrap ────────────────────────────────────────────────────
_NODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Always make local py importable for node-local helpers.
_SRC_DIR = os.path.join(_NODE_DIR, "py")
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from .dramabox_clip import DramaBoxTextEncoderLoader  # noqa: E402


def _get_models_dir() -> str:
    """Resolve ComfyUI models root at runtime.

    This avoids locking in a node-local fallback when folder_paths is not yet
    importable during early module import.
    """
    try:
        import folder_paths as _folder_paths
        models_dir = _folder_paths.models_dir
    except Exception:
        models_dir = os.path.join(_NODE_DIR, "models")

    os.makedirs(os.path.join(models_dir, "dramabox"), exist_ok=True)
    return models_dir

_DEFAULT_NEG = (
    "worst quality, inconsistent, robotic, distorted, noise, static, muffled, unclear, unnatural, monotone"
)
_AUTO_DURATION_SAFETY_PAD_SEC = 0.8


# ─────────────────────────────────────────────────────────────────────────────
# Model caches — keyed by str(device)
#
# _LOADED_MODELS:        audio VAE (conditioner + decoder) + transformer
# _LOADED_TEXT_ENCODER:  Gemma + embeddings processor (text encoder only)
#
# Both caches are wrapped in ModelPatcher so ComfyUI moves them GPU↔CPU
# automatically per generation stage.  The text encoder lives in a separate
# cache so an external DRAMABOX_CLIP can be used without reloading everything.
# ─────────────────────────────────────────────────────────────────────────────

_LOADED_MODELS: dict[str, dict] = {}
_LOADED_TEXT_ENCODER: dict[str, dict] = {}
_OG_SERVER_CACHE: dict[str, object] = {}
_DRAMABOX_WRAPPER_COMPAT_APPLIED = False


_GEMMA_FP4_FILENAME  = "gemma_3_12B_it_fp4_mixed.safetensors"
_GEMMA_FP4_REPO      = "Comfy-Org/ltx-2"
_GEMMA_FP4_REPO_PATH = "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"


def _get_dramabox_setting(key, default=None):
    """Read a value from ComfyUI's persisted user settings JSON."""
    try:
        import json
        import folder_paths as _fp
        settings_path = os.path.join(_fp.base_path, "user", "default", "comfy.settings.json")
        if os.path.isfile(settings_path):
            with open(settings_path, encoding="utf-8") as _f:
                return json.load(_f).get(key, default)
    except Exception:
        pass
    return default


def _get_dramabox_bool_setting(key: str, default: bool = False) -> bool:
    """Read a boolean DramaBox user setting with tolerant coercion."""
    v = _get_dramabox_setting(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def _offload_clip_patcher_to_cpu(clip_obj) -> None:
    """Offload a CLIP patcher to CPU and remove it from loaded-GPU tracking."""
    import comfy.model_management as mm

    patcher = getattr(clip_obj, "patcher", None)
    if patcher is None:
        return

    try:
        patcher.unpatch_model(patcher.offload_device)
    except Exception as e:
        logger.debug("[DramaBox] clip patcher offload failed: %s", e)

    mm.current_loaded_models[:] = [
        m for m in mm.current_loaded_models if id(m.model) != id(patcher)
    ]
    mm.soft_empty_cache()

def _find_or_download_gemma_path():
    """Return a local path to the default Gemma text encoder.

    Resolution order:
      1. User preference set in ComfyUI Settings → DramaBox → Default Text Encoder.
      2. ``gemma_3_12B_it_fp4_mixed.safetensors`` in any ``text_encoders`` folder.
      3. Auto-download ``gemma_3_12B_it_fp4_mixed.safetensors`` from ``Comfy-Org/ltx-2``.

    To use a different model per-workflow, connect a DramaBox CLIP Loader node.
    """
    import shutil
    _fp = None
    try:
        import folder_paths as _fp
        te_dirs = _fp.get_folder_paths("text_encoders")
    except Exception:
        te_dirs = [os.path.join(_get_models_dir(), "text_encoders")]

    gguf_dirs = []
    if _fp is not None:
        for key in ("clip_gguf", "clip"):
            try:
                gguf_dirs.extend(_fp.get_folder_paths(key))
            except Exception:
                pass

    # Keep order stable and remove duplicates.
    all_dirs = list(dict.fromkeys(te_dirs + gguf_dirs))

    # 1. User preference from ComfyUI settings
    preferred = (_get_dramabox_setting("DramaBox.defaultTextEncoder") or "").strip()
    if preferred:
        stem, ext = os.path.splitext(preferred)
        preferred_names = [preferred] if ext else [stem + ".safetensors", stem + ".gguf"]

        import glob as _glob
        for preferred_name in preferred_names:
            # Exact file-key lookup first.
            if _fp is not None:
                for key in ("text_encoders", "clip_gguf", "clip"):
                    try:
                        candidate = _fp.get_full_path(key, preferred_name)
                    except Exception:
                        candidate = None
                    if candidate and os.path.isfile(candidate):
                        return candidate

            # Fallback recursive search in known directories.
            for folder in all_dirs:
                candidate = os.path.join(folder, preferred_name)
                if os.path.isfile(candidate):
                    return candidate
                matches = _glob.glob(os.path.join(folder, "**", preferred_name), recursive=True)
                if matches:
                    return matches[0]

        print(
            f"[DramaBox] Warning: preferred text encoder '{preferred}' not found "
            "(checked safetensors/gguf) — falling back to default."
        )

    # 2. Default fp4 mixed file
    import glob as _glob
    for folder in te_dirs:
        exact = os.path.join(folder, _GEMMA_FP4_FILENAME)
        if os.path.isfile(exact):
            return exact
        matches = _glob.glob(os.path.join(folder, "**", _GEMMA_FP4_FILENAME), recursive=True)
        if matches:
            return matches[0]
    target_dir = te_dirs[0]
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, _GEMMA_FP4_FILENAME)
    print(f"[DramaBox] Gemma not found — downloading {_GEMMA_FP4_FILENAME} from {_GEMMA_FP4_REPO} …")
    from huggingface_hub import snapshot_download
    import logging as _logging
    _httpx_log = _logging.getLogger("httpx")
    _prev = _httpx_log.level
    _httpx_log.setLevel(_logging.WARNING)
    try:
        snapshot_download(
            repo_id=_GEMMA_FP4_REPO,
            allow_patterns=[_GEMMA_FP4_REPO_PATH],
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN"),
        )
    finally:
        _httpx_log.setLevel(_prev)
    # snapshot_download places the file at local_dir/repo_path; move it flat
    dl_path = os.path.join(target_dir, _GEMMA_FP4_REPO_PATH)
    if os.path.isfile(dl_path) and os.path.abspath(dl_path) != os.path.abspath(target_path):
        shutil.move(dl_path, target_path)
    # Remove leftover subdirs created by huggingface_hub
    for leftover in ("split_files", ".cache"):
        d = os.path.join(target_dir, leftover)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    if os.path.isfile(target_path):
        print(f"[DramaBox] Downloaded: {target_path}")
        return target_path
    raise FileNotFoundError(
        f"[DramaBox] Failed to download {_GEMMA_FP4_FILENAME}. "
        "Place it manually in your ComfyUI/models/text_encoders/ folder, "
        "or connect a DramaBox CLIP Loader node to supply the text encoder."
    )


def _estimate_bytes(module: torch.nn.Module) -> int:
    """Rough VRAM estimate for a module (parameters + buffers)."""
    total = 0
    for t in (*module.parameters(), *module.buffers()):
        if not getattr(t, 'is_meta', False):
            total += t.numel() * t.element_size()
    return max(total, 1)


def _make_patcher(module: torch.nn.Module, load_device: torch.device):
    """Wrap *module* in a ComfyUI ModelPatcher so the memory manager tracks it."""
    import comfy.model_management as _mm
    import comfy.model_patcher
    return comfy.model_patcher.ModelPatcher(
        module,
        load_device=load_device,
        offload_device=_mm.unet_offload_device(),
        size=_estimate_bytes(module),
    )


class _DramaBoxTextBundle(torch.nn.Module):
    """Single nn.Module wrapping Gemma + embeddings processor.

    Treated as one unit by ComfyUI's ModelPatcher so both models move
    GPU↔CPU together as a single eviction unit — matching how LTXVideo
    wraps CLIP text encoders inside comfy.sd.CLIP.
    """

    def __init__(self, text_encoder, embeddings_processor):
        super().__init__()
        self.text_encoder = text_encoder
        self.embeddings_processor = embeddings_processor


def _load_models(device, attention_policy: str = "auto") -> dict:
    """Load the audio VAE, transformer, and audio decoder with CPU-offload.

    All components are moved to GPU per-stage via mm.load_models_gpu().
    ComfyUI's free_memory() evicts each stage's models back to CPU when the
    next stage loads, so only one large model occupies VRAM at a time.

    The Gemma text encoder is NOT loaded here — it is handled separately by
    _load_text_encoder() or supplied via a DRAMABOX_CLIP input.
    """
    import comfy.model_management as mm

    policy_map = {
        "auto": "best_available",
        "flash-attn": "force_fa2",
        "spda": "force_sdpa",
        "default": "force_default",
    }
    policy_key = str(attention_policy).strip().lower()
    if policy_key not in policy_map:
        logger.warning(
            "[DramaBox] Unknown attention policy '%s' - using auto",
            attention_policy,
        )
        attention_policy = "best_available"
    else:
        attention_policy = policy_map[policy_key]

    cache_key = f"{device}|{attention_policy}"
    if cache_key in _LOADED_MODELS:
        logger.debug("[DramaBox] Using cached models.")
        cached = _LOADED_MODELS[cache_key]
        if "attention_backend" not in cached:
            cached["attention_backend"] = "unknown"
        if "attention_backend_source" not in cached:
            cached["attention_backend_source"] = "unknown"
        if "attention_policy" not in cached:
            cached["attention_policy"] = attention_policy
        return cached

    offload_device = mm.unet_offload_device()   # typically torch.device('cpu')
    torch_dtype = torch.bfloat16

    # -- Resolve weight paths (local-first via patched model_downloader) -------
    from model_downloader import get_model_path
    models_dir = _get_models_dir()
    logger.debug("[DramaBox] Resolving model weights...")
    ckpt_transformer = get_model_path("transformer",      cache_dir=models_dir)
    ckpt_audio       = get_model_path("audio_components", cache_dir=models_dir)

    from dramabox_ltx_compat import AudioConditioner, AudioDecoder

    # ── 1. AudioConditioner (VAE encoder) — load to CPU ──────────────────────
    logger.debug("[DramaBox] Loading AudioConditioner (CPU)...")
    try:
        audio_conditioner = AudioConditioner(
            checkpoint_path=ckpt_audio,
            dtype=torch_dtype,
            device=offload_device,
            warm=True,
        )
    except Exception as exc:
        logger.warning("[DramaBox] CPU init failed for AudioConditioner (%s) — falling back to GPU", exc)
        audio_conditioner = AudioConditioner(
            checkpoint_path=ckpt_audio,
            dtype=torch_dtype,
            device=device,
            warm=True,
        )
    mm.soft_empty_cache()

    # ── 2. Transformer (LTX audio-only DiT) — build on CPU ───────────────────
    logger.debug("[DramaBox] Loading Transformer (CPU)...")
    from safetensors import safe_open
    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.loader.sd_ops import SDOps
    from ltx_core.model.transformer.model import LTXModel, LTXModelType
    from ltx_core.model.transformer.rope import LTXRopeType
    from ltx_core.model.transformer.text_projection import create_caption_projection
    try:
        from ltx_core.model.transformer.attention import AttentionFunction, get_best_attention_function
    except Exception:
        from ltx_core.model.transformer.attention import AttentionFunction
        get_best_attention_function = None
    from ltx_core.model.model_protocol import ModelConfigurator

    with safe_open(ckpt_transformer, framework="pt") as f:
        config = json.loads(f.metadata()["config"])

    _t_cfg = config.get("transformer", {})
    def _attention_from_name(name: str):
        try:
            return AttentionFunction(name)
        except Exception:
            try:
                return AttentionFunction(str(name).lower())
            except Exception:
                return None

    _ckpt_attn_name = str(_t_cfg.get("attention_type", "default"))
    selected_attention = None
    _attn_source = "checkpoint-config"

    if attention_policy == "best_available":
        if callable(get_best_attention_function):
            try:
                selected_attention = get_best_attention_function()
                _attn_source = "best-available"
            except Exception as exc:
                logger.warning(
                    "[DramaBox] get_best_attention_function() failed (%s) - falling back to checkpoint attention_type='%s'",
                    exc,
                    _ckpt_attn_name,
                )

    elif attention_policy == "checkpoint_config":
        selected_attention = _attention_from_name(_ckpt_attn_name)
        _attn_source = "checkpoint-config"

    elif attention_policy in {"force_fa2", "force_sdpa", "force_default"}:
        force_candidates = {
            "force_fa2": ["flash_attention_2", "flash_attention2", "fa2", "flash_attn_2", "flash_attn2"],
            "force_sdpa": ["sdpa"],
            "force_default": ["default", "eager"],
        }
        for candidate in force_candidates[attention_policy]:
            selected_attention = _attention_from_name(candidate)
            if selected_attention is not None:
                _attn_source = attention_policy
                break
        if selected_attention is None:
            logger.warning(
                "[DramaBox] Forced attention policy '%s' unavailable - falling back to checkpoint attention_type='%s'",
                attention_policy,
                _ckpt_attn_name,
            )

    if selected_attention is None:
        selected_attention = _attention_from_name(_ckpt_attn_name)
        _attn_source = "checkpoint-config"

    if selected_attention is None:
        selected_attention = AttentionFunction("default")
        _attn_source = "default-fallback"
        logger.warning("[DramaBox] Could not resolve checkpoint attention_type='%s' - using default", _ckpt_attn_name)

    _attn_name = getattr(selected_attention, "value", str(selected_attention))
    logger.debug(
        "[DramaBox] Attention backend: %s (%s) [policy=%s]",
        _attn_source,
        _attn_name,
        attention_policy,
    )

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
                attention_type=selected_attention,
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
    try:
        transformer = (
            Builder(
                model_path=ckpt_transformer,
                model_class_configurator=_AudioOnlyConfigurator,
                model_sd_ops=sd_ops,
                registry=DummyRegistry(),
            )
            .build(device=offload_device, dtype=torch_dtype)
            .to(offload_device)
            .eval()
        )
    except Exception as exc:
        logger.warning("[DramaBox] CPU build failed for Transformer (%s) — falling back to GPU", exc)
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
    logger.debug("[DramaBox] Transformer: %.1fB params", n_params)
    mm.soft_empty_cache()

    # ── 3. AudioDecoder (VAE decoder + vocoder) — load to CPU ────────────────
    logger.debug("[DramaBox] Loading AudioDecoder (CPU)...")
    try:
        audio_decoder = AudioDecoder(
            checkpoint_path=ckpt_audio,
            dtype=torch_dtype,
            device=offload_device,
            warm=True,
        )
    except Exception as exc:
        logger.warning("[DramaBox] CPU init failed for AudioDecoder (%s) — falling back to GPU", exc)
        audio_decoder = AudioDecoder(
            checkpoint_path=ckpt_audio,
            dtype=torch_dtype,
            device=device,
            warm=True,
        )
    mm.soft_empty_cache()

    # ── Build stage-specific patcher groups ───────────────────────────────────
    # voice_patchers : audio VAE encoder (step 3, optional)
    # xfmr_patchers  : diffusion transformer (step 7)
    # dec_patchers   : audio VAE decoder + vocoder (step 9)
    voice_patchers = [_make_patcher(audio_conditioner._warm_encoder, device)]
    xfmr_patchers  = [_make_patcher(transformer, device)]
    dec_patchers   = [
        _make_patcher(audio_decoder._warm_decoder, device),
        _make_patcher(audio_decoder._warm_vocoder, device),
    ]

    model = {
        "voice_patchers":    voice_patchers,
        "xfmr_patchers":     xfmr_patchers,
        "dec_patchers":      dec_patchers,
        "audio_conditioner": audio_conditioner,
        "transformer":       transformer,
        "audio_decoder":     audio_decoder,
        "device":            device,
        "dtype":             torch_dtype,
        "attention_backend": _attn_name,
        "attention_backend_source": _attn_source,
        "attention_policy": attention_policy,
    }
    _LOADED_MODELS[cache_key] = model
    logger.debug("[DramaBox] Non-text components loaded (audio VAE, transformer, decoder).")
    return model


def _load_text_encoder(device):
    """Auto-load the DramaBox Gemma text encoder as a standard ComfyUI CLIP.

    Uses DramaBoxTextEncoderLoader so the fallback path is identical to the
    connected-CLIP path — both go through DramaBoxTEModel + ModelPatcher.
    Result is cached by (device, path); a settings change triggers a reload.
    """
    from .dramabox_clip import DramaBoxTextEncoderLoader

    gemma_path = _find_or_download_gemma_path()
    cache_key = f"{device}:{gemma_path}"

    if cache_key in _LOADED_TEXT_ENCODER:
        logger.info("[DramaBox] Text encoder: %s (cached)", os.path.basename(gemma_path))
        return _LOADED_TEXT_ENCODER[cache_key]

    # Evict any stale entry for this device (different path)
    stale = [k for k in list(_LOADED_TEXT_ENCODER) if k.startswith(f"{device}:")]
    for k in stale:
        del _LOADED_TEXT_ENCODER[k]

    logger.info("[DramaBox] Text encoder: %s", os.path.basename(gemma_path))
    clip_device = "default"  # follow Comfy's default text-encoder device policy
    (clip,) = DramaBoxTextEncoderLoader().load(gemma_path, clip_device)
    _LOADED_TEXT_ENCODER[cache_key] = clip
    return clip


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_lora_path(lora_name: str) -> str | None:
    """Resolve a LoRA name to an absolute path via ComfyUI's folder system."""
    import folder_paths
    path = folder_paths.get_full_path("loras", lora_name)
    if path and os.path.isfile(path):
        return path
    if os.path.isabs(lora_name) and os.path.isfile(lora_name):
        return lora_name
    return None


def _detect_lora_rank(lora_path: str) -> int | None:
    """Best-effort LoRA rank detection from a safetensors adapter file."""
    try:
        from safetensors import safe_open
    except Exception:
        return None

    try:
        with safe_open(str(lora_path), framework="pt") as f:
            keys = list(f.keys())

            for key in keys:
                if key.endswith(".lora_A.weight"):
                    tensor = f.get_tensor(key)
                    if tensor.ndim >= 2 and tensor.shape[0] > 0:
                        return int(tensor.shape[0])

            for key in keys:
                if key.endswith(".lora_B.weight"):
                    tensor = f.get_tensor(key)
                    if tensor.ndim >= 2 and tensor.shape[1] > 0:
                        return int(tensor.shape[1])
    except Exception:
        return None

    return None


def _normalize_lora_stack(lora_stack) -> list[tuple[str, float, float]]:
    """Normalize common ComfyUI LoRA stack variants.

    Returns a list of (lora_name_or_path, strength_model, strength_clip).
    """
    if not lora_stack:
        return []

    source = lora_stack
    if isinstance(source, dict):
        for key in ("lora_stack", "stack", "loras", "items", "value"):
            if key in source:
                source = source.get(key)
                break
        else:
            return []

    if isinstance(source, (str, Path)):
        return [(str(source), 1.0, 1.0)]

    if isinstance(source, tuple) and source and isinstance(source[0], (str, Path)):
        items = [source]
    elif isinstance(source, list):
        items = source
    elif isinstance(source, tuple):
        items = list(source)
    else:
        items = [source]

    normalized: list[tuple[str, float, float]] = []
    for item in items:
        name = None
        strength_model = 1.0
        strength_clip = 1.0

        if isinstance(item, (str, Path)):
            name = str(item)
        elif isinstance(item, dict):
            name = item.get("lora_name") or item.get("name") or item.get("path") or item.get("lora")
            strength_model = item.get("strength_model", item.get("strength", 1.0))
            strength_clip = item.get("strength_clip", 1.0)
        elif isinstance(item, (list, tuple)):
            if len(item) >= 3:
                name, strength_model, strength_clip = item[0], item[1], item[2]
            elif len(item) == 2:
                name, strength_model = item[0], item[1]
            elif len(item) == 1:
                name = item[0]

        if name is None:
            continue

        try:
            strength_model = float(strength_model)
        except Exception:
            strength_model = 1.0
        try:
            strength_clip = float(strength_clip)
        except Exception:
            strength_clip = 1.0

        normalized.append((str(name), strength_model, strength_clip))

    return normalized


def _resolve_wrapper_loras(lora_stack) -> list[tuple[str, float, int]]:
    """Resolve wrapper-mode LoRA stack for one-shot OG invocation.

    Returns:
        [(lora_path, strength, lora_rank), ...]
    """
    normalized_stack = _normalize_lora_stack(lora_stack)
    if not normalized_stack:
        return []

    resolved: list[tuple[str, float, int]] = []
    for lora_name, strength_model, _strength_clip in normalized_stack:
        path = _resolve_lora_path(lora_name)
        if path is None:
            logger.warning("[DramaBox] LoRA not found for wrapper mode, skipping: %s", lora_name)
            continue

        try:
            strength_val = float(strength_model)
        except Exception:
            strength_val = 1.0

        if strength_val <= 0:
            continue

        lora_rank = _detect_lora_rank(path) or 128
        resolved.append((path, strength_val, int(lora_rank)))

    return resolved


def _apply_lora_deltas(transformer: torch.nn.Module, lora_path: str, strength: float) -> list:
    """Apply LoRA weights to transformer in-place using manual delta math.

    Handles both PEFT format (base_model.model.*) and original ID-LoRA format
    (diffusion_model.*), matching the approach in py/db_inference.py.

    Returns a list of (param_name, delta_tensor) so the caller can undo the
    changes after inference by subtracting each delta.
    """
    from safetensors.torch import load_file as _st_load

    lora_sd = _st_load(lora_path)
    is_peft = any("base_model.model." in k for k in lora_sd)
    is_idlora = any("diffusion_model." in k for k in lora_sd)

    # Build {param_path: {"A": tensor, "B": tensor, "alpha": float}} mapping
    pairs: dict[str, dict] = {}
    for k, v in lora_sd.items():
        if is_peft:
            if "base_model.model." not in k:
                continue
            base = k.replace("base_model.model.", "")
            if ".lora_A." in base:
                pp = base[: base.index(".lora_A.")]
                pairs.setdefault(pp, {})["A"] = v
            elif ".lora_B." in base:
                pp = base[: base.index(".lora_B.")]
                pairs.setdefault(pp, {})["B"] = v
        elif is_idlora:
            if "diffusion_model." not in k:
                continue
            base = k.replace("diffusion_model.", "")
            if ".lora_A.weight" in base:
                pp = base.replace(".lora_A.weight", "")
                pairs.setdefault(pp, {})["A"] = v
            elif ".lora_B.weight" in base:
                pp = base.replace(".lora_B.weight", "")
                pairs.setdefault(pp, {})["B"] = v

    param_dict = dict(transformer.named_parameters())
    key_prefixes = (
        "_orig_mod.",
        "model.",
        "module.",
        "_orig_mod.model.",
        "model._orig_mod.",
    )
    applied: list[tuple[str, torch.Tensor]] = []

    for pp, pair in pairs.items():
        if "A" not in pair or "B" not in pair:
            continue
        lora_A, lora_B = pair["A"], pair["B"]
        rank = lora_A.shape[0]
        scale = strength  # alpha == rank by default (scale = alpha/rank * strength = strength)

        weight_key = pp + ".weight"
        resolved_key = weight_key
        if resolved_key not in param_dict:
            for prefix in key_prefixes:
                candidate = prefix + weight_key
                if candidate in param_dict:
                    resolved_key = candidate
                    break

        if resolved_key not in param_dict:
            continue

        param = param_dict[resolved_key]
        dev, dt = param.device, param.dtype

        # delta = scale * lora_B @ lora_A   (computed in float32 for precision)
        delta = scale * (
            lora_B.to(device=dev, dtype=torch.float32)
            @ lora_A.to(device=dev, dtype=torch.float32)
        ).to(dtype=dt)

        param.data.add_(delta)
        applied.append((resolved_key, delta))

    return applied


def _remove_lora_deltas(transformer: torch.nn.Module, applied: list) -> None:
    """Undo deltas previously applied by _apply_lora_deltas."""
    param_dict = dict(transformer.named_parameters())
    for weight_key, delta in applied:
        if weight_key in param_dict:
            p = param_dict[weight_key]
            p.data.sub_(delta.to(device=p.device, dtype=p.dtype))


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


def _is_cuda_cpu_device_mismatch(exc: RuntimeError) -> bool:
    """Return True for common torch CUDA/CPU mixed-tensor runtime errors."""
    msg = str(exc).lower()
    return (
        "expected all tensors to be on the same device" in msg
        and "cuda" in msg
        and "cpu" in msg
    )


def _is_cuda_oom(exc: RuntimeError) -> bool:
    """Return True for common CUDA out-of-memory runtime errors."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    if "cuda error: out of memory" in msg:
        return True
    return "out of memory" in msg and any(tok in msg for tok in ("cuda", "cudnn", "cublas"))


def _build_device_mismatch_message(generation_mode: str, offload_policy: str) -> str:
    return (
        "[DramaBox] Denoising failed due to mixed CUDA/CPU tensors.\n"
        "Likely cause: VRAM pressure caused a partial transformer load.\n"
        f"Current mode: {generation_mode}, post_generate_model_policy: {offload_policy}.\n"
        "Recommended actions:\n"
        "1. Set post_generate_model_policy to offload_to_cpu or offload.\n"
        "2. Avoid keep_loaded on lower VRAM systems.\n"
        "3. Run DramaBox Unload once, then retry.\n"
        "4. Close other heavy models/workflows.\n"
        "5. If needed, use dramabox_wrapper with offload_to_cpu."
    )


def _build_cuda_oom_message(generation_mode: str, offload_policy: str) -> str:
    return (
        "[DramaBox] Denoising failed due to CUDA out-of-memory (OOM).\n"
        f"Current mode: {generation_mode}, post_generate_model_policy: {offload_policy}.\n"
        "Recommended actions:\n"
        "1. Set post_generate_model_policy to offload_to_cpu or offload.\n"
        "2. Avoid keep_loaded on lower VRAM systems.\n"
        "3. Run DramaBox Unload once, then retry.\n"
        "4. Close other heavy models/workflows.\n"
        "5. If needed, use dramabox_wrapper with offload_to_cpu."
    )


def _collect_dramabox_patchers() -> list:
    """Return all internal DramaBox ModelPatchers currently cached."""
    all_patchers = []
    for clip in _LOADED_TEXT_ENCODER.values():
        try:
            all_patchers.append(clip.patcher)
        except AttributeError:
            pass
    for model in _LOADED_MODELS.values():
        for group in ("voice_patchers", "xfmr_patchers", "dec_patchers"):
            all_patchers.extend(model.get(group, []))
    return all_patchers


def _offload_dramabox_models_to_cpu():
    """Move cached DramaBox models to CPU but keep them in RAM for fast reuse."""
    import comfy.model_management as mm

    if not _LOADED_MODELS and not _LOADED_TEXT_ENCODER:
        logger.info("[DramaBox] No models loaded — nothing to offload.")
        return

    all_patchers = _collect_dramabox_patchers()

    for patcher in all_patchers:
        try:
            patcher.unpatch_model(patcher.offload_device)
        except Exception as e:
            logger.debug("[DramaBox] unpatch_model failed during offload: %s", e)

    patcher_set = set(id(p) for p in all_patchers)
    mm.current_loaded_models[:] = [
        m for m in mm.current_loaded_models if id(m.model) not in patcher_set
    ]

    mm.soft_empty_cache()
    logger.info("[DramaBox] Cached models offloaded to CPU (kept in RAM).")


def _unload_dramabox_models() -> int:
    """Free all DramaBox model weights from GPU and CPU RAM.

    Unpatches all ComfyUI-managed ModelPatchers (moving them to CPU), removes
    them from current_loaded_models, then clears _LOADED_MODELS so GC can
    reclaim memory.  After this call the next DramaBox TTS generation will
    reload everything from disk.
    """
    import gc
    import comfy.model_management as mm

    released = len(_LOADED_MODELS) + len(_LOADED_TEXT_ENCODER)

    if released <= 0:
        logger.info("[DramaBox] No models loaded — nothing to free.")
        return 0

    # Collect all internally-managed patchers (external CLIP is user-managed)
    all_patchers = _collect_dramabox_patchers()

    for patcher in all_patchers:
        try:
            patcher.unpatch_model(patcher.offload_device)
        except Exception as e:
            logger.debug("[DramaBox] unpatch_model failed: %s", e)

    # ── 2. Remove patchers from ComfyUI's tracking list ──────────────────
    patcher_set = set(id(p) for p in all_patchers)
    mm.current_loaded_models[:] = [
        m for m in mm.current_loaded_models if id(m.model) not in patcher_set
    ]

    _LOADED_MODELS.clear()
    _LOADED_TEXT_ENCODER.clear()
    gc.collect()
    mm.soft_empty_cache()
    logger.info("[DramaBox] All clip_loader models unloaded. VRAM freed.")
    return released


def _normalize_generation_mode(mode) -> str:
    mode = str(mode).strip().lower()
    if mode in {"clip_loader", "dramabox_wrapper"}:
        return mode
    return "clip_loader"


def _default_generation_mode() -> str:
    """Default generation mode from global DramaBox preference."""
    return "dramabox_wrapper" if _get_dramabox_bool_setting("DramaBox.defaultWrapperMode", False) else "clip_loader"


def _apply_dramabox_wrapper_compat(force: bool = False) -> None:
    """Apply runtime compatibility shims used by DramaBox_Wrapper mode."""
    global _DRAMABOX_WRAPPER_COMPAT_APPLIED
    if _DRAMABOX_WRAPPER_COMPAT_APPLIED and not force:
        return

    try:
        from transformers.models.siglip.modeling_siglip import SiglipVisionModel

        if not hasattr(SiglipVisionModel, "vision_model"):
            # ltx_core expects vision_tower.vision_model in some versions.
            SiglipVisionModel.vision_model = property(lambda self: self)
            logger.info("[DramaBox] DramaBox_Wrapper compat: patched SiglipVisionModel.vision_model")
    except Exception as exc:
        logger.debug("[DramaBox] DramaBox_Wrapper compat patch skipped: %s", exc)

    try:
        from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

        def _rope_local_base_freq(self):
            rope_params = getattr(self, "rope_parameters", {}) or {}
            sliding = rope_params.get("sliding_attention", {}) or {}
            default_theta = getattr(self, "default_theta", {"local": 10000.0})
            return float(sliding.get("rope_theta", default_theta.get("local", 10000.0)))

        def _rope_scaling(self):
            rope_params = getattr(self, "rope_parameters", {}) or {}
            full = rope_params.get("full_attention", {}) or {}
            try:
                stored = object.__getattribute__(self, "__dict__").get("rope_scaling")
            except Exception:
                stored = None
            out = dict(stored) if isinstance(stored, dict) else {}
            out.setdefault("rope_type", full.get("rope_type", "default"))
            if "rope_theta" in full and "rope_theta" not in out:
                out["rope_theta"] = full["rope_theta"]
            rope_type = str(out.get("rope_type", "default"))
            if rope_type in {"linear", "dynamic", "yarn", "llama3", "longrope"} and "factor" not in out:
                out["factor"] = float(full.get("factor", 1.0))
            return out

        Gemma3TextConfig.rope_local_base_freq = property(_rope_local_base_freq)
        Gemma3TextConfig.rope_scaling = property(_rope_scaling)
        logger.info("[DramaBox] DramaBox_Wrapper compat: patched Gemma3TextConfig rope fields")
    except Exception as exc:
        logger.debug("[DramaBox] DramaBox_Wrapper Gemma config patch skipped: %s", exc)

    try:
        from transformers import modeling_rope_utils as _rope_utils

        if not getattr(_rope_utils, "_dramabox_wrapper_compat", False):
            _orig_linear_scaling = _rope_utils._compute_linear_scaling_rope_parameters

            def _linear_scaling_with_factor_fallback(config, device=None, seq_len=None, **rope_kwargs):
                try:
                    return _orig_linear_scaling(config, device=device, seq_len=seq_len, **rope_kwargs)
                except KeyError as rope_exc:
                    if str(rope_exc).strip("'\"") != "factor":
                        raise

                    scaling = getattr(config, "rope_scaling", None)
                    if not isinstance(scaling, dict):
                        raise

                    scaling = dict(scaling)
                    scaling.setdefault("factor", 1.0)
                    try:
                        setattr(config, "rope_scaling", scaling)
                    except Exception:
                        raise

                    logger.warning(
                        "[DramaBox] DramaBox_Wrapper compat: transformers linear rope missing factor; defaulted to 1.0"
                    )
                    return _orig_linear_scaling(config, device=device, seq_len=seq_len, **rope_kwargs)

            _rope_utils._compute_linear_scaling_rope_parameters = _linear_scaling_with_factor_fallback
            _rope_utils._dramabox_wrapper_compat = True
            logger.info("[DramaBox] DramaBox_Wrapper compat: patched transformers linear rope factor fallback")
    except Exception as exc:
        logger.debug("[DramaBox] DramaBox_Wrapper rope utils patch skipped: %s", exc)

    try:
        from ltx_core.text_encoders.gemma.encoders import encoder_configurator as _enc_cfg

        if not getattr(_enc_cfg, "_dramabox_wrapper_compat", False):
            def _create_and_populate_compat(module):
                model = module.model
                vision_tower = model.model.vision_tower
                v_model = getattr(vision_tower, "vision_model", vision_tower)
                l_model = model.model.language_model

                config = model.config.text_config
                dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

                rope_params = getattr(config, "rope_parameters", {}) or {}
                sliding = rope_params.get("sliding_attention", {}) or {}
                full = rope_params.get("full_attention", {}) or {}
                default_theta = getattr(config, "default_theta", {"local": 10000.0})
                base = getattr(config, "rope_local_base_freq", None)
                if base is None:
                    base = sliding.get("rope_theta", default_theta.get("local", 10000.0))

                local_rope_freqs = 1.0 / (
                    float(base) ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim)
                )

                rope_scaling = getattr(config, "rope_scaling", None)
                rope_type = rope_scaling.get("rope_type") if isinstance(rope_scaling, dict) else None
                if rope_type is None:
                    rope_type = full.get("rope_type", "default")

                try:
                    merged_rope_scaling = dict(rope_scaling) if isinstance(rope_scaling, dict) else {}
                    merged_rope_scaling.setdefault("rope_type", rope_type)
                    if "rope_theta" in full and "rope_theta" not in merged_rope_scaling:
                        merged_rope_scaling["rope_theta"] = full["rope_theta"]
                    if rope_type in {"linear", "dynamic", "yarn", "llama3", "longrope"} and "factor" not in merged_rope_scaling:
                        merged_rope_scaling["factor"] = float(full.get("factor", 1.0))
                    setattr(config, "rope_scaling", merged_rope_scaling)
                except Exception:
                    pass

                if rope_type not in _enc_cfg.ROPE_INIT_FUNCTIONS:
                    rope_type = "default"
                    try:
                        merged_rope_scaling = dict(getattr(config, "rope_scaling", {}) or {})
                        merged_rope_scaling["rope_type"] = rope_type
                        setattr(config, "rope_scaling", merged_rope_scaling)
                    except Exception:
                        pass

                try:
                    inv_freqs, _ = _enc_cfg.ROPE_INIT_FUNCTIONS[rope_type](config)
                except KeyError as rope_exc:
                    if str(rope_exc).strip("'\"") != "factor":
                        raise
                    merged_rope_scaling = dict(getattr(config, "rope_scaling", {}) or {})
                    merged_rope_scaling.setdefault("factor", 1.0)
                    try:
                        setattr(config, "rope_scaling", merged_rope_scaling)
                    except Exception:
                        pass
                    inv_freqs, _ = _enc_cfg.ROPE_INIT_FUNCTIONS[rope_type](config)
                    logger.warning(
                        "[DramaBox] DramaBox_Wrapper compat: missing rope_scaling.factor for %s; defaulted to 1.0",
                        rope_type,
                    )

                positions_length = len(v_model.embeddings.position_ids[0])
                position_ids = torch.arange(positions_length, dtype=torch.long, device="cpu").unsqueeze(0)
                v_model.embeddings.register_buffer("position_ids", position_ids)
                embed_scale = torch.tensor(model.config.text_config.hidden_size**0.5, device="cpu")
                l_model.embed_tokens.register_buffer("embed_scale", embed_scale)
                l_model.rotary_emb_local.register_buffer("inv_freq", local_rope_freqs)
                l_model.rotary_emb.register_buffer("inv_freq", inv_freqs)

                return module

            _enc_cfg.create_and_populate = _create_and_populate_compat
            if hasattr(_enc_cfg, "GEMMA_MODEL_OPS") and hasattr(_enc_cfg.GEMMA_MODEL_OPS, "mutator"):
                _enc_cfg.GEMMA_MODEL_OPS.mutator = _create_and_populate_compat
            _enc_cfg._dramabox_wrapper_compat = True
            logger.info("[DramaBox] DramaBox_Wrapper compat: patched encoder_configurator.create_and_populate")
    except Exception as exc:
        logger.debug("[DramaBox] DramaBox_Wrapper encoder configurator patch skipped: %s", exc)

    _DRAMABOX_WRAPPER_COMPAT_APPLIED = True


def _get_og_server(device):
    """Return a cached DramaBox_Wrapper TTSServer bound to the current torch device."""
    cache_key = str(device)
    if cache_key in _OG_SERVER_CACHE:
        return _OG_SERVER_CACHE[cache_key]

    _apply_dramabox_wrapper_compat()

    from db_inference_server import TTSServer
    from model_downloader import get_gemma_path, get_model_path

    models_dir = _get_models_dir()
    ckpt_transformer = get_model_path("transformer", cache_dir=models_dir)
    ckpt_audio = get_model_path("audio_components", cache_dir=models_dir)
    gemma_root = get_gemma_path(cache_dir=models_dir)

    use_bnb_4bit = str(device).startswith("cuda")
    server_kwargs = dict(
        checkpoint=ckpt_transformer,
        full_checkpoint=ckpt_audio,
        gemma_root=gemma_root,
        device=cache_key,
        dtype="bf16",
        compile_model=False,
    )

    def _new_server(bnb_flag: bool):
        return TTSServer(bnb_4bit=bnb_flag, **server_kwargs)

    try:
        server = _new_server(use_bnb_4bit)
    except Exception as exc:
        msg = str(exc)
        needs_compat_retry = (
            "vision_model" in msg
            or "rope_local_base_freq" in msg
            or "rope_type" in msg
            or "factor" in msg
        )
        if not needs_compat_retry:
            raise

        logger.warning(
            "[DramaBox] DramaBox_Wrapper init mismatch (%s). Re-applying compat shims and retrying.",
            exc,
        )
        _apply_dramabox_wrapper_compat(force=True)

        try:
            server = _new_server(use_bnb_4bit)
        except Exception as exc2:
            if not use_bnb_4bit:
                raise
            logger.warning(
                "[DramaBox] DramaBox_Wrapper retry failed (%s). Retrying with bnb_4bit disabled.",
                exc2,
            )
            server = _new_server(False)
    _OG_SERVER_CACHE[cache_key] = server
    logger.info("[DramaBox] DramaBox_Wrapper server ready on %s", cache_key)
    return server


def _release_og_server(device):
    """Release cached DramaBox_Wrapper server for a device."""
    import gc

    cache_key = str(device)
    if cache_key in _OG_SERVER_CACHE:
        del _OG_SERVER_CACHE[cache_key]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def unload_og_servers() -> int:
    """Release all cached DramaBox_Wrapper servers and free VRAM cache.

    Returns:
        Number of cached OG server entries that were released.
    """
    import gc

    released = len(_OG_SERVER_CACHE)
    if released <= 0:
        return 0

    for key in list(_OG_SERVER_CACHE.keys()):
        del _OG_SERVER_CACHE[key]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("[DramaBox] Released %d cached OG server(s)", released)
    return released


def unload_all_dramabox_caches() -> tuple[int, int]:
    """Unload both clip_loader caches and OG wrapper caches.

    Returns:
        (clip_loader_released, og_released)
    """
    clip_released = _unload_dramabox_models()
    og_released = unload_og_servers()
    return clip_released, og_released


def _summarize_voice_ref(voice_ref) -> str:
    """Build a concise debug summary for ComfyUI AUDIO input."""
    if voice_ref is None:
        return "none"
    if not isinstance(voice_ref, dict):
        return f"invalid type={type(voice_ref)}"

    waveform = voice_ref.get("waveform")
    sample_rate = voice_ref.get("sample_rate")
    if waveform is None or sample_rate is None:
        return "missing waveform/sample_rate"

    if not torch.is_tensor(waveform):
        return f"waveform is not tensor (type={type(waveform)}), sr={sample_rate}"

    shape = tuple(waveform.shape)
    dtype = str(waveform.dtype)
    device = str(waveform.device)
    samples = int(shape[-1]) if len(shape) > 0 else 0
    duration = (samples / float(sample_rate)) if sample_rate else 0.0
    peak = float(waveform.detach().abs().max().item()) if waveform.numel() > 0 else 0.0
    return (
        f"shape={shape}, sr={int(sample_rate)}, duration={duration:.2f}s, "
        f"dtype={dtype}, device={device}, peak={peak:.4f}"
    )


def _voice_ref_to_temp_wav(voice_ref):
    """Write ComfyUI AUDIO input to a temporary wav file and return its path."""
    if voice_ref is None:
        return None

    waveform = voice_ref.get("waveform") if isinstance(voice_ref, dict) else None
    sample_rate = voice_ref.get("sample_rate") if isinstance(voice_ref, dict) else None
    if waveform is None or sample_rate is None:
        logger.info("[DramaBox] voice_ref temp wav skipped: missing waveform/sample_rate")
        return None

    import tempfile
    import soundfile as sf

    wav = waveform[0].detach().cpu().float().numpy()  # [C, S]
    if wav.ndim == 2:
        wav = wav.T

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, int(sample_rate))
        logger.info(
            "[DramaBox] voice_ref temp wav created: %s (sr=%d, samples=%d)",
            tmp.name,
            int(sample_rate),
            int(wav.shape[0]) if wav.ndim >= 1 else 0,
        )
        return tmp.name


def _run_og_low_memory_once(
    prompt: str,
    *,
    seed: int,
    voice_ref_path: str | None,
    speed: float,
    cfg_scale: float,
    stg_scale: float,
    duration_multiplier: float,
    ref_duration: float,
    gen_duration: float,
    steps: int,
    negative_prompt: str,
    rescale_raw,
    lora_entries: list[tuple[str, float, int]] | None = None,
) -> tuple[torch.Tensor, int]:
    """Run OG inference in one-shot mode (VCS-style low-memory path)."""
    import subprocess
    import tempfile
    import soundfile as sf

    from model_downloader import get_gemma_path, get_model_path

    models_dir = _get_models_dir()
    ckpt_transformer = get_model_path("transformer", cache_dir=models_dir)
    ckpt_audio = get_model_path("audio_components", cache_dir=models_dir)
    gemma_root = get_gemma_path(cache_dir=models_dir)

    infer_script = os.path.join(_SRC_DIR, "db_inference.py")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
        output_path = tmp_out.name

    cmd = [
        sys.executable,
        infer_script,
        "--prompt",
        str(prompt),
        "--output",
        output_path,
        "--seed",
        str(int(seed)),
        "--checkpoint",
        str(ckpt_transformer),
        "--full-checkpoint",
        str(ckpt_audio),
        "--gemma-root",
        str(gemma_root),
        "--cfg-scale",
        str(float(cfg_scale)),
        "--stg-scale",
        str(float(stg_scale)),
        "--duration-multiplier",
        str(float(duration_multiplier)),
        "--speed",
        str(float(speed)),
        "--ref-duration",
        str(float(ref_duration)),
        "--steps",
        str(int(steps)),
        "--negative-prompt",
        str(negative_prompt),
        "--no-watermark",
    ]

    if gen_duration and float(gen_duration) > 0:
        cmd.extend(["--gen-duration", str(float(gen_duration))])

    if rescale_raw != "auto":
        cmd.extend(["--rescale-scale", str(float(rescale_raw))])

    if voice_ref_path:
        cmd.extend(["--voice-sample", str(voice_ref_path)])
        logger.info("[DramaBox] one-shot wrapper: passing --voice-sample")
    else:
        cmd.append("--no-ref")
        logger.info("[DramaBox] one-shot wrapper: no voice sample provided")

    if lora_entries:
        for lora_path, lora_strength, lora_rank in lora_entries:
            cmd.extend(
                [
                    "--lora",
                    str(lora_path),
                    "--lora-strength",
                    str(float(lora_strength)),
                    "--lora-rank",
                    str(int(lora_rank)),
                ]
            )

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "").strip().splitlines()[-20:]
            raise RuntimeError("Low-memory OG inference failed:\n" + "\n".join(stderr_tail))

        wav_np, sample_rate = sf.read(output_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(wav_np.T).unsqueeze(0)
        return waveform, int(sample_rate)
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


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
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**31 - 1, "control_after_generate": True},
                ),
                "use_prompt_input": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "on",
                        "label_off": "off",
                        "tooltip": "When on, use the connected prompt input instead of the text widget.",
                    },
                ),
                "text": (
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
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "forceInput": True,
                        "lazy": True,
                        "tooltip": "Connect any text source here to override the text widget.",
                    },
                ),
                "lora_stack": (
                    "LORA_STACK",
                    {"tooltip": "Optional LoRA stack (from any LoRA stacker node). Applied to the transformer only."},
                ),
                "options": (
                    "DRAMABOX_OPTIONS",
                    {"tooltip": "Connect DramaBox Options for cfg, steps, duration, and more."},
                ),
                "dramabox_clip": (
                    "CLIP",
                    {
                        "lazy": True,
                        "tooltip": "Pre-loaded text encoder from DramaBox CLIP Loader. Used only in clip_loader mode.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, seed, use_prompt_input=False, text="", prompt=None, options=None, **kwargs):
        # Cache key should follow the effective prompt source and options.
        effective_prompt = prompt if (use_prompt_input and prompt is not None) else text
        try:
            options_sig = json.dumps(options or {}, sort_keys=True, ensure_ascii=True)
        except Exception:
            options_sig = str(options)
        return (seed, bool(use_prompt_input), effective_prompt, options_sig)

    def check_lazy_status(self, seed, use_prompt_input=False, text="", options=None, **kwargs):
        """Request lazy inputs only when they are needed for the current mode."""
        needed = ["prompt"] if use_prompt_input else []

        opts = options if isinstance(options, dict) else {}
        generation_mode = _normalize_generation_mode(
            opts.get("generation_mode", _default_generation_mode())
        )
        # For lazy optional inputs, presence in kwargs indicates a connection.
        # The value may still be unresolved/None at this stage.
        has_clip_input = "dramabox_clip" in kwargs
        if generation_mode != "dramabox_wrapper" and has_clip_input:
            needed.append("dramabox_clip")

        return needed

    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def generate(
        self,
        seed: int = 0,
        use_prompt_input: bool = False,
        text: str = "",
        prompt: str | None = None,
        voice_ref=None,
        options: dict | None = None,
        lora_stack=None,
        dramabox_clip=None,
        unique_id=None,
    ):
        import comfy.model_management as mm
        device = mm.get_torch_device()

        # Use connected text source when the toggle is on, else fall back to widget text
        used_prompt = prompt if (use_prompt_input and prompt) else text

        vr_summary = _summarize_voice_ref(voice_ref)
        logger.debug("[DramaBox] voice_ref input summary: %s", vr_summary)

        # Notify the frontend so it can display the active prompt in the widget
        if unique_id is not None and _PromptServer is not None:
            try:
                _PromptServer.instance.send_sync(
                    "dramabox-tts-update",
                    {"node_id": unique_id, "prompt": used_prompt, "use_prompt_input": use_prompt_input},
                )
            except Exception:
                pass

        opts = options or {}
        normalized_lora_stack = _normalize_lora_stack(lora_stack)
        if lora_stack and not normalized_lora_stack:
            logger.warning("[DramaBox] lora_stack input was provided but no valid entries were parsed.")
        elif normalized_lora_stack:
            logger.debug("[DramaBox] LoRA stack parsed: %d adapter(s)", len(normalized_lora_stack))

        generation_mode = _normalize_generation_mode(opts.get("generation_mode", _default_generation_mode()))
        attention_policy = str(opts.get("attention_policy", "auto"))
        ref_duration: float  = opts.get("ref_duration", 10.0)
        gen_duration: float  = opts.get("gen_duration", 0.0)
        duration_mult: float = opts.get("duration_multiplier", 1.1)
        speed: float         = float(opts.get("speed", 1.0))
        steps: int           = int(opts.get("steps", 30))
        cfg_scale: float     = float(opts.get("cfg_scale", 2.5))
        stg_scale: float     = float(opts.get("stg_scale", 1.5))
        negative_prompt: str = opts.get("negative_prompt", _DEFAULT_NEG)
        rescale_raw          = opts.get("rescale_scale", "auto")
        rescale_scale: float = (
            _auto_rescale(cfg_scale) if rescale_raw == "auto" else float(rescale_raw)
        )
        default_policy = "offload_to_cpu" if _get_dramabox_bool_setting("DramaBox.autoOffload", True) else "keep_loaded"
        offload_policy = str(opts.get("post_generate_model_policy", default_policy))
        if offload_policy not in {"keep_loaded", "offload_to_cpu", "offload"}:
            offload_policy = default_policy

        if generation_mode == "dramabox_wrapper":
            logger.info("[DramaBox] generation_mode=dramabox_wrapper")
            server = None
            _applied_deltas: list = []

            tmp_voice_path = _voice_ref_to_temp_wav(voice_ref)
            if tmp_voice_path:
                logger.debug("[DramaBox] wrapper voice_ref temp path ready")
            else:
                logger.debug("[DramaBox] wrapper voice_ref missing/invalid; running without reference")
            try:
                if offload_policy in {"offload_to_cpu", "offload"}:
                    logger.info("[DramaBox] dramabox_wrapper + offload policy -> one-shot low-memory run")

                    lora_entries = _resolve_wrapper_loras(normalized_lora_stack)
                    if normalized_lora_stack and not lora_entries:
                        logger.warning("[DramaBox] no valid wrapper LoRA entries resolved for one-shot run")

                    _release_og_server(device)
                    waveform, sample_rate = _run_og_low_memory_once(
                        used_prompt,
                        seed=int(seed),
                        voice_ref_path=tmp_voice_path,
                        speed=speed,
                        cfg_scale=cfg_scale,
                        stg_scale=stg_scale,
                        duration_multiplier=duration_mult,
                        ref_duration=float(ref_duration),
                        gen_duration=float(gen_duration),
                        steps=steps,
                        negative_prompt=negative_prompt,
                        rescale_raw=rescale_raw,
                        lora_entries=lora_entries or None,
                    )
                else:
                    server = _get_og_server(device)

                    if normalized_lora_stack:
                        for lora_name, strength_model, _strength_clip in normalized_lora_stack:
                            lora_path = _resolve_lora_path(lora_name)
                            if lora_path is None:
                                logger.warning("[DramaBox] LoRA not found, skipping: %s", lora_name)
                                continue
                            deltas = _apply_lora_deltas(server._velocity_model, lora_path, float(strength_model))
                            _applied_deltas.extend(deltas)
                            logger.debug(
                                f"[DramaBox] Wrapper LoRA applied: {os.path.basename(lora_path)} "
                                f"(strength={float(strength_model):.2f}, params={len(deltas)})"
                            )
                            if not deltas:
                                logger.warning(
                                    f"[DramaBox] Warning: no matching transformer params found for "
                                    f"{os.path.basename(lora_path)}"
                                )

                    waveform, sample_rate = server.generate(
                        prompt=used_prompt,
                        voice_ref=tmp_voice_path,
                        speed=speed,
                        cfg_scale=cfg_scale,
                        stg_scale=stg_scale,
                        duration_multiplier=duration_mult,
                        seed=int(seed),
                        ref_duration=float(ref_duration),
                        rescale_scale=rescale_raw,
                        gen_duration=float(gen_duration),
                    )
            finally:
                if _applied_deltas and server is not None:
                    _remove_lora_deltas(server._velocity_model, _applied_deltas)
                    _applied_deltas.clear()
                if tmp_voice_path:
                    try:
                        os.unlink(tmp_voice_path)
                    except OSError:
                        pass

            waveform = waveform.detach().cpu().float()
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)

            if offload_policy in {"offload_to_cpu", "offload"}:
                logger.info("[DramaBox] dramabox_wrapper + offload policy -> releasing wrapper server cache")
                _release_og_server(device)

            return ({"waveform": waveform, "sample_rate": int(sample_rate)},)

        logger.info("[DramaBox] generation_mode=clip_loader")

        # ── Load (or retrieve cached) model components ───────────────────
        # _load_models() builds ModelPatchers on first call; subsequent calls
        # return the cache. load_models_gpu() is called per-stage so only one
        # large model occupies VRAM at a time; ComfyUI evicts the previous
        # stage's weights to CPU automatically.
        model = _load_models(device, attention_policy=attention_policy)
        logger.debug(
            "[DramaBox] Attention backend in use (this run): %s (%s) [policy=%s]",
            model.get("attention_backend_source", "unknown"),
            model.get("attention_backend", "unknown"),
            model.get("attention_policy", attention_policy),
        )

        # ── Resolve text encoder (external CLIP or auto-loaded CLIP) ───────────
        # Both paths produce a standard comfy.sd.CLIP — same encoding code runs
        # regardless of whether the user connected a DramaBoxTextEncoderLoader.
        if dramabox_clip is not None:
            clip_name = getattr(dramabox_clip, "dramabox_model_name", None)
            if not clip_name:
                clip_path = getattr(dramabox_clip, "dramabox_model_path", None)
                if isinstance(clip_path, str) and clip_path:
                    clip_name = os.path.basename(clip_path)
            if not clip_name:
                clip_name = "connected DRAMABOX_CLIP"

            logger.info("[DramaBox] Text encoder: %s", clip_name)
            _clip_enc = dramabox_clip
        else:
            _clip_enc = _load_text_encoder(device)

        audio_cond  = model["audio_conditioner"]
        transformer = model["transformer"]
        decoder     = model["audio_decoder"]
        torch_dtype = model["dtype"]

        # ── imports from ltx packages + local compatibility shims ───────────
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.components.patchifiers import AudioPatchifier
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        from ltx_core.components.schedulers import LTX2Scheduler
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.model.transformer.model import X0Model
        from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
        from ltx_core.tools import AudioLatentTools
        from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
        from dramabox_ltx_compat import GuidedDenoiser, euler_denoising_loop
        from audio_conditioning import AudioConditionByReferenceLatent
        from db_inference import estimate_speech_duration

        t_total = time.time()
        patchifier = AudioPatchifier(patch_size=1)

        # ── 1. Compute target shape ──────────────────────────────────────
        if gen_duration and gen_duration > 0:
            gen_dur = float(gen_duration)
        else:
            est_base = estimate_speech_duration(used_prompt, speed)
            gen_dur = max(3.0, round(est_base * duration_mult + _AUTO_DURATION_SAFETY_PAD_SEC, 1))
            logger.debug(
                "[DramaBox] Auto duration: base=%.1fs, multiplier=%.2f, speed=%.2f, safety=%.1fs -> target=%.1fs",
                est_base,
                duration_mult,
                speed,
                _AUTO_DURATION_SAFETY_PAD_SEC,
                gen_dur,
            )
        fps = 25.0
        n_frames = int(round(gen_dur * fps)) + 1
        n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1
        pixel_shape  = VideoPixelShape(batch=1, frames=n_frames, height=64, width=64, fps=fps)
        target_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        audio_tools  = AudioLatentTools(patchifier=patchifier, target_shape=target_shape)
        logger.debug("[DramaBox] target: %.1fs -> %d frames", gen_dur, n_frames)

        # ── 2. Initial latent state ──────────────────────────────────────
        state = audio_tools.create_initial_state(device=device, dtype=torch_dtype)

        # ── 3. Voice reference conditioning ─────────────────────────────
        mm.load_models_gpu(model["voice_patchers"])
        if voice_ref is not None:
            try:
                from ltx_pipelines.utils.media_io import decode_audio_from_file

                tmp_voice_path = _voice_ref_to_temp_wav(voice_ref)
                if not tmp_voice_path:
                    raise ValueError("voice_ref is missing waveform/sample_rate")

                try:
                    voice = decode_audio_from_file(tmp_voice_path, device, 0.0, ref_duration)
                finally:
                    try:
                        os.unlink(tmp_voice_path)
                    except OSError:
                        pass

                if voice is None:
                    raise ValueError("decode_audio_from_file returned no audio")

                logger.debug(
                    "[DramaBox] voice_ref decoded: sr=%d, waveform_shape=%s",
                    int(voice.sampling_rate),
                    tuple(voice.waveform.shape),
                )

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
                pre_tokens = int(state.latent.shape[1])
                ref_latent = audio_cond(lambda enc: vae_encode_audio(voice, enc, None))
                logger.debug(
                    "[DramaBox] reference latent encoded: shape=%s, dtype=%s, device=%s",
                    tuple(ref_latent.shape),
                    ref_latent.dtype,
                    ref_latent.device,
                )
                cond = AudioConditionByReferenceLatent(
                    latent=ref_latent.to(device, torch_dtype), strength=1.0
                )
                state = cond.apply_to(state, audio_tools)
                post_tokens = int(state.latent.shape[1])
                logger.debug(
                    "[DramaBox] conditioning applied: latent tokens %d -> %d",
                    pre_tokens,
                    post_tokens,
                )
                logger.debug("[DramaBox] Voice reference encoded (wrapper-style decode path).")
            except Exception as exc:
                logger.warning("[DramaBox] voice_ref failed (%s) — running without it", exc)

        # ── 4. Add noise ────────────────────────────────────────────────
        gen = torch.Generator(device=device).manual_seed(seed)
        state = GaussianNoiser(generator=gen)(state, noise_scale=1.0)

        # ── 5. Encode text prompts ───────────────────────────────────────
        # Use DramaBoxTEModel.encode_token_weights() for both backends so the
        # Comfy node path follows one shared, DramaBox-compatible routine.
        logger.debug("[DramaBox] Encoding prompts...")
        mm.load_models_gpu([_clip_enc.patcher])
        te_model = _clip_enc.cond_stage_model          # DramaBoxTEModel
        te_model.set_clip_options({"execution_device": _clip_enc.patcher.load_device})

        def _encode_one(prompt: str) -> torch.Tensor:
            tokens = _clip_enc.tokenize(prompt)
            audio_enc, _, _ = te_model.encode_token_weights(tokens)
            return audio_enc.to(device=device, dtype=torch_dtype)

        a_ctx     = _encode_one(used_prompt)
        a_ctx_neg = None
        if cfg_scale > 1.0:
            a_ctx_neg = _encode_one(negative_prompt)
        te_model.reset_clip_options()

        # Unified offload policy from Options (or preference-based default).
        if offload_policy in {"offload_to_cpu", "offload"}:
            _offload_clip_patcher_to_cpu(_clip_enc)

        mm.soft_empty_cache()

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
            separate_passes=False,
        )

        # ── 7. Diffusion sampling ────────────────────────────────────────
        # Apply LoRAs via manual delta math (matching py/db_inference.py's PEFT
        # approach). Deltas are added before moving the model to GPU and
        # subtracted after inference so the cached transformer stays unmodified.
        xfmr_patchers = model["xfmr_patchers"]
        _applied_deltas: list = []  # [(weight_key, delta_tensor), …]
        if normalized_lora_stack:
            for lora_name, strength_model, _strength_clip in normalized_lora_stack:
                lora_path = _resolve_lora_path(lora_name)
                if lora_path is None:
                    logger.warning("[DramaBox] LoRA not found, skipping: %s", lora_name)
                    continue
                deltas = _apply_lora_deltas(transformer, lora_path, float(strength_model))
                _applied_deltas.extend(deltas)
                logger.debug(
                    f"[DramaBox] LoRA applied: {os.path.basename(lora_path)} "
                    f"(strength={float(strength_model):.2f}, params={len(deltas)})"
                )
                if not deltas:
                    logger.warning(
                        f"[DramaBox] Warning: no matching transformer params found for "
                        f"{os.path.basename(lora_path)}"
                    )
        mm.load_models_gpu(xfmr_patchers)
        logger.info(
            "[DramaBox] Denoising (%d steps, cfg=%.1f, stg=%.1f)…",
            steps, cfg_scale, stg_scale,
        )
        sigmas = LTX2Scheduler().execute(steps=steps, latent=state.latent).to(device)
        x0 = X0Model(transformer)
        audio_state = None
        try:
            _, audio_state = euler_denoising_loop(
                sigmas=sigmas,
                video_state=None,
                audio_state=state,
                stepper=EulerDiffusionStep(),
                transformer=x0,
                denoiser=denoiser,
            )
        except RuntimeError as exc:
            if _is_cuda_cpu_device_mismatch(exc):
                clean_msg = _build_device_mismatch_message(generation_mode, offload_policy)
                logger.error("%s Original error: %s", clean_msg, exc)
                raise RuntimeError(clean_msg) from exc
            if _is_cuda_oom(exc):
                clean_msg = _build_cuda_oom_message(generation_mode, offload_policy)
                logger.error("%s Original error: %s", clean_msg, exc)
                raise RuntimeError(clean_msg) from exc
            raise
        finally:
            # Always restore cached transformer if LoRA deltas were applied.
            if _applied_deltas:
                _remove_lora_deltas(transformer, _applied_deltas)
                _applied_deltas.clear()
            mm.soft_empty_cache()

        if audio_state is None:
            raise RuntimeError("[DramaBox] Denoising finished without audio output state.")

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
        mm.load_models_gpu(model["dec_patchers"])
        logger.debug("[DramaBox] Decoding...")
        decoded = decoder(latent)

        mm.soft_empty_cache()

        # ── 10. Return as ComfyUI AUDIO ──────────────────────────────────
        waveform = decoded.waveform.cpu().float()
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)   # [B=1, C, S]
        sample_rate = decoded.sampling_rate
        audio_dur = waveform.shape[-1] / sample_rate
        elapsed = time.time() - t_total

        if offload_policy == "offload_to_cpu":
            logger.debug("[DramaBox] offload_policy=offload_to_cpu")
            _offload_dramabox_models_to_cpu()
        elif offload_policy == "offload":
            logger.debug("[DramaBox] offload_policy=offload")
            _unload_dramabox_models()

        return ({"waveform": waveform, "sample_rate": sample_rate},)


# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "DramaBoxTTS": DramaBoxTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DramaBoxTTS": "DramaBox TTS",
}
