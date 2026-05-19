"""
DramaBox Gemma text encoder loader node — ComfyUI CLIP approach.

Wraps DramaBox's Gemma + EmbeddingsProcessor in a standard ComfyUI CLIP
object so the model benefits from full VRAM management (ModelPatcher,
CPU↔GPU offloading, fp8 support) — exactly like LTXAVTextEncoderLoader.

Architecture overview
---------------------
ComfyUI's Gemma3_12BModel (layer="all") returns hidden states of shape
[B, L, T, D] where L = num_layers (49), T = seq_len, D = hidden_dim (3840).

After movedim(1, -1) this becomes [B, T, D, L], which is exactly what
DramaBox's FeatureExtractor expects (it stacks only when given a tuple;
a plain tensor passes through directly).

DramaBoxTEModel.encode_token_weights() therefore:
  1. Calls gemma3_12b.encode_token_weights() → [B, L, T, D]
  2. movedim(1, -1)                          → [B, T, D, L]
  3. embeddings_processor.process_hidden_states() → EmbeddingsProcessorOutput
    4. Returns audio_encoding as the primary cond.
"""

import glob
import itertools
import logging
import os
import sys
import gc
import types

import torch

# ── local import bootstrap ────────────────────────────────────────────────────
_NODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Always make local py importable for node-local helpers.
_SRC_DIR = os.path.join(_NODE_DIR, "py")
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
from comfy.text_encoders.lt import Gemma3_12BModel, LTXAVGemmaTokenizer

logger = logging.getLogger(__name__)


def _fix_rope_position_ids(transformer) -> None:
    """Patch a LlamaModel instance to use HuggingFace-style RoPE position_ids.

    ComfyUI's LlamaModel.forward computes position_ids = arange(seq_len) when
    position_ids is None.  For a left-padded 1024-token sequence with 32 real
    tokens this assigns RoPE positions 992-1023 to the real tokens, whereas
    HuggingFace Gemma3 uses cumsum(attention_mask)-1 so real tokens always
    start at position 0.  DramaBox's EmbeddingsProcessor was trained with the
    HuggingFace convention, so we patch the transformer instance once at init
    time and leave all higher-level encode_token_weights calls unchanged.
    """
    if getattr(transformer, '_hf_rope_position_ids_fixed', False):
        return
    _orig_forward = transformer.forward

    def _fixed_forward(x, attention_mask=None, **kwargs):
        if kwargs.get('position_ids') is None and attention_mask is not None:
            kwargs['position_ids'] = (
                attention_mask.long().cumsum(dim=-1) - 1
            ).clamp(min=0)
        return _orig_forward(x, attention_mask, **kwargs)

    transformer.forward = _fixed_forward
    transformer._hf_rope_position_ids_fixed = True


def _find_bnb_gemma_root() -> "str | None":
    """Return path to the downloaded unsloth/gemma-3-12b-it-bnb-4bit directory.

    Does NOT trigger a download — only returns a path if the directory is
    already present on disk (i.e. the user has run the DramaBox wrapper at
    least once, which downloads it automatically).
    """
    try:
        _py_dir = os.path.join(_NODE_DIR, "py")
        if _py_dir not in sys.path:
            sys.path.insert(0, _py_dir)
        from model_downloader import GEMMA_REPO, _resolve_models_dir
        models_dir = _resolve_models_dir(None)
        gemma_name = GEMMA_REPO.split("/")[-1]
        candidate = models_dir / "dramabox" / gemma_name
        if candidate.is_dir() and any(candidate.iterdir()):
            return str(candidate)
    except Exception:
        pass
    return None


def _load_bnb_encoder(bnb_root: str, dtype: "torch.dtype", device):
    """Load a GemmaTextEncoder backed by the BNB NF4-quantized Gemma directory.

    This replicates dramabox_ltx_compat._load_bnb_4bit_encoder so that the
    clip_loader path produces hidden states identical to the wrapper path.
    """
    import json
    from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration
    from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
    from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
    from ltx_core.utils import find_matching_file

    try:
        import bitsandbytes as _bnb
        _bnb.functional.get_4bit_type("nf4")
    except Exception as exc:
        raise RuntimeError(
            f"[DramaBox] bitsandbytes is not available — cannot use BNB NF4 encoder: {exc}\n"
            "Install bitsandbytes or fall back to a safetensors Gemma file."
        ) from exc

    prequantized = False
    cfg_path = os.path.join(bnb_root, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            prequantized = "quantization_config" in cfg
        except Exception:
            prequantized = False

    from_kwargs: dict = {
        "device_map": str(device) if not isinstance(device, str) else device,
        "torch_dtype": dtype if dtype is not None else torch.bfloat16,
    }
    if not prequantized:
        from_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=from_kwargs["torch_dtype"],
        )

    logger.info("[DramaBox] Loading BNB NF4 Gemma encoder from %s", bnb_root)
    hf_model = Gemma3ForConditionalGeneration.from_pretrained(bnb_root, **from_kwargs)
    tokenizer = LTXVGemmaTokenizer(
        str(find_matching_file(bnb_root, "tokenizer.model").parent),
        1024,
    )
    return GemmaTextEncoder(model=hf_model, tokenizer=tokenizer, dtype=from_kwargs["torch_dtype"])



# ─────────────────────────────────────────────────────────────────────────────
# DramaBoxTEModel
# ─────────────────────────────────────────────────────────────────────────────

class DramaBoxTEModel(torch.nn.Module):
    """
    ComfyUI-native DramaBox text encoder.

    Combines ComfyUI's Gemma3_12BModel (VRAM management / fp8 support)
    with DramaBox's EmbeddingsProcessor (feature extractor + audio/video
    connectors).  Designed to be wrapped in comfy.sd.CLIP exactly as
    LTXAVTEModel is.
    """

    def __init__(
        self,
        dtype_llama=None,
        device="cpu",
        dtype=None,
        model_options={},
        audio_components_path=None,
        bnb_gemma_root=None,
    ):
        super().__init__()
        self.dtypes = set()
        self.dtypes.add(dtype)
        self.execution_device = None
        # _bnb_encoder is intentionally NOT stored via nn.Module.__setattr__ so
        # that ComfyUI's ModelPatcher never sees the BNB Params4bit weights.
        # We use object.__setattr__ to bypass nn.Module's registry entirely.
        # It is loaded lazily at first encode time (so it lands on the correct
        # GPU device rather than the "cpu" device used during init).
        object.__setattr__(self, '_bnb_encoder', None)

        if bnb_gemma_root is not None:
            # BNB mode: encode via HuggingFace BNB NF4-quantized Gemma model.
            # Gemma weights are NOT loaded into self.gemma3_12b — the BNB encoder
            # owns them.  We store the root path and dtype for lazy loading.
            self.gemma3_12b = None
            bnb_dtype = dtype_llama if dtype_llama is not None else (dtype or torch.bfloat16)
            object.__setattr__(self, '_bnb_gemma_root', bnb_gemma_root)
            object.__setattr__(self, '_bnb_dtype', bnb_dtype)
            self.dtypes.add(bnb_dtype)
        else:
            object.__setattr__(self, '_bnb_gemma_root', None)
            object.__setattr__(self, '_bnb_dtype', None)
            # ComfyUI mode: standard LlamaModel via Gemma3_12BModel.
            self.gemma3_12b = Gemma3_12BModel(
                device=device,
                dtype=dtype_llama,
                model_options=model_options,
                layer="all",
                layer_idx=None,
            )
            # Fix RoPE position_ids on the underlying LlamaModel so that
            # encode_token_weights works correctly for left-padded sequences.
            _fix_rope_position_ids(self.gemma3_12b.transformer)
            self.dtypes.add(dtype_llama)

        # Build EmbeddingsProcessor on the initial device.
        # Weights are random here and will be overwritten by load_sd().
        self.embeddings_processor = None
        if audio_components_path is not None:
            from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
            from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
                EmbeddingsProcessorConfigurator,
                EMBEDDINGS_PROCESSOR_KEY_OPS,
            )
            builder = SingleGPUModelBuilder(
                model_class_configurator=EmbeddingsProcessorConfigurator,
                model_path=audio_components_path,
                model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            )
            config = builder.model_config()
            with torch.device(device):
                self.embeddings_processor = EmbeddingsProcessorConfigurator.from_config(config)
            # Cast to bf16: checkpoint weights are bf16, and the video connector
            # has no weights in this checkpoint (audio-only) so its random-init
            # weights must also be bf16 to avoid dtype errors during forward.
            self.embeddings_processor = self.embeddings_processor.to(torch.bfloat16)
            self._enable_audio_only_embeddings()

    def _enable_audio_only_embeddings(self):
        """Drop video-only EP branches to match DramaBox audio-only usage."""
        ep = self.embeddings_processor
        if ep is None:
            return

        freed = 0

        # 1) Remove video connector weights.
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

        # 2) Replace video aggregate projection with a tiny no-op module.
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
            out_features = fe.video_aggregate_embed.out_features
            del fe.video_aggregate_embed

            class _DummyVideoEmbed(torch.nn.Module):
                def __init__(self, out_f):
                    super().__init__()
                    self.out_features = out_f

                def forward(self, x):
                    return torch.zeros(
                        x.shape[0], x.shape[1], self.out_features,
                        device=x.device, dtype=x.dtype,
                    )

            fe.video_aggregate_embed = _DummyVideoEmbed(out_features)

        # 2b) Patch feature extractor to compute audio branch only.
        # This avoids allocating full-size video features during prompt encoding.
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
                logger.debug(
                    "[DramaBox] encoded shape=%s dtype=%s is_tuple=%s "
                    "mean=%.4f std=%.4f",
                    tuple(encoded.shape), encoded.dtype,
                    isinstance(hidden_states, (list, tuple)),
                    encoded.float().mean().item(), encoded.float().std().item(),
                )
                normed = norm_and_concat_per_token_rms(encoded, attention_mask).to(encoded.dtype)
                logger.debug(
                    "[DramaBox] normed shape=%s mean=%.4f std=%.4f",
                    tuple(normed.shape), normed.float().mean().item(), normed.float().std().item(),
                )
                a_dim = self.audio_aggregate_embed.out_features
                audio = self.audio_aggregate_embed(_rescale_norm(normed, a_dim, self.embedding_dim))
                logger.debug(
                    "[DramaBox] audio shape=%s mean=%.4f std=%.4f",
                    tuple(audio.shape), audio.float().mean().item(), audio.float().std().item(),
                )
                video = audio.new_zeros((audio.shape[0], audio.shape[1], 1))
                return video, audio

            fe.forward = types.MethodType(_forward_audio_only, fe)

        # 3) Patch create_embeddings to skip video connector entirely.
        def _audio_only_create(video_features, audio_features, additive_attention_mask, _ep=ep):
            m = additive_attention_mask
            while m.dim() > 2:
                m = m[:, 0]
            binary_mask = (m >= -1.0).to(torch.int64)

            audio_encoded = None
            if _ep.audio_connector is not None:
                audio_encoded, _ = _ep.audio_connector(audio_features, additive_attention_mask)

            # Keep API contract: return video tensor, audio tensor, and binary mask.
            # DramaBox TTS consumes only audio_encoding.
            return video_features, audio_encoded, binary_mask

        ep.create_embeddings = _audio_only_create

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if freed:
            logger.info("[DramaBox] Text encoder audio-only mode enabled; dropped ~%.2f GB video branch", freed / 1e9)

    # ── ComfyUI CLIP options protocol ────────────────────────────────────────

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)
        if self.gemma3_12b is not None:
            self.gemma3_12b.set_clip_options(options)

    def reset_clip_options(self):
        if self.gemma3_12b is not None:
            self.gemma3_12b.reset_clip_options()
        self.execution_device = None

    # ── Core encoding ────────────────────────────────────────────────────────

    def _encode_token_weights_bnb(self, token_weight_pairs):
        """Encode via the BNB NF4 HuggingFace Gemma model.

        Extracts token IDs and the attention mask from the ComfyUI
        token_weight_pairs (same tokenizer, same IDs) and calls the HuggingFace
        model's inner language model directly, bypassing the lm_head to save
        memory.  Returns hidden states in the same [B, T, D, L] format that
        the EmbeddingsProcessor expects.
        """
        # Lazy-load the BNB encoder on first call so it lands on the GPU
        # device ComfyUI selects rather than the cpu used during init.
        # _bnb_encoder is stored via object.__setattr__ so ModelPatcher
        # never sees the BNB Params4bit weights.
        if self._bnb_encoder is None:
            dev = comfy.model_management.get_torch_device()
            encoder = _load_bnb_encoder(self._bnb_gemma_root, self._bnb_dtype, dev)
            object.__setattr__(self, '_bnb_encoder', encoder)
        token_weight_pairs_g = token_weight_pairs["gemma3_12b"]

        # token_weight_pairs_g is a list (batch) of lists of (token_id, weight).
        # The weight is the attention mask value (1 = real token, 0 = padding).
        input_ids_list   = [[int(t[0]) for t in seq] for seq in token_weight_pairs_g]
        attn_mask_list   = [[int(bool(t[1])) for t in seq] for seq in token_weight_pairs_g]

        hf_model = self._bnb_encoder.model
        dev = next(hf_model.parameters()).device
        input_ids    = torch.tensor(input_ids_list,  dtype=torch.long, device=dev)
        attn_mask    = torch.tensor(attn_mask_list,  dtype=torch.long, device=dev)

        with torch.inference_mode():
            outputs = hf_model.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

        # outputs.hidden_states: tuple of (num_layers+1) tensors [B, T, D]
        # Stack → [B, T, D, L] which is what EmbeddingsProcessor.process_hidden_states expects.
        out = torch.stack(outputs.hidden_states, dim=-1).float()  # [B, T, D, L]
        del outputs

        # Wrapper parity: keep the full padded sequence and let
        # EmbeddingsProcessor handle masking internally.
        attention_mask = attn_mask.cpu()

        ep_device = next(self.embeddings_processor.parameters()).device
        ep_dtype  = next(self.embeddings_processor.parameters()).dtype
        ep_out = self.embeddings_processor.process_hidden_states(
            out.to(device=ep_device, dtype=ep_dtype),
            attention_mask.to(ep_device),
            "left",
        )

        return (ep_out.audio_encoding, None, {})

    def encode_token_weights(self, token_weight_pairs):
        if self._bnb_gemma_root is not None:
            return self._encode_token_weights_bnb(token_weight_pairs)

        token_weight_pairs_g = token_weight_pairs["gemma3_12b"]

        # [B, L, T, D]  —  L=49 layers, T=seq_len, D=3840
        # The underlying LlamaModel has been patched at init time (_fix_rope_position_ids)
        # to use HF-style cumsum position_ids, so this call is correct as-is.
        out, _pooled, extra = self.gemma3_12b.encode_token_weights(token_weight_pairs_g)
        out = out.float()

        # Binary attention mask [B, T] (1 = valid token, 0 = padding)
        attention_mask = extra["attention_mask"]

        # → [B, T, D, L]  (DramaBox FeatureExtractor accepts this directly)
        out = out.movedim(1, -1)

        # Gemma returns activations on intermediate_device (CPU by default).
        # EmbeddingsProcessor weights live on the execution device (GPU) and
        # are bf16. Align device AND dtype before calling the EP.
        ep_device = next(self.embeddings_processor.parameters()).device
        ep_dtype  = next(self.embeddings_processor.parameters()).dtype
        ep_out = self.embeddings_processor.process_hidden_states(
            out.to(device=ep_device, dtype=ep_dtype),
            attention_mask.to(ep_device),
            "left"
        )

        return (
            ep_out.audio_encoding,
            None,
            {},
        )

    # ── Weight loading ───────────────────────────────────────────────────────

    # Checkpoint key → EmbeddingsProcessor submodule key mapping.
    # Mirrors DramaBox's EMBEDDINGS_PROCESSOR_KEY_OPS.
    _EP_KEY_MAP = {
        "text_embedding_projection.aggregate_embed.":
            "feature_extractor.aggregate_embed.",
        "text_embedding_projection.video_aggregate_embed.":
            "feature_extractor.video_aggregate_embed.",
        "text_embedding_projection.audio_aggregate_embed.":
            "feature_extractor.audio_aggregate_embed.",
        "model.diffusion_model.video_embeddings_connector.":
            "video_connector.",
        "model.diffusion_model.audio_embeddings_connector.":
            "audio_connector.",
    }

    def load_sd(self, sd):
        # Gemma checkpoint — detected by a Gemma-specific key.
        # In BNB mode the Gemma weights are already loaded from the BNB directory;
        # skip the safetensors Gemma SD entirely (it was only loaded to satisfy
        # ComfyUI's CLIP state-dict pipeline and can be discarded).
        if "model.layers.47.self_attn.q_norm.weight" in sd:
            if self._bnb_gemma_root is not None:
                return ([], [])  # BNB mode — Gemma weights are irrelevant here
            missing, unexpected = self.gemma3_12b.load_sd(sd)
            # DramaBox uses Gemma as a text-only encoder.  GGUF and safetensors
            # exports of Gemma 3 12B omit the vision tower; filter those keys so
            # ComfyUI doesn't log spurious "clip missing" warnings for them.
            missing = [k for k in missing if not (
                k.startswith("vision_model.") or
                k.startswith("multi_modal_projector.")
            )]
            return (missing, unexpected)

        if self.embeddings_processor is None:
            return ([], [])

        # Remap audio-components keys → EmbeddingsProcessor submodule keys
        ep_sd = {}
        for key, value in sd.items():
            for old_pfx, new_pfx in self._EP_KEY_MAP.items():
                if key.startswith(old_pfx):
                    ep_sd[new_pfx + key[len(old_pfx):]] = value
                    break

        if not ep_sd:
            return ([], [])

        missing, unexpected = self.embeddings_processor.load_state_dict(
            ep_sd, strict=False, assign=getattr(self, "can_assign_sd", False)
        )
        # video_connector and feature_extractor.video_aggregate_embed weights
        # do not exist in the audio-only DramaBox checkpoint — filter them out
        # so ComfyUI doesn't log spurious "clip missing" warnings.
        missing = [k for k in missing if not (
            k.startswith("video_connector.") or
            k.startswith("feature_extractor.video_aggregate_embed.")
        )]
        return (list(missing), list(unexpected))

    # ── Memory estimation (mirrors LTXAVTEModel) ─────────────────────────────

    def memory_estimation_function(self, token_weight_pairs, device=None):
        constant = 6.0
        if comfy.model_management.should_use_bf16(device):
            constant /= 2.0
        token_weight_pairs_g = token_weight_pairs.get("gemma3_12b", [])
        if not token_weight_pairs_g:
            return 0
        m = min(
            sum(1 for _ in itertools.takewhile(lambda x: x[0] == 0, sub))
            for sub in token_weight_pairs_g
        )
        num_tokens = sum(len(a) for a in token_weight_pairs_g) - m
        return max(num_tokens, 642) * constant * 1024 * 1024


def dramabox_te(
    dtype_llama=None,
    llama_quantization_metadata=None,
    audio_components_path=None,
    bnb_gemma_root=None,
):
    """Factory that returns a DramaBoxTEModel subclass capturing checkpoint params."""

    class DramaBoxTEModel_(DramaBoxTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            mo = model_options
            if llama_quantization_metadata is not None:
                mo = mo.copy()
                mo["llama_quantization_metadata"] = llama_quantization_metadata
            dtype_eff = dtype_llama if dtype_llama is not None else dtype
            super().__init__(
                dtype_llama=dtype_llama,
                device=device,
                dtype=dtype_eff,
                model_options=mo,
                audio_components_path=audio_components_path,
                bnb_gemma_root=bnb_gemma_root,
            )

    return DramaBoxTEModel_


# ─────────────────────────────────────────────────────────────────────────────
# File discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def _list_audio_component_files():
    try:
        dirs = list(folder_paths.get_folder_paths("checkpoints"))
    except Exception:
        dirs = []
    dirs.append(os.path.join(_NODE_DIR, "models"))
    found = []
    for folder in dirs:
        for pat in (
            "*audio*component*.safetensors",
            "*audio*components*.safetensors",
            "*dramabox*audio*.safetensors",
        ):
            found.extend(glob.glob(os.path.join(folder, "**", pat), recursive=True))
    seen = set()
    result = []
    for p in found:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _safe_filename_list(key: str) -> list[str]:
    try:
        return list(folder_paths.get_filename_list(key))
    except Exception:
        return []


def _get_gemma_model_choices() -> list[str]:
    """Return selectable Gemma model filenames (safetensors + optional gguf).

    Only files whose basename contains 'gemma' (case-insensitive) are included
    so that unrelated text encoders (T5, CLIP-L, LTX projectors, …) don't
    clutter the dropdown.  Searched across text_encoders, clip_gguf, and clip
    so that files stored in any of these ComfyUI folders are found.
    """
    names: set[str] = set()
    for name in _safe_filename_list("text_encoders"):
        if "gemma" in name.lower():
            names.add(name)
    for key in ("clip_gguf", "clip"):
        for name in _safe_filename_list(key):
            if "gemma" in name.lower():
                names.add(name)
    return sorted(names, key=str.casefold)


def _resolve_gemma_model_path(gemma_model: str) -> str:
    """Resolve a Gemma model name or absolute path across known Comfy folders."""
    if os.path.isabs(gemma_model) and os.path.isfile(gemma_model):
        return gemma_model

    for key in ("text_encoders", "clip_gguf", "clip"):
        try:
            p = folder_paths.get_full_path(key, gemma_model)
        except Exception:
            p = None
        if p and os.path.isfile(p):
            return p

    # Keep legacy behavior for existing workflows that assume text_encoders key.
    return folder_paths.get_full_path_or_raise("text_encoders", gemma_model)


def _discover_gguf_bridge():
    """Find ComfyUI-GGUF symbols if the extension is installed and loaded."""
    gguf_clip_loader = None
    gguf_ops = None
    gguf_model_patcher = None

    for mod in tuple(sys.modules.values()):
        if mod is None:
            continue

        # Avoid torch.ops dynamic namespaces (they pretend every attribute exists).
        mod_name = str(getattr(mod, "__name__", ""))
        mod_file = str(getattr(mod, "__file__", ""))
        mod_hint = f"{mod_name} {mod_file}".lower()
        if "gguf" not in mod_hint:
            continue

        if gguf_clip_loader is None and hasattr(mod, "gguf_clip_loader"):
            candidate = getattr(mod, "gguf_clip_loader")
            if callable(candidate) and candidate.__class__.__name__ != "_OpNamespace":
                gguf_clip_loader = candidate

        if gguf_ops is None and hasattr(mod, "GGMLOps"):
            candidate = getattr(mod, "GGMLOps")
            if candidate.__class__.__name__ != "_OpNamespace" and hasattr(candidate, "Embedding"):
                gguf_ops = candidate

        if gguf_model_patcher is None and hasattr(mod, "GGUFModelPatcher"):
            candidate = getattr(mod, "GGUFModelPatcher")
            if candidate.__class__.__name__ != "_OpNamespace" and hasattr(candidate, "clone"):
                gguf_model_patcher = candidate

        if gguf_clip_loader is not None and gguf_ops is not None and gguf_model_patcher is not None:
            break

    return gguf_clip_loader, gguf_ops, gguf_model_patcher


# ─────────────────────────────────────────────────────────────────────────────
# Loader node
# ─────────────────────────────────────────────────────────────────────────────

class DramaBoxTextEncoderLoader:
    """
    Load the DramaBox Gemma text encoder as a standard ComfyUI CLIP.

    Uses ComfyUI's ModelPatcher infrastructure for proper GPU↔CPU offloading
    and fp8 support — identical approach to LTXAVTextEncoderLoader.

    Connect the CLIP output to DramaBox TTS's ``clip`` input.
    """

    CATEGORY = "DramaBox"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load"
    DESCRIPTION = (
        "Loads the DramaBox Gemma text encoder as a ComfyUI CLIP with full VRAM management.\n"
        "Connect to DramaBox TTS's clip input."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemma_model": (
                    _get_gemma_model_choices(),
                    {
                        "tooltip": (
                            "Gemma model from ComfyUI/models/text_encoders/ "
                            "(safetensors or gguf when ComfyUI-GGUF is installed)."
                        ),
                    },
                ),
            },
            "optional": {
                "device": (
                    ["default", "cpu"],
                    {"advanced": True, "tooltip": "Force CPU offload for the text encoder."},
                ),
            },
        }

    def load(self, gemma_model, device="default"):
        # ── mirror LTXAVTextEncoderLoader / load_text_encoder_state_dicts ──
        # Accept either a bare filename (from the dropdown) or a full absolute
        # path (when called programmatically from _load_text_encoder).
        gemma_path = _resolve_gemma_model_path(gemma_model)
        is_gemma_gguf = gemma_path.lower().endswith(".gguf")

        audio_path = self._find_audio_components()
        if audio_path is None:
            raise FileNotFoundError(
                "[DramaBox] No audio-components safetensors found.\n"
                "Ensure the DramaBox checkpoint has been downloaded into "
                "ComfyUI/models/checkpoints/ or the DramaBox models/ folder."
            )

        # model_options: only CPU override if requested (same as LTXAVTextEncoderLoader)
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        # ── Safetensors / GGUF mode ──

        # Load Gemma state dict from safetensors or GGUF.
        gguf_model_patcher = None
        if is_gemma_gguf:
            gguf_clip_loader, gguf_ops, gguf_model_patcher = _discover_gguf_bridge()
            if gguf_clip_loader is None or gguf_ops is None:
                raise RuntimeError(
                    "[DramaBox] Selected Gemma GGUF model but ComfyUI-GGUF is not available.\n"
                    "Install ComfyUI-GGUF and restart ComfyUI, or select a safetensors Gemma model."
                )
            if not callable(gguf_clip_loader):
                raise RuntimeError(
                    f"[DramaBox] Invalid GGUF bridge: gguf_clip_loader is not callable ({type(gguf_clip_loader)})."
                )
            gemma_sd = gguf_clip_loader(gemma_path)
            model_options["custom_operations"] = gguf_ops
        else:
            gemma_sd, _ = comfy.utils.load_torch_file(gemma_path, safe_load=True, return_metadata=True)

        # Audio components remain safetensors for DramaBox at this time.
        audio_sd, _ = comfy.utils.load_torch_file(audio_path, safe_load=True, return_metadata=True)

        # Mirror comfy.sd.load_clip quant conversion behavior.
        if model_options.get("custom_operations", None) is None:
            gemma_sd, _ = comfy.utils.convert_old_quants(gemma_sd, model_prefix="", metadata=None)
            audio_sd, _ = comfy.utils.convert_old_quants(audio_sd, model_prefix="", metadata=None)

        clip_data = [gemma_sd, audio_sd]

        # Detect fp8/quantization metadata — use sd.llama_detect exactly as
        # load_text_encoder_state_dicts does (scans all clip_data internally)
        from comfy.sd import llama_detect as _sd_llama_detect
        llama_info = _sd_llama_detect(clip_data)

        # Build clip_target — mirrors load_text_encoder_state_dicts for CLIPType.LTXV
        class _ClipTarget:
            pass

        clip_target = _ClipTarget()
        clip_target.params = {}
        clip_target.clip = dramabox_te(
            dtype_llama=llama_info.get("dtype_llama"),
            llama_quantization_metadata=llama_info.get("llama_quantization_metadata"),
            audio_components_path=audio_path,
        )
        clip_target.tokenizer = LTXAVGemmaTokenizer

        tokenizer_data = {"spiece_model": gemma_sd.get("spiece_model", None)}

        # Compute parameters from ALL state dicts — exactly as load_text_encoder_state_dicts does
        # Also call model_options_long_clip per state dict (no-op for Gemma, matches native path)
        import comfy.text_encoders.long_clipl as _long_clipl
        parameters = 0
        for c in clip_data:
            parameters += comfy.utils.calculate_parameters(c)
            tokenizer_data, model_options = _long_clipl.model_options_long_clip(
                c, tokenizer_data, model_options
            )

        clip = comfy.sd.CLIP(
            clip_target,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            parameters=parameters,
            tokenizer_data=tokenizer_data,
            state_dict=clip_data,
            model_options=model_options,
        )
        if is_gemma_gguf and gguf_model_patcher is not None:
            try:
                clip.patcher = gguf_model_patcher.clone(clip.patcher)
            except Exception as e:
                logger.debug("[DramaBox] GGUF patcher clone skipped: %s", e)

        # Keep source metadata so downstream nodes can display the model name.
        try:
            clip.dramabox_model_path = gemma_path
            clip.dramabox_model_name = os.path.basename(gemma_path)
        except Exception:
            pass

        return (clip,)

    def _load_bnb(self, bnb_root: str, audio_path: str, model_options: dict):
        """Build a ComfyUI CLIP backed by the BNB NF4-quantized Gemma directory.

        The BNB model is loaded directly into DramaBoxTEModel via _load_bnb_encoder;
        only the EmbeddingsProcessor weights come from audio_path.  The SentencePiece
        tokenizer model is read from the BNB directory so no separate safetensors is needed.
        """
        import comfy.text_encoders.long_clipl as _long_clipl
        from ltx_core.utils import find_matching_file

        logger.info("[DramaBox] Using BNB NF4 Gemma encoder from %s", bnb_root)

        # Load only the EP (audio components) state dict.
        audio_sd, _ = comfy.utils.load_torch_file(audio_path, safe_load=True, return_metadata=True)
        audio_sd, _ = comfy.utils.convert_old_quants(audio_sd, model_prefix="", metadata=None)

        # Load the SentencePiece tokenizer binary from the BNB directory.
        spiece_path = str(find_matching_file(bnb_root, "tokenizer.model"))
        with open(spiece_path, "rb") as f:
            spiece_bytes = f.read()
        spiece_tensor = torch.frombuffer(bytearray(spiece_bytes), dtype=torch.uint8)

        clip_data = [audio_sd]

        class _ClipTarget:
            pass

        clip_target = _ClipTarget()
        clip_target.params = {}
        clip_target.clip = dramabox_te(
            dtype_llama=None,
            llama_quantization_metadata=None,
            audio_components_path=audio_path,
            bnb_gemma_root=bnb_root,
        )
        clip_target.tokenizer = LTXAVGemmaTokenizer

        tokenizer_data = {"spiece_model": spiece_tensor}
        for c in clip_data:
            tokenizer_data, model_options = _long_clipl.model_options_long_clip(
                c, tokenizer_data, model_options
            )

        parameters = sum(comfy.utils.calculate_parameters(c) for c in clip_data)

        clip = comfy.sd.CLIP(
            clip_target,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            parameters=parameters,
            tokenizer_data=tokenizer_data,
            state_dict=clip_data,
            model_options=model_options,
        )

        # Keep source metadata so downstream nodes can display the model name.
        try:
            clip.dramabox_model_path = bnb_root
            clip.dramabox_model_name = os.path.basename(os.path.normpath(bnb_root))
        except Exception:
            pass

        return (clip,)

    @staticmethod
    def _find_audio_components():
        """Scan known locations for the DramaBox audio-components safetensors."""
        candidates = _list_audio_component_files()
        if candidates:
            def _score(path):
                name = os.path.basename(path).lower()
                if "audio" in name and "component" in name:
                    return (0, name)
                if "dit" in name or "transformer" in name:
                    return (2, name)
                return (1, name)

            candidates = sorted(candidates, key=_score)
            best = candidates[0]
            if _score(best)[0] < 2:
                return best
        # Last resort: model_downloader path (may trigger a download)
        try:
            from model_downloader import get_model_path
            p = get_model_path("audio_components")
            if os.path.isfile(p):
                return p
        except Exception:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "DramaBoxTextEncoderLoader": DramaBoxTextEncoderLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DramaBoxTextEncoderLoader": "DramaBox CLIP Loader",
}
