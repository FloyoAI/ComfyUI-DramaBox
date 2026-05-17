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
  4. Returns audio_encoding as the primary cond, with extra dict carrying
     the full output for the TTS node.
"""

import glob
import itertools
import logging
import os
import sys

import torch

# ── ensure src/ and ltx2/ are on sys.path ────────────────────────────────────
_NODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("src", "ltx2"):
    _p = os.path.join(_NODE_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
from comfy.text_encoders.lt import Gemma3_12BModel, LTXAVGemmaTokenizer

logger = logging.getLogger(__name__)



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
    ):
        super().__init__()
        self.dtypes = set()
        self.dtypes.add(dtype)
        self.execution_device = None

        self.gemma3_12b = Gemma3_12BModel(
            device=device,
            dtype=dtype_llama,
            model_options=model_options,
            layer="all",
            layer_idx=None,
        )
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

    # ── ComfyUI CLIP options protocol ────────────────────────────────────────

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)
        self.gemma3_12b.set_clip_options(options)

    def reset_clip_options(self):
        self.gemma3_12b.reset_clip_options()
        self.execution_device = None

    # ── Core encoding ────────────────────────────────────────────────────────

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_g = token_weight_pairs["gemma3_12b"]

        # [B, L, T, D]  —  L=49 layers, T=seq_len, D=3840
        out, _pooled, extra = self.gemma3_12b.encode_token_weights(token_weight_pairs_g)

        # → [B, T, D, L]  (DramaBox FeatureExtractor accepts this directly)
        out = out.movedim(1, -1)

        # Binary attention mask [B, T] (1 = valid token, 0 = padding)
        attention_mask = extra["attention_mask"]

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
            {
                "dramabox_audio_encoding": ep_out.audio_encoding,
                "dramabox_video_encoding": ep_out.video_encoding,
                "dramabox_attention_mask": ep_out.attention_mask,
                "attention_mask": attention_mask,
            },
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
        # Gemma checkpoint — detected by a Gemma-specific key
        if "model.layers.47.self_attn.q_norm.weight" in sd:
            return self.gemma3_12b.load_sd(sd)

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


def dramabox_te(dtype_llama=None, llama_quantization_metadata=None, audio_components_path=None):
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
            "*dramabox*.safetensors",
        ):
            found.extend(glob.glob(os.path.join(folder, "**", pat), recursive=True))
    seen = set()
    result = []
    for p in found:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


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
                    folder_paths.get_filename_list("text_encoders"),
                    {"tooltip": "Gemma fp8/bf16 safetensors from ComfyUI/models/text_encoders/"},
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
        if os.path.isabs(gemma_model) and os.path.isfile(gemma_model):
            gemma_path = gemma_model
        else:
            gemma_path = folder_paths.get_full_path_or_raise("text_encoders", gemma_model)

        audio_path = self._find_audio_components()
        if audio_path is None:
            raise FileNotFoundError(
                "[DramaBox] No audio-components safetensors found.\n"
                "Ensure the DramaBox checkpoint has been downloaded into "
                "ComfyUI/models/checkpoints/ or the DramaBox models/ folder."
            )

        # Load both state dicts — exactly as load_clip / load_text_encoder_state_dicts does
        gemma_sd, _ = comfy.utils.load_torch_file(gemma_path, return_metadata=True)
        audio_sd, _ = comfy.utils.load_torch_file(audio_path, return_metadata=True)
        clip_data = [gemma_sd, audio_sd]

        # model_options: only CPU override if requested (same as LTXAVTextEncoderLoader)
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

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
        return (clip,)

    @staticmethod
    def _find_audio_components():
        """Scan known locations for the DramaBox audio-components safetensors."""
        candidates = _list_audio_component_files()
        if candidates:
            return candidates[0]
        # Last resort: model_downloader path (may trigger a download)
        try:
            from model_downloader import get_model_path
            p = get_model_path("audio_components", cache_dir=os.path.join(_NODE_DIR, "models"))
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
    "DramaBoxTextEncoderLoader": "DramaBox Text Encoder Loader",
}
