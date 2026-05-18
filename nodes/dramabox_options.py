"""
DramaBox Options — optional configuration node for DramaBox TTS.

Mirrors the pattern of PromptGenOptions: all inputs are optional, the node
returns a plain dict that DramaBox TTS reads when connected.

rescale_scale uses the special sentinel value -1.0 to mean "auto" (the CFG-
aware schedule in inference_server.py). Set it to any value in [0, 1] to
override, or leave at -1 for the recommended automatic behaviour.
"""

import logging
import os

logger = logging.getLogger(__name__)


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
    """Read a boolean DramaBox setting with tolerant coercion."""
    v = _get_dramabox_setting(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def _default_offload_policy() -> str:
    """Default unified policy follows global auto-offload preference."""
    return "offload_to_cpu" if _get_dramabox_bool_setting("DramaBox.autoOffload", True) else "keep_loaded"


class DramaBoxOptions:
    """Advanced generation options for DramaBox TTS."""

    CATEGORY = "DramaBox"
    DESCRIPTION = (
        "Optional advanced controls for the DramaBox TTS node.\n"
        "All fields are optional — unconnected inputs use sensible defaults."
    )
    RETURN_TYPES = ("DRAMABOX_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "create_options"

    @classmethod
    def INPUT_TYPES(cls):
        default_offload_policy = _default_offload_policy()
        return {
            "optional": {
                # ── Generation ───────────────────────────────────────────────
                "steps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Denoising steps. 30 is recommended.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "worst quality, inconsistent, robotic, distorted, noise, static, muffled, unclear, unnatural, monotone"
                        ),
                        "tooltip": "What the model should avoid.",
                    },
                ),
                # ── Guidance ────────────────────────────────────────────────
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 2.5,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": (
                            "Classifier-Free Guidance scale.\n"
                            "Higher = more faithful to the prompt but less natural.\n"
                            "1.0 = disabled.  Recommended: 2.0–3.5."
                        ),
                    },
                ),
                "stg_scale": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": (
                            "Spatiotemporal Guidance scale.\n"
                            "Improves audio coherence at a small quality cost.\n"
                            "0.0 = disabled.  Recommended: 1.0–2.0."
                        ),
                    },
                ),
                "rescale_scale": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "CFG std-rescale — prevents clipping at high CFG values.\n"
                            "-1 (default) = auto schedule (recommended).\n"
                            "Set to 0.0 to disable, or [0, 1] to override manually."
                        ),
                    },
                ),
                "id_guidance_scale": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": (
                            "Identity guidance scale for voice-cloning fidelity.\n"
                            "Higher values push the output closer to the reference voice identity.\n"
                            "0.0 = disabled.  Default: 3.0."
                        ),
                    },
                ),
                # ── Duration ────────────────────────────────────────────────
                "gen_duration": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.5,
                        "tooltip": (
                            "Target output duration in seconds.\n"
                            "0 = auto-estimate from prompt text (recommended).\n"
                            "Set explicitly for music/long-form content (e.g. 30.0)."
                        ),
                    },
                ),
                "duration_multiplier": (
                    "FLOAT",
                    {
                        "default": 1.1,
                        "min": 0.5,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": (
                            "Scale factor applied to the auto-estimated duration.\n"
                            "1.1 = 10% extra breathing room (default).\n"
                            "Ignored when gen_duration > 0."
                        ),
                    },
                ),
                "speed": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": (
                            "Speaking rate multiplier for auto duration estimation.\n"
                            "1.0 = normal pace.  0.7 = slower / more deliberate.\n"
                            "1.3 = faster.  Ignored when gen_duration > 0."
                        ),
                    },
                ),
                # ── Voice reference ─────────────────────────────────────────
                "ref_duration": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 1.0,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": (
                            "How many seconds of the voice_ref audio to use.\n"
                            "5–15 s of clean speech gives the best cloning quality."
                        ),
                    },
                ),
                "post_generate_model_policy": (
                    ["keep_loaded", "offload_to_cpu", "offload"],
                    {
                        "default": default_offload_policy,
                        "advanced": True,
                        "tooltip": (
                            "Unified memory policy for both stages (text encode + post-generate):\n"
                            "keep_loaded: keep Gemma and DramaBox on GPU (fastest, highest VRAM).\n"
                            "offload_to_cpu: offload Gemma after encoding and DramaBox after generation.\n"
                            "offload: offload Gemma after encoding and fully unload DramaBox after generation.\n"
                            "Default follows Settings > DramaBox > Automatic model offload."
                        ),
                    },
                ),
                "attention_policy": (
                    ["best_available", "checkpoint_config", "force_fa2", "force_sdpa", "force_default"],
                    {
                        "default": "best_available",
                        "advanced": True,
                        "tooltip": (
                            "Transformer attention backend policy:\n"
                            "best_available: auto-select fastest supported backend (recommended).\n"
                            "checkpoint_config: use attention_type from checkpoint metadata.\n"
                            "force_fa2 / force_sdpa / force_default: force a backend when available;\n"
                            "falls back to checkpoint_config if unavailable."
                        ),
                    },
                ),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Options node has no side effects; always cache.
        return float("nan")

    def create_options(
        self,
        steps: int = 30,
        negative_prompt: str = (
            "worst quality, inconsistent, robotic, distorted, noise, static, muffled, unclear, unnatural, monotone"
        ),
        cfg_scale: float = 2.5,
        stg_scale: float = 1.5,
        rescale_scale: float = -1.0,
        id_guidance_scale: float = 3.0,
        gen_duration: float = 0.0,
        duration_multiplier: float = 1.1,
        speed: float = 1.0,
        ref_duration: float = 10.0,
        post_generate_model_policy: str = "",
        attention_policy: str = "best_available",
    ):
        valid_policies = {"keep_loaded", "offload_to_cpu", "offload"}
        if post_generate_model_policy not in valid_policies:
            post_generate_model_policy = _default_offload_policy()

        valid_attention_policies = {
            "best_available",
            "checkpoint_config",
            "force_fa2",
            "force_sdpa",
            "force_default",
        }
        attention_policy = str(attention_policy).strip().lower()
        if attention_policy not in valid_attention_policies:
            attention_policy = "best_available"

        options = {
            "steps": steps,
            "negative_prompt": negative_prompt,
            "cfg_scale": cfg_scale,
            "stg_scale": stg_scale,
            # -1.0 sentinel → pass "auto" string to TTSServer.generate()
            "rescale_scale": "auto" if rescale_scale < 0.0 else float(rescale_scale),
            "id_guidance_scale": id_guidance_scale,
            "gen_duration": gen_duration,
            "duration_multiplier": duration_multiplier,
            "speed": speed,
            "ref_duration": ref_duration,
            "post_generate_model_policy": str(post_generate_model_policy),
            "attention_policy": attention_policy,
        }
        return (options,)


# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "DramaBoxOptions": DramaBoxOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DramaBoxOptions": "DramaBox Options",
}
