"""Utility node to unload DramaBox model caches.

This node is a any: it accepts any input type and returns it unchanged,
while forcing cleanup of both clip_loader and wrapper (OG) model caches.
"""

import logging


logger = logging.getLogger(__name__)


class DramaBoxUnload:
    """Force unload DramaBox caches while passing data through."""

    CATEGORY = "DramaBox"
    DESCRIPTION = "Unload DramaBox models and pass input through unchanged."
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("any",)
    FUNCTION = "unload"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (
                    "*",
                    {
                        "tooltip": "Any input. Value is returned unchanged after DramaBox cache unload.",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, *_args, **_kwargs):
        return True

    def unload(self, any):
        import comfy.model_management as mm
        from . import dramabox_tts

        clip_released, og_released = dramabox_tts.unload_all_dramabox_caches()
        mm.soft_empty_cache()

        total = clip_released + og_released
        if total > 0:
            logger.info(
                "[DramaBox] Unload node released clip=%d, og=%d cache entries",
                clip_released,
                og_released,
            )
        else:
            logger.info("[DramaBox] Unload node found no cached DramaBox models")

        return (any,)


NODE_CLASS_MAPPINGS = {
    "DramaBoxUnload": DramaBoxUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DramaBoxUnload": "DramaBox Unload",
}
