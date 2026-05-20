# ComfyUI-DramaBox

ComfyUI custom nodes for [DramaBox](https://github.com/resemble-ai/DramaBox) — ResembleAI's expressive text-to-speech system built on the LTX-2.3 audio diffusion transformer.

**Please Note:** DramaBox offers two generation methods. `dramabox_wrapper` runs the original DramaBox pipeline for native behavior, while `clip_loader` uses ComfyUI memory management for a more integrated and reliable workflow. Users wanting to experiment, can enable wrapper mode in Preferences or in the DramaBox Options node.

**VRAM Recommendation:** For the smoothest experience, a **16 GB GPU** is strongly recommended. Lower-VRAM GPUs can still work with aggressive offloading, but may be slower and more prone to out-of-memory errors.

**Users with sufficient VRAM can disable model offloading in Preferences for faster generation.**

## Nodes

| Node | Description |
|------|-------------|
| **DramaBox TTS** | Generates speech audio from a text prompt. Optionally accepts a voice reference clip and advanced options. All model weights are downloaded automatically on first use. |
| **DramaBox CLIP Loader** | Loads a Gemma text encoder from your `text_encoders` folder. Supports `.safetensors` and optional `.gguf` (when ComfyUI-GGUF is installed). Connect to the TTS node's `dramabox_clip` input to override the default encoder. |
| **DramaBox Options** | Advanced generation settings (steps, CFG scale, duration, memory policy, etc.). (Optional) Connect to the DramaBox TTS node's `options` input. |
| **DramaBox Unload** | Utility passthrough node that releases both clip_loader and wrapper caches, then returns input unchanged. Useful at workflow boundaries to force memory cleanup. |

## Text Encoder

DramaBox uses a Gemma 3 12B text encoder. By default the node loads **`gemma_3_12B_it_fp4_mixed.safetensors`** — the same file used by ComfyUI's own LTX-2 workflows, so if you already have it you're good to go. If it is not present it is downloaded automatically into your `ComfyUI/models/text_encoders/` folder on first use.

### Changing the default encoder

**Per-installation preference** — open *ComfyUI Settings → DramaBox → Default Text Encoder filename* and enter a Gemma filename from your models folder. Supports `.safetensors` and `.gguf` (with ComfyUI-GGUF installed). If you omit the extension, DramaBox tries `.safetensors` first, then `.gguf`. Leave it blank to keep the fp4 default.

**Memory preference** — open *ComfyUI Settings → DramaBox → Memory* and keep **Automatic model offload (text encoder + post-generate)** enabled (default). This does two things automatically:

- Offloads Gemma right after prompt encoding.
- Offloads DramaBox models to CPU after generation.

If you have plenty of VRAM and prefer maximum throughput, you can disable this preference to keep models resident on GPU.

**Per-workflow override** — add a **DramaBox CLIP Loader** node, select the model you want (`.safetensors` or optional `.gguf`), and connect its output to the TTS node's `dramabox_clip` input. This takes precedence over the global preference and lets you switch encoders between workflows without touching settings.

### Post-generate memory policy (Options node)

The **DramaBox Options** node includes `post_generate_model_policy` with three modes:

- `keep_loaded` — keep Gemma and DramaBox on GPU (fastest next run, highest VRAM usage).
- `offload_to_cpu` — offload Gemma after text encoding and offload DramaBox to CPU after generation. (Default)
- `offload` — offload Gemma after text encoding and fully unload DramaBox after generation.

For most users, `offload_to_cpu` is a good balance between memory savings and iteration speed.

## Generation modes (advantages / disadvantages)

Set via **DramaBox Options** (`generation_mode`) or globally in *ComfyUI Settings → DramaBox → Use DramaBox Wrapper mode by default*.

### `clip_loader`

Advantages:
- Best ComfyUI-native VRAM control.
- Text encoding and diffusion are staged cleanly (less overlap in VRAM).
- Best fit when using custom CLIP Loader / GGUF text encoders.

Disadvantages:
- Not the original native DramaBox warm-server path.
- Output can differ from native DramaBox (tone, pacing, or phrasing may vary).

### `dramabox_wrapper`

Advantages:
- Closest to original DramaBox warm-server behavior.
- Best match when you want native DramaBox-like output.

Disadvantages:
- In `keep_loaded`, prompt encoder and transformer can overlap in VRAM.
- ComfyUI cannot fully manage wrapper internals like it does patcher-based models.

Wrapper + memory policy notes:
- `keep_loaded`: fastest repeated runs, highest VRAM use.
- `offload_to_cpu` / `offload`: uses one-shot low-memory execution (including LoRA entries) to avoid warm overlap, then releases wrapper cache.
- Add **DramaBox Unload** to force cleanup at a specific workflow point.

<div align="center">
  <img src="docs/images/example.png" alt="DramaBox">
</div>

## LoRA Support

The **DramaBox TTS** node accepts a `lora_stack` input (connect any ComfyUI **LORA_STACK** output). LoRA weights are applied directly into the already-loaded model and removed immediately after generation, so you can switch LoRAs between runs without triggering a slow model reload.

LoRAs and voice reference samples work independently and can be used together. A LoRA bakes a trained voice style into the model weights, while a voice reference sample is fed as audio conditioning during generation. Using both at once — for example a LoRA trained on a voice alongside a reference clip of that same voice — will reinforce the effect.

> **Note:** DramaBox LoRAs are specific to this model and cannot be used with other ComfyUI nodes such as LTX Video.


### Training your own voice LoRAs

Voice LoRAs for DramaBox can be trained with **[Voice Clone Studio — DramaBox Edition](https://github.com/FranckyB/Voice-Clone-Studio-DramaBox)**, a stripped-down version of Voice Clone Studio, made for DramaBox, for both inference and LoRA training. It provides a dataset creator that automatically transcribes and splits your long audio clips into smaller clips ready to be used by the trainer. Drop the resulting `.safetensors` file into ComfyUI's `models/loras/` folder.

## Installation

Follow these steps:

### 1) Add DramaBox to `custom_nodes`

Open a terminal in your ComfyUI `custom_nodes` folder, then run:

```bash
git clone https://github.com/FranckyB/ComfyUI-DramaBox.git
```

### 2) Install requirements with the SAME Python ComfyUI uses

Choose **one** option below that matches your ComfyUI install.

#### A) ComfyUI Windows Portable

1. Open your `ComfyUI_windows_portable` folder.
2. Open a terminal in that folder.
3. Run:

```bat
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-DramaBox\requirements.txt
```

(If your folder is named `python_embedded`, use that instead of `python_embeded`.)

#### B) ComfyUI Desktop

1. Open the folder where ComfyUI Desktop is installed.
2. Open a terminal in that folder.
3. Run (adjust the path if your layout differs):

```bat
python\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-DramaBox\requirements.txt
```

#### C) Manual Python / venv install

1. Open a terminal in your ComfyUI folder.
2. Activate the same environment you use to launch ComfyUI.
3. Install the requirements:

**Windows (cmd):**

```bat
cd C:\path\to\ComfyUI
venv\Scripts\activate
pip install -r custom_nodes\ComfyUI-DramaBox\requirements.txt
```

**Linux / macOS (bash/zsh):**

```bash
cd /path/to/ComfyUI
source venv/bin/activate
pip install -r custom_nodes/ComfyUI-DramaBox/requirements.txt
```

### 3) Restart ComfyUI

After restart, search for `DramaBox TTS` in the node menu.

### Quick troubleshooting

- If install fails, check that the requirements path is correct.
- If ComfyUI starts but node is missing, confirm folder is `custom_nodes/ComfyUI-DramaBox` (not nested like `ComfyUI-DramaBox-main/ComfyUI-DramaBox`).
- If `pip` command installs to the wrong place, you are likely using a different Python than ComfyUI. Re-run using option A, B, or your active venv Python.

## Changelog

### May 2026
- **Generation mode control** — added `generation_mode` (`clip_loader` or `dramabox_wrapper`) in DramaBox Options plus a global default wrapper-mode preference.
- **Wrapper LoRA parity** — wrapper mode now supports direct LoRA stack/strength application and cleanup, matching non-wrapper behavior.
- **Wrapper offload behavior** — when wrapper mode uses `offload_to_cpu` / `offload`, it now runs through one-shot low-memory execution (with LoRA support) to reduce VRAM overlap.
- **DramaBox Unload node** — added a passthrough unload node to release both clip and wrapper caches on demand.
- **Text encoder overhaul** — DramaBox now uses ComfyUI's standard CLIP infrastructure for the Gemma text encoder, matching the native LTX-2 loading path for correct VRAM management.
- **Default encoder** — switched to `gemma_3_12B_it_fp4_mixed.safetensors` (ComfyUI/Comfy-Org's own quantized file, ~8 GB vs ~24 GB for the previous bnb-4bit snapshot). Downloaded automatically into `text_encoders/` on first use if not already present.
- **DramaBox CLIP Loader node** — optional node to load Gemma text encoders from `text_encoders/` (safetensors) and GGUF files when ComfyUI-GGUF is installed. Connect to the TTS node's `dramabox_clip` input for per-workflow encoder selection.
- **Settings preference** — added *ComfyUI Settings → DramaBox → Default Text Encoder filename* to set a global default without needing a CLIP Loader node in every workflow.
- **LTX package compatibility** — DramaBox now relies on installed `ltx-core` / `ltx-pipelines` packages and ships only DramaBox-specific compatibility code in `py/`, reducing conflicts with other LTX-based custom nodes.
- **Old Gemma snapshot cleanup** — the large `gemma-3-12b-it-bnb-4bit/` model directory (previously downloaded into `models/dramabox/`) is automatically removed on startup since it is no longer needed.
- **Removed info output** — the `info` string output has been removed from the DramaBox TTS node.

## Credits

- [DramaBox](https://github.com/resemble-ai/DramaBox) by [ResembleAI](https://www.resemble.ai/)
- Built on the [LTX-Video](https://github.com/Lightricks/LTX-Video) audio diffusion architecture by Lightricks
