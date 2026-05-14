# ComfyUI-DramaBox

ComfyUI custom nodes for [DramaBox](https://github.com/resemble-ai/DramaBox) — ResembleAI's expressive text-to-speech system built on the LTX-2.3 audio diffusion transformer.

## Nodes

| Node | Description |
|------|-------------|
| **DramaBox TTS** | Generates speech audio from a text prompt. Optionally accepts a voice reference clip and advanced options. All model weights are downloaded automatically on first use. |
| **DramaBox Options** | Advanced generation settings (steps, CFG scale, duration, etc.). Connect to the DramaBox TTS node's `options` input. |

<div align="center">
  <img src="docs/images/example.png" alt="DramaBox">
</div>

## Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```
   cd ComfyUI/custom_nodes/
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/FranckyB/ComfyUI-DramaBox.git
   ```
3. Activate your ComfyUI virtual environment:
   Windows (cmd):
   ```bat
   ..\ComfyUI\venv\Scripts\activate
   ```
   Linux/macOS (bash/zsh):
   ```bash
   source ../ComfyUI/venv/bin/activate
   ```
4. Enter the repository:
   ```bash
   cd ComfyUI-DramaBox
   ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

6. On first run, the node will automatically download all model weights into `custom_nodes/ComfyUI-DramaBox/models/`.

### bitsandbytes on newer CUDA versions (13.x)

The released `bitsandbytes` packages only ship pre-compiled CUDA binaries up to a certain version. If your CUDA toolkit is newer (e.g. 13.2) you will see an error like:

```
bitsandbytes library load error: Configured CUDA binary not found at …libbitsandbytes_cuda132.dll
```

### Fix — build bitsandbytes from source

You need NVCC installed (part of the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)) and matching the CUDA version used by your PyTorch build. Then run these commands inside your ComfyUI venv:

```powershell
# Activate your ComfyUI virtual environment (See Above):
# 1. Install the build prerequisites 
pip install scikit-build-core cmake ninja

# 2. Build bitsandbytes from source with the CUDA backend
$env:CMAKE_ARGS = "-DCOMPUTE_BACKEND=cuda"
pip install "git+https://github.com/bitsandbytes-foundation/bitsandbytes.git" --no-build-isolation --force-reinstall
```

The build takes a few minutes. When it completes you should see a `libbitsandbytes_cudaXXX.dll` in your site-packages. Verify it works:

```powershell
python -c "import bitsandbytes as bnb; bnb.functional.get_4bit_type('nf4'); print('OK', bnb.__version__)"
```

**Note:** the `--force-reinstall` flag will also reinstall PyTorch to satisfy bitsandbytes' dependency resolver. If it downgrades torch, restore your original version afterwards:

```powershell
pip install "torch==<your-version>+cu<xyz>" --index-url https://download.pytorch.org/whl/nightly/cu<xyz> --no-deps
```
Replace `<your-version>` and `<xyz>` with your actual torch version and CUDA tag (e.g. `2.13.0.dev20260507+cu132` / `cu132`).

## Credits

- [DramaBox](https://github.com/resemble-ai/DramaBox) by [ResembleAI](https://www.resemble.ai/)
- Built on the [LTX-Video](https://github.com/Lightricks/LTX-Video) audio diffusion architecture by Lightricks
