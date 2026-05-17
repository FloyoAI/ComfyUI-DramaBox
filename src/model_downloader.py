#!/usr/bin/env python3
"""
Download DramaBox models from HuggingFace into a clean local layout.

After the first download, files are loaded directly from disk — no HuggingFace
API calls are made when the weights are already present.

Default storage (ComfyUI models folder)
----------------------------------------
<comfyui>/models/
    dramabox/
        dramabox-dit-v1.safetensors
        dramabox-audio-components.safetensors
        silence_latent_frame.pt
        gemma-3-12b-it-bnb-4bit/
            <full snapshot contents>

Fallback (standalone / no ComfyUI)
------------------------------------
<node_root>/models/   (same sub-layout as above)
"""
import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

DRAMABOX_REPO = "ResembleAI/Dramabox"
GEMMA_REPO = "unsloth/gemma-3-12b-it-bnb-4bit"

_SRC_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
# Old node-local models directory — used as migration source
_NODE_MODELS_DIR = _SRC_DIR.parent / "models"

# Prefer ComfyUI's central models folder; fall back to the node-local directory
try:
    import folder_paths as _fp
    DEFAULT_MODELS_DIR = str(Path(_fp.models_dir))
except Exception:
    DEFAULT_MODELS_DIR = str(_NODE_MODELS_DIR)

# HF repo path → local filename (always stored flat inside dramabox/)
MODEL_FILES = {
    "transformer":      "dramabox-dit-v1.safetensors",
    "audio_components": "dramabox-audio-components.safetensors",
    "silence_latent":   "assets/silence_latent_frame.pt",   # repo sub-path; stored flat locally
}

# Track which model directories have already been migrated this session
_MIGRATED = set()


def migrate_old_layout(models_dir):
    """Silently move files from the old HuggingFace blob-cache layout to the
    new clean layout.  Safe to call repeatedly — already-migrated directories
    are skipped via an in-process cache.

    Old layout (created by hf_hub_download / snapshot_download with cache_dir=):
        models/models--ResembleAI--Dramabox/snapshots/<hash>/dramabox-*.safetensors
        models/models--unsloth--gemma-3-12b-it-bnb-4bit/snapshots/<hash>/<files>

    New layout:
        models/dramabox/dramabox-*.safetensors
        models/dramabox/gemma-3-12b-it-bnb-4bit/<files>

    Also handles:
      - safetensors placed flat in the models root (models/dramabox-dit-v1.safetensors)
      - Gemma at old top-level location (models/gemma-3-12b-it-bnb-4bit/) before it
        was moved inside dramabox/
      - Files in the node-local models/ dir when ComfyUI models dir is in use
    """
    models_dir = Path(models_dir)
    key = str(models_dir)
    if key in _MIGRATED:
        return []
    _MIGRATED.add(key)

    migrated = []

    # ── DramaBox ──────────────────────────────────────────────────────────────
    dramabox_dir = models_dir / "dramabox"
    dramabox_names = [Path(v).name for v in MODEL_FILES.values()]

    # Pattern 1: HF blob cache  (models--ResembleAI--Dramabox/)
    old_dramabox_cache = models_dir / "models--ResembleAI--Dramabox"
    if old_dramabox_cache.is_dir():
        snapshots_root = old_dramabox_cache / "snapshots"
        snapshot_dirs = sorted(snapshots_root.iterdir()) if snapshots_root.is_dir() else []
        if snapshot_dirs:
            snapshot = snapshot_dirs[-1]  # use the most recent hash
            dramabox_dir.mkdir(parents=True, exist_ok=True)
            for fname in dramabox_names:
                src = snapshot / fname
                dst = dramabox_dir / fname
                if src.exists() and not dst.exists():
                    logger.info(f"[DramaBox] Migrating {fname} …")
                    shutil.copy2(str(src), str(dst))  # follows symlinks → copies real content
                    migrated.append(fname)
            # Clean up old cache once all expected .safetensors are in place
            safetensors = ["dramabox-dit-v1.safetensors", "dramabox-audio-components.safetensors"]
            if all((dramabox_dir / f).exists() for f in safetensors):
                logger.info(f"[DramaBox] Removing old cache: {old_dramabox_cache}")
                shutil.rmtree(str(old_dramabox_cache), ignore_errors=True)

    # Pattern 2: files placed flat in models root (dramabox-dit-v1.safetensors, etc.)
    for fname in dramabox_names:
        src = models_dir / fname
        dst = dramabox_dir / fname
        if src.is_file() and not dst.exists():
            dramabox_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[DramaBox] Migrating {fname} from models root …")
            shutil.move(str(src), str(dst))
            migrated.append(fname)

    # ── Gemma ─────────────────────────────────────────────────────────────────
    gemma_name = GEMMA_REPO.split("/")[-1]   # "gemma-3-12b-it-bnb-4bit"
    new_gemma_dir = models_dir / "dramabox" / gemma_name
    old_gemma_cache = models_dir / f"models--{GEMMA_REPO.replace('/', '--')}"

    already_present = new_gemma_dir.is_dir() and any(new_gemma_dir.iterdir())
    if old_gemma_cache.is_dir() and not already_present:
        snapshots_root = old_gemma_cache / "snapshots"
        snapshot_dirs = sorted(snapshots_root.iterdir()) if snapshots_root.is_dir() else []
        if snapshot_dirs:
            snapshot = snapshot_dirs[-1]
            new_gemma_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[DramaBox] Migrating Gemma snapshot → {new_gemma_dir} (this may take a moment) …")
            for item in snapshot.iterdir():
                dst = new_gemma_dir / item.name
                if dst.exists():
                    continue
                if item.is_symlink() or item.is_file():
                    shutil.copy2(str(item), str(dst))  # follows symlinks
                elif item.is_dir():
                    shutil.copytree(str(item), str(dst))
            migrated.append(gemma_name)
            logger.info(f"[DramaBox] Removing old Gemma cache: {old_gemma_cache}")
            shutil.rmtree(str(old_gemma_cache), ignore_errors=True)

    # ── Migrate from old node-local models dir (when using ComfyUI models dir) ──
    if _NODE_MODELS_DIR.is_dir() and _NODE_MODELS_DIR.resolve() != models_dir.resolve():
        node_dramabox = _NODE_MODELS_DIR / "dramabox"
        if node_dramabox.is_dir():
            dramabox_dir.mkdir(parents=True, exist_ok=True)
            for fname in dramabox_names:
                src = node_dramabox / fname
                dst = dramabox_dir / fname
                if src.exists() and not dst.exists():
                    logger.info(f"[DramaBox] Moving {fname} from node models dir…")
                    shutil.move(str(src), str(dst))
                    migrated.append(fname)
            # Remove empty node dramabox dir
            try:
                if not any(node_dramabox.iterdir()):
                    node_dramabox.rmdir()
            except Exception:
                pass

        node_gemma = _NODE_MODELS_DIR / gemma_name
        new_gemma_dir = models_dir / "dramabox" / gemma_name
        if node_gemma.is_dir() and not (new_gemma_dir.is_dir() and any(new_gemma_dir.iterdir())):
            logger.info(f"[DramaBox] Moving Gemma from node models dir → {new_gemma_dir}…")
            new_gemma_dir.mkdir(parents=True, exist_ok=True)
            for item in node_gemma.iterdir():
                dst = new_gemma_dir / item.name
                if dst.exists():
                    continue
                shutil.move(str(item), str(dst))
            migrated.append(gemma_name)
            # Remove empty node gemma dir
            try:
                if not any(node_gemma.iterdir()):
                    node_gemma.rmdir()
            except Exception:
                pass

    # ── Remove old Gemma model directories (replaced by fp8 safetensors) ────────
    # The full bnb-4bit snapshot is no longer needed — DramaBox now loads Gemma
    # from a standard fp8 safetensors in the text_encoders folder.
    old_gemma_dirs = [
        models_dir / "dramabox" / gemma_name,          # migrated location
        old_gemma_cache,                                # HF blob cache
        _NODE_MODELS_DIR / "dramabox" / gemma_name,    # node-local dramabox/
        _NODE_MODELS_DIR / gemma_name,                  # old node-local top-level
    ]
    for gemma_dir in old_gemma_dirs:
        if gemma_dir.is_dir():
            logger.info(f"[DramaBox] Removing old Gemma directory (no longer needed): {gemma_dir}")
            shutil.rmtree(str(gemma_dir), ignore_errors=True)

    if migrated:
        logger.info(f"[DramaBox] Migration complete. Items moved: {migrated}")
    return migrated


def get_model_path(name, cache_dir=None):
    """Return local path for a DramaBox model file, downloading only if absent.

    Files are stored flat inside <cache_dir>/dramabox/ — no HuggingFace blob
    cache structure is created.  Any old HF cache layout is migrated silently
    on first call.

    Args:
        name: One of 'transformer', 'audio_components', 'silence_latent'
        cache_dir: Root models directory (default: <node_root>/models/)

    Returns:
        Absolute local file path as a string
    """
    models_dir = Path(cache_dir or DEFAULT_MODELS_DIR)
    migrate_old_layout(models_dir)
    dramabox_dir = models_dir / "dramabox"

    if name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(MODEL_FILES.keys())}")

    repo_filename = MODEL_FILES[name]
    # Always store flat — strip any sub-directory from the repo path
    local_path = dramabox_dir / Path(repo_filename).name

    if local_path.is_file():
        logger.info(f"[DramaBox] Found {name} locally: {local_path}")
        return str(local_path)

    dramabox_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DramaBox] Downloading {name} ({Path(repo_filename).name}) from HuggingFace...")
    _httpx_log = logging.getLogger("httpx")
    _prev = _httpx_log.level
    _httpx_log.setLevel(logging.WARNING)
    try:
        snapshot_download(
            repo_id=DRAMABOX_REPO,
            allow_patterns=[repo_filename],
            local_dir=str(dramabox_dir),
            token=os.environ.get("HF_TOKEN"),
        )
    finally:
        _httpx_log.setLevel(_prev)
    # huggingface_hub creates a .cache folder in local_dir for tracking — remove it
    cache_dir_path = dramabox_dir / ".cache"
    if cache_dir_path.is_dir():
        shutil.rmtree(str(cache_dir_path), ignore_errors=True)
    print(f"[DramaBox] Done: {local_path}")
    return str(local_path)


def get_gemma_path(cache_dir=None):
    """Return local Gemma snapshot directory, downloading only if absent.

    The snapshot is stored in <cache_dir>/gemma-3-12b-it-bnb-4bit/ — no
    HuggingFace blob cache structure is created.  Any old HF cache layout is
    migrated silently on first call.

    Using the pre-quantized bnb-4bit variant skips runtime bitsandbytes
    quantization and ~halves the Gemma load time.

    Args:
        cache_dir: Root models directory (default: <node_root>/models/)

    Returns:
        Absolute local directory path as a string
    """
    models_dir = Path(cache_dir or DEFAULT_MODELS_DIR)
    migrate_old_layout(models_dir)
    gemma_name = GEMMA_REPO.split("/")[-1]   # "gemma-3-12b-it-bnb-4bit"
    local_dir = models_dir / "dramabox" / gemma_name

    if local_dir.is_dir() and any(local_dir.iterdir()):
        logger.info(f"[DramaBox] Found Gemma locally: {local_dir}")
        return str(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DramaBox] Downloading Gemma ({GEMMA_REPO}) from HuggingFace...")

    print(f"[DramaBox] Destination: {local_dir}")
    _httpx_log = logging.getLogger("httpx")
    _prev = _httpx_log.level
    _httpx_log.setLevel(logging.WARNING)
    try:
        snapshot_download(
            repo_id=GEMMA_REPO,
            local_dir=str(local_dir),
            token=os.environ.get("HF_TOKEN"),
        )
    finally:
        _httpx_log.setLevel(_prev)
    # huggingface_hub creates a .cache folder in local_dir for tracking — remove it
    cache_dir_path = local_dir / ".cache"
    if cache_dir_path.is_dir():
        shutil.rmtree(str(cache_dir_path), ignore_errors=True)
    print(f"[DramaBox] Done: {local_dir}")
    return str(local_dir)


def get_all_paths(cache_dir=None):
    """Download all required models and return a paths dict.

    Returns:
        {
            'transformer':      '/path/to/dramabox/dramabox-dit-v1.safetensors',
            'audio_components': '/path/to/dramabox/dramabox-audio-components.safetensors',
            'silence_latent':   '/path/to/dramabox/silence_latent_frame.pt',
        }

    Note: Gemma is no longer downloaded here. The ComfyUI node loads Gemma from
    an fp8 safetensors placed in the text_encoders folder by the user.
    """
    paths = {}
    for name in MODEL_FILES:
        paths[name] = get_model_path(name, cache_dir)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = get_all_paths()
    print("\nAll models downloaded:")
    for k, v in paths.items():
        size = os.path.getsize(v) / 1e9 if os.path.isfile(v) else "dir"
        print(f"  {k}: {v} ({size:.2f} GB)" if isinstance(size, float) else f"  {k}: {v} (directory)")
