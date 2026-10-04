"""Finding and loading a Cue model: a short name, a Hugging Face repo, or a local folder."""
from __future__ import annotations
import json
import os
from pathlib import Path

MODELS = {
    "tiny": "IOTEverythin/cue-tiny",
    "v5": "IOTEverythin/cue-v5",
    "v4": "IOTEverythin/cue-v4",
    "v3": "IOTEverythin/cue-v3",
}
_FILES = ["config.json", "cue_tiny.onnx", "mel_filters.npy", "cue.safetensors", "encoder/*"]


def fetch(name: str = "tiny", revision: str | None = None) -> Path:
    """A local folder holding the model: `name` is "tiny", "v4", "v3", a Hugging Face
    repo id, or a folder. Downloads (once, to the Hugging Face cache) when needed;
    only the files needed to run it, not the training checkpoints."""
    p = Path(name).expanduser()
    if p.is_dir():
        return p
    repo = MODELS.get(name, name)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, revision=revision, allow_patterns=_FILES))


def load(name: str | None = None, device: str | None = None, revision: str | None = None, threads: int = 1):
    """Load a Cue model, once per process: share it across calls, one stream() per call.

        cue = cue_turn.load()            # Cue Tiny (CPU, numpy + onnxruntime)
        cue = cue_turn.load("v4")        # Cue v4 (needs: pip install "cue-turn[full]")

    `name` defaults to $CUE_MODEL, else "tiny". `device` and `threads` apply to v4/v3 and
    Tiny respectively."""
    folder = fetch(name or os.getenv("CUE_MODEL") or "tiny", revision)
    cfg = json.loads((folder / "config.json").read_text())
    if cfg.get("encoder_type") == "tiny-onnx":
        from .tiny import CueTiny
        return CueTiny(folder, threads=threads)
    try:
        from .model import CueModel
    except ImportError as e:
        raise ImportError(f"Cue {cfg.get('version', '')} needs torch and transformers: "
                          f'pip install "cue-turn[full]" ({e})') from e
    return CueModel(folder, device)
