"""CLIP embedding pipeline for spatial memory observations.

Runs on the observer (sim adapter on RTX, or Jetson in production).
Embeds camera frames at a configurable rate and sends pre-computed
512-dim embeddings to the gateway via WS/Intercom — NOT raw images.

The gateway stores embeddings directly in ChromaDB without running CLIP.
This keeps bandwidth low (~2KB per observation vs ~40KB per image).
"""

from __future__ import annotations

import io
import time
from typing import Any

_model = None
_preprocess = None
_tokenizer = None
_device: str = "cpu"


def init(device: str = "cuda", model_name: str = "ViT-B-32") -> bool:
    """Load CLIP model. Returns True on success."""
    global _model, _preprocess, _tokenizer, _device
    try:
        import open_clip
        import torch

        _device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai", device=_device
        )
        _tokenizer = open_clip.get_tokenizer(model_name)
        _model.eval()
        print(f"[clip] CLIP loaded: {model_name} on {_device}", flush=True)
        return True
    except ImportError:
        print("[clip] ERROR: open_clip not installed. pip install open-clip-torch", flush=True)
        return False
    except Exception as exc:
        print(f"[clip] ERROR loading CLIP: {exc}", flush=True)
        return False


def embed_image(frame_jpeg: bytes) -> list[float] | None:
    """Embed a JPEG frame into a 512-dim CLIP vector.

    Returns list of 512 floats, or None on failure.
    """
    if _model is None or _preprocess is None:
        return None

    try:
        import torch
        from PIL import Image

        img = Image.open(io.BytesIO(frame_jpeg)).convert("RGB")
        img_tensor = _preprocess(img).unsqueeze(0).to(_device)

        with torch.no_grad():
            features = _model.encode_image(img_tensor)
            features /= features.norm(dim=-1, keepdim=True)

        return features[0].cpu().tolist()
    except Exception as exc:
        print(f"[clip] embed_image error: {exc}", flush=True)
        return None


def embed_text(text: str) -> list[float] | None:
    """Embed text into a 512-dim CLIP vector.

    Returns list of 512 floats, or None on failure.
    """
    if _model is None or _tokenizer is None:
        return None

    try:
        import torch

        tokens = _tokenizer([text]).to(_device)
        with torch.no_grad():
            features = _model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)

        return features[0].cpu().tolist()
    except Exception as exc:
        print(f"[clip] embed_text error: {exc}", flush=True)
        return None


def check_available() -> dict:
    """Check if CLIP is available and loaded."""
    try:
        import open_clip
        return {
            "available": True,
            "loaded": _model is not None,
            "device": _device,
            "package": "open_clip",
        }
    except ImportError:
        return {"available": False, "error": "open-clip-torch not installed"}
