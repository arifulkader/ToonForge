"""
Utility functions for image I/O, resolution management, and display.
"""

import io
import logging
from base64 import b64encode
from typing import Tuple

from PIL import Image

log = logging.getLogger(__name__)


def cap_resolution(image: Image.Image, max_dim: int = 640) -> Image.Image:
    """Downscale image so longest side <= max_dim. No-op if already within limit."""
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    log.debug("Capping resolution: %dx%d → %dx%d", w, h, *new_size)
    return image.resize(new_size, Image.LANCZOS)


def to_html_img(image: Image.Image, width: int = 384, quality: int = 88) -> str:
    """Convert PIL image to inline HTML <img> tag with base64 JPEG data."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    b64 = b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/jpeg;base64,{b64}" width="{width}"/>'


def side_by_side_html(
    images: list,
    labels: list,
    width: int = 400,
    quality: int = 88,
) -> str:
    """Generate HTML for side-by-side image comparison."""
    cards = []
    for img, label in zip(images, labels):
        html_img = to_html_img(img, width, quality)
        cards.append(
            f'<div style="text-align:center">'
            f'<div style="font-size:12px;margin-bottom:4px">{label}</div>'
            f'{html_img}</div>'
        )
    return f'<div style="display:flex;gap:12px;font-family:monospace;">{"".join(cards)}</div>'


def get_gpu_stats() -> Tuple[float, float]:
    """Return (current_gb, peak_gb) GPU memory usage. Returns (0, 0) if no CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0, 0.0
        current = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        return current, peak
    except ImportError:
        return 0.0, 0.0
