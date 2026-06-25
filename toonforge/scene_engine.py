"""
SceneNet — Full-scene anime stylization engine.

Wraps AnimeGANv2 (face_paint_512_v2) for body + background rendering.
Handles model loading, resolution padding, and inference.
"""

import logging
import os
import importlib.util
import sys
from typing import Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image

log = logging.getLogger(__name__)

_WEIGHT_URL = (
    "https://github.com/bryandlee/animegan2-pytorch/raw/main/weights/face_paint_512_v2.pt"
)


class SceneEngine:
    """Full-frame anime-style renderer using AnimeGANv2 backbone."""

    def __init__(self, device: str = "cuda", repo_path: str = "./third_party/animegan2"):
        self.device = device
        self._model: Optional[torch.nn.Module] = None
        self._repo_path = repo_path

    def load(self) -> "SceneEngine":
        """Download weights (if needed) and load the generator."""
        model_file = os.path.join(self._repo_path, "model.py")
        weight_file = os.path.join(self._repo_path, "weights", "face_paint_512_v2.pt")

        # Dynamic import to avoid name collision with other 'model' modules
        spec = importlib.util.spec_from_file_location("_scenenet_gen", model_file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_scenenet_gen"] = mod
        spec.loader.exec_module(mod)

        if not os.path.exists(weight_file):
            os.makedirs(os.path.dirname(weight_file), exist_ok=True)
            log.info("Downloading SceneNet weights...")
            torch.hub.download_url_to_file(_WEIGHT_URL, weight_file)

        self._model = mod.Generator().eval().to(self.device)
        self._model.load_state_dict(torch.load(weight_file, map_location=self.device))
        log.info("SceneNet loaded on %s", self.device)
        return self

    @torch.inference_mode()
    def render(self, image: Image.Image) -> Image.Image:
        """
        Render a PIL image through the anime stylization network.

        Input is padded to nearest multiple of 8 (required by generator arch).
        Output is returned at padded resolution — caller handles resizing.
        """
        if self._model is None:
            raise RuntimeError("Call .load() before .render()")

        w, h = image.size
        # Generator requires dims divisible by 8
        nw = max(8, (w // 8) * 8)
        nh = max(8, (h // 8) * 8)
        resized = image.resize((nw, nh), Image.LANCZOS)

        tensor = TF.to_tensor(resized).unsqueeze(0).to(self.device) * 2 - 1
        out = self._model(tensor)
        out = ((out + 1) / 2).clamp(0, 1)

        return TF.to_pil_image(out.squeeze(0).cpu())
