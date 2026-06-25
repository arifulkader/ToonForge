"""
FaceForge — Per-face premium stylization engine.

Wraps VToonify for high-quality face rendering with identity preservation.
Each face is independently cropped, aligned, stylized, and returned.
"""

import logging
import os
import tempfile
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class FaceForgeEngine:
    """High-fidelity face stylizer using VToonify backbone."""

    STYLE_FAMILIES = {
        "cartoon": ["cartoon1-d", "cartoon2-d", "cartoon3-d", "cartoon4-d", "cartoon5-d"],
        "arcane": ["arcane1-d", "arcane2-d"],
        "pixar": ["pixar1-d"],
        "comic": ["comic1-d", "comic2-d"],
        "caricature": ["caricature1-d", "caricature2-d"],
    }

    def __init__(self, device: str = "cuda", vtoonify_path: str = "./third_party/vtoonify"):
        self.device = device
        self._vtoonify_path = vtoonify_path
        self._model = None
        self._current_style: Optional[str] = None
        self._exstyle = None

    def load(self) -> "FaceForgeEngine":
        """Initialize the VToonify model (downloads weights from HuggingFace)."""
        os.chdir(self._vtoonify_path)

        from vtoonify_model import Model
        self._model = Model(self.device)
        log.info("FaceForge model initialized on %s", self.device)
        return self

    def set_style(self, style_name: str) -> str:
        """
        Load a specific style's weights.

        Args:
            style_name: e.g. 'cartoon1-d', 'arcane1-d', 'pixar1-d'

        Returns:
            Status message from model loader.
        """
        if self._model is None:
            raise RuntimeError("Call .load() before .set_style()")

        self._exstyle, info = self._model.load_model(style_name)
        self._current_style = style_name
        log.info("FaceForge style loaded: %s", style_name)
        return info

    @property
    def available_styles(self) -> List[str]:
        """All available style names."""
        if self._model is None:
            return []
        return list(self._model.style_types.keys())

    def stylize_face(
        self,
        face_crop: Image.Image,
        style_degree: float = 0.5,
        padding: Tuple[int, ...] = (200, 200, 200, 200),
    ) -> Optional[Image.Image]:
        """
        Stylize a single cropped face region.

        Args:
            face_crop: PIL image of the face region (generous crop around face).
            style_degree: 0.0 (subtle) to 1.0 (strong). Only for '-d' styles.
            padding: (top, bottom, left, right) padding for face alignment.

        Returns:
            Stylized PIL image, or None if alignment/stylization fails.
        """
        if self._model is None or self._exstyle is None:
            raise RuntimeError("Call .load() and .set_style() first")

        # VToonify expects a file path for alignment
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            face_crop.save(tmp.name, quality=95)
            tmp_path = tmp.name

        try:
            aligned, instyle, msg = self._model.detect_and_align_image(tmp_path, *padding)
            if aligned is None:
                log.debug("Face alignment failed: %s", msg)
                return None

            result, msg2 = self._model.image_toonify(
                aligned, instyle, self._exstyle, style_degree, self._current_style
            )
            if result is None:
                log.debug("Face stylization failed: %s", msg2)
                return None

            return Image.fromarray(result)

        finally:
            os.unlink(tmp_path)
