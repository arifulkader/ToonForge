"""
ToonForge Pipeline — Hybrid multi-model cartoon stylization.

Orchestrates SceneNet (full-frame anime) + FaceForge (per-face stylization)
with intelligent fallback and feathered compositing.
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .blender import composite_stylized_face, preserve_face_regions
from .detector import FaceDetector
from .face_engine import FaceForgeEngine
from .scene_engine import SceneEngine
from .utils import cap_resolution

log = logging.getLogger(__name__)


class ToonForgePipeline:
    """
    Hybrid cartoon stylization pipeline.

    Workflow per frame:
        1. Detect all faces
        2. Run SceneNet on full frame (anime body + background)
        3. Preserve original face regions in SceneNet output (prevent double-style)
        4. Run FaceForge on each face crop → composite back with elliptical blending

    Falls back to SceneNet-only when no faces detected.
    """

    def __init__(
        self,
        device: str = "cuda",
        scene_repo: str = "./third_party/animegan2",
        vtoonify_repo: str = "./third_party/vtoonify",
        face_min_size: int = 60,
        max_input_res: int = 640,
    ):
        self.device = device
        self.max_input_res = max_input_res

        self._detector = FaceDetector(min_size=face_min_size)
        self._scene = SceneEngine(device=device, repo_path=scene_repo)
        self._face = FaceForgeEngine(device=device, vtoonify_path=vtoonify_repo)

        self._loaded = False

    def load(self, style: str = "cartoon1-d") -> "ToonForgePipeline":
        """Load both engines and set initial face style."""
        log.info("Loading ToonForge pipeline...")
        t0 = time.time()

        self._scene.load()
        self._face.load()
        self._face.set_style(style)

        self._loaded = True
        log.info("Pipeline ready in %.1fs", time.time() - t0)
        return self

    def set_style(self, style: str) -> str:
        """Switch face stylization style on the fly."""
        return self._face.set_style(style)

    @property
    def available_styles(self):
        return self._face.available_styles

    def process(
        self,
        image: Image.Image,
        style_degree: float = 0.5,
        padding: Tuple[int, ...] = (200, 200, 200, 200),
    ) -> Tuple[Image.Image, str]:
        """
        Process a single frame through the hybrid pipeline.

        Args:
            image: Input PIL image (any resolution).
            style_degree: FaceForge intensity (0.0-1.0).
            padding: Face crop padding for alignment.

        Returns:
            (result_image, mode_string)
            mode_string examples: 'hybrid(2/3 faces)', 'anime_only(0 faces)'
        """
        if not self._loaded:
            raise RuntimeError("Call .load() before .process()")

        # Cap resolution for performance
        image = cap_resolution(image, self.max_input_res)

        # Step 1: Detect faces
        faces = self._detector.detect(image)

        # Step 2: Full-frame anime pass
        anime_full = self._scene.render(image)

        if not faces:
            return anime_full, "anime_only(0 faces)"

        # Step 3: Preserve original faces in anime output
        orig_np = np.array(image)
        anime_np = np.array(anime_full.resize(image.size, Image.LANCZOS))
        canvas = preserve_face_regions(orig_np, anime_np, faces)

        # Step 4: Stylize each face and composite
        faces_done = 0
        for i, (x, y, w, h) in enumerate(faces):
            # Generous crop around face for VToonify alignment
            pad_x, pad_y = int(w * 0.8), int(h * 0.8)
            cx1 = max(0, x - pad_x)
            cy1 = max(0, y - pad_y)
            cx2 = min(orig_np.shape[1], x + w + pad_x)
            cy2 = min(orig_np.shape[0], y + h + pad_y)

            face_crop = image.crop((cx1, cy1, cx2, cy2))
            stylized = self._face.stylize_face(face_crop, style_degree, padding)

            if stylized is None:
                continue

            canvas = composite_stylized_face(canvas, stylized, (x, y, w, h))
            faces_done += 1

        if faces_done > 0:
            mode = f"hybrid({faces_done}/{len(faces)} faces)"
        else:
            mode = f"anime_only({len(faces)} detected, 0 toonified)"

        return Image.fromarray(canvas), mode
