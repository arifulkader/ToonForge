"""
Multi-face detection using OpenCV Haar cascades.

Designed as a swappable module — replace internals with MediaPipe
or dlib without changing the public interface.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

FaceBox = Tuple[int, int, int, int]  # (x, y, w, h)


class FaceDetector:
    """Detect all faces in an image, sorted largest-first."""

    def __init__(self, min_size: int = 60, scale_factor: float = 1.1, min_neighbors: int = 5):
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._min_size = (min_size, min_size)
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors

        if self._cascade.empty():
            raise RuntimeError("Failed to load Haar cascade — check OpenCV installation")

    def detect(self, image: Image.Image) -> List[FaceBox]:
        """
        Detect all faces in a PIL image.

        Returns list of (x, y, w, h) tuples sorted by area (largest first).
        Empty list if no faces found.
        """
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
        )

        if len(faces) == 0:
            return []

        # Sort by area descending — process largest face first
        sorted_faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        log.debug("Detected %d face(s)", len(sorted_faces))
        return [tuple(f) for f in sorted_faces]
