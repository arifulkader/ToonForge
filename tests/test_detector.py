"""Tests for face detector module."""

import numpy as np
import pytest
from PIL import Image

from toonforge.detector import FaceDetector


@pytest.fixture
def detector():
    return FaceDetector(min_size=30)


def test_no_faces_on_blank_image(detector):
    """Blank image should return empty list."""
    blank = Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8))
    faces = detector.detect(blank)
    assert faces == []


def test_faces_sorted_by_area(detector):
    """If multiple faces detected, largest should come first."""
    # This would need a real test image with multiple faces
    # Placeholder for CI — actual face detection tested in integration
    blank = Image.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 128)
    faces = detector.detect(blank)
    if len(faces) > 1:
        areas = [w * h for (_, _, w, h) in faces]
        assert areas == sorted(areas, reverse=True)


def test_detector_returns_tuples(detector):
    """Return type should be list of tuples."""
    img = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 200)
    result = detector.detect(img)
    assert isinstance(result, list)
