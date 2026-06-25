"""Tests for blender compositing module."""

import numpy as np
import pytest
from PIL import Image

from toonforge.blender import preserve_face_regions, composite_stylized_face


def test_preserve_no_faces_returns_scene():
    """With no faces, output should equal scene_rendered."""
    orig = np.ones((100, 100, 3), dtype=np.uint8) * 50
    scene = np.ones((100, 100, 3), dtype=np.uint8) * 200
    result = preserve_face_regions(orig, scene, faces=[])
    np.testing.assert_array_equal(result, scene)


def test_preserve_with_face_blends():
    """Face region should contain some original pixels."""
    orig = np.zeros((200, 200, 3), dtype=np.uint8)
    scene = np.ones((200, 200, 3), dtype=np.uint8) * 255
    faces = [(60, 60, 80, 80)]
    result = preserve_face_regions(orig, scene, faces)

    # Center of face region should have some dark (original) pixels
    center_val = result[100, 100].mean()
    corner_val = result[0, 0].mean()
    assert center_val < corner_val, "Face center should be darker than corner"


def test_composite_tiny_region_skipped():
    """Very small face regions should be skipped gracefully."""
    canvas = np.ones((100, 100, 3), dtype=np.uint8) * 128
    face_img = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
    face_box = (48, 48, 4, 4)  # tiny box → region < 10px
    result = composite_stylized_face(canvas, face_img, face_box)
    # Should not crash, canvas mostly unchanged
    assert result.shape == (100, 100, 3)
