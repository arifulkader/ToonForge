"""
Blender — Feathered alpha compositing for face/scene fusion.

Handles Gaussian-blurred rectangular masks (for preserving faces in scene pass)
and elliptical feathered masks (for compositing stylized faces back in).
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

FaceBox = Tuple[int, int, int, int]


def preserve_face_regions(
    original: np.ndarray,
    scene_rendered: np.ndarray,
    faces: List[FaceBox],
    expand_ratio: float = 0.3,
) -> np.ndarray:
    """
    Blend original face pixels back into scene-rendered frame.

    Prevents double-stylization when FaceForge runs on top of SceneNet output.
    Uses rectangular masks with Gaussian blur for soft transitions.

    Args:
        original: Original frame as numpy array (H, W, 3).
        scene_rendered: SceneNet output resized to match original.
        faces: List of (x, y, w, h) face bounding boxes.
        expand_ratio: How much to expand each face box (0.3 = 30%).

    Returns:
        Blended frame with original faces preserved in anime background.
    """
    h_img, w_img = original.shape[:2]
    mask = np.zeros((h_img, w_img), dtype=np.float32)

    for (x, y, w, h) in faces:
        ex, ey = int(w * expand_ratio), int(h * expand_ratio)
        x1 = max(0, x - ex)
        y1 = max(0, y - ey)
        x2 = min(w_img, x + w + ex)
        y2 = min(h_img, y + h + ey)
        mask[y1:y2, x1:x2] = 1.0

    # Gaussian blur for soft transition
    if faces:
        ref_w, ref_h = faces[0][2], faces[0][3]
        ksize = max(31, int(min(ref_w, ref_h) * 0.4) | 1)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    mask_3d = mask[:, :, np.newaxis]
    blended = (original * mask_3d + scene_rendered * (1 - mask_3d)).astype(np.uint8)
    return blended


def composite_stylized_face(
    canvas: np.ndarray,
    stylized_face: Image.Image,
    face_box: FaceBox,
    expand_x_ratio: float = 0.45,
    expand_y_ratio: float = 0.55,
    ellipse_x_ratio: float = 0.42,
    ellipse_y_ratio: float = 0.45,
    blur_ratio: float = 0.3,
) -> np.ndarray:
    """
    Composite a single stylized face onto the canvas using elliptical feathered mask.

    Args:
        canvas: Working canvas (H, W, 3) numpy array (modified in place).
        stylized_face: FaceForge output for this face.
        face_box: Original detection box (x, y, w, h).
        expand_*_ratio: How much to expand the composite region.
        ellipse_*_ratio: Ellipse radii as fraction of region size.
        blur_ratio: Gaussian kernel size as fraction of region.

    Returns:
        Modified canvas with face composited in.
    """
    x, y, w, h = face_box
    h_img, w_img = canvas.shape[:2]

    # Compute composite region
    exp_x = int(w * expand_x_ratio)
    exp_y = int(h * expand_y_ratio)
    fx1 = max(0, x - exp_x)
    fy1 = max(0, y - exp_y)
    fx2 = min(w_img, x + w + exp_x)
    fy2 = min(h_img, y + h + exp_y)
    rw, rh = fx2 - fx1, fy2 - fy1

    if rw < 10 or rh < 10:
        log.debug("Face region too small to composite (%dx%d)", rw, rh)
        return canvas

    # Resize stylized face to fit region
    resized = stylized_face.resize((rw, rh), Image.LANCZOS)
    fg = np.array(resized).astype(np.float32)

    # Elliptical feathered mask
    mask = np.zeros((rh, rw), dtype=np.float32)
    cx, cy = rw // 2, rh // 2
    ax = int(rw * ellipse_x_ratio)
    ay = int(rh * ellipse_y_ratio)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)

    ks = max(31, int(min(rw, rh) * blur_ratio) | 1)
    mask = cv2.GaussianBlur(mask, (ks, ks), 0)[:, :, np.newaxis]

    bg = canvas[fy1:fy2, fx1:fx2].astype(np.float32)
    canvas[fy1:fy2, fx1:fx2] = (fg * mask + bg * (1 - mask)).astype(np.uint8)

    return canvas
