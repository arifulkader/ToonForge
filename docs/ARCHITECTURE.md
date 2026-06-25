# Architecture

## Overview

ToonForge uses a **dual-engine hybrid architecture** that separates full-scene rendering from face-specific stylization. This design solves the core problem with single-model approaches: uniform stylization that degrades facial identity and detail.

## Engine Separation

### SceneNet (scene_engine.py)

Handles everything **except** face regions. Based on AnimeGANv2's `face_paint_512_v2` generator, which despite its name works on arbitrary content (landscapes, bodies, objects). The generator architecture requires input dimensions divisible by 8, so inputs are padded before inference.

**Input → Pad to 8x → Generator → Denormalize → Output**

### FaceForge (face_engine.py)

Handles face regions **only**. Wraps VToonify which uses DualStyleGAN for controllable, identity-preserving face stylization. Each face goes through:

**Crop → Save to disk → Align (dlib landmarks) → StyleGAN inversion → Style mixing → Output**

The disk I/O step exists because VToonify's alignment pipeline expects file paths. This is a known bottleneck and a candidate for optimization.

## Compositing Strategy

The blender module (`blender.py`) uses a two-phase approach:

### Phase 1: Face Preservation

After SceneNet renders the full frame in anime style, the original face pixels are blended back in using rectangular Gaussian-blurred masks. This creates a frame where the background and body are anime-styled, but face regions retain the original appearance.

**Why not skip SceneNet on face regions?** Because SceneNet runs on the full frame in one pass — masking regions would require modifying the generator or running inference twice.

### Phase 2: Face Compositing

Each FaceForge output is composited into its corresponding face region using an elliptical mask with Gaussian feathering. The ellipse parameters were tuned to produce seamless edges across multiple face sizes and angles.

Key parameters:
- Ellipse X/Y radii: 42% and 45% of region dimensions
- Gaussian blur kernel: 30% of min(width, height)
- Composite region expansion: 45% horizontal, 55% vertical beyond detection box

## Face Detection

Currently uses OpenCV Haar cascades (`haarcascade_frontalface_default.xml`). Detected faces are sorted by area (largest first) since VToonify alignment quality degrades on very small faces. Minimum detection size is configurable (default 60px).

**Planned upgrade**: MediaPipe Face Mesh would provide 468 landmarks vs Haar's bounding box, enabling better alignment and potentially eliminating the dlib dependency inside VToonify.

## Performance Considerations

- SceneNet is the fastest component (~50ms on T4)
- FaceForge is the bottleneck (~200-400ms per face including alignment)
- Total pipeline scales linearly with face count
- GPU memory is dominated by VToonify's StyleGAN backbone (~2GB)
