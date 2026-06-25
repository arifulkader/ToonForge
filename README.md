# 🎨 ToonForge — Real-Time Multi-Model Cartoon Stylization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/toonforge-realtime-cartoon-stylization/blob/main/notebooks/ToonForge_Demo.ipynb)

> **A hybrid multi-model pipeline that blends full-scene anime stylization with high-fidelity face toonification in real-time — from webcam, video, or image input.**

<p align="center">
  <img src="assets/banner_demo.gif" alt="ToonForge real-time cartoon stylization demo" width="720"/>
</p>

---

## ✨ What Makes ToonForge Different

Most cartoonization tools apply **one** model uniformly across the entire frame. This creates a common problem: bodies and backgrounds look great, but faces lose detail and identity.

ToonForge solves this with a **dual-engine architecture**:

| Engine | Handles | Quality |
|--------|---------|---------|
| **SceneNet** (AnimeGANv2-based) | Full frame — body, clothes, background | Fast anime-style rendering |
| **FaceForge** (VToonify-based) | Detected face regions only | Premium stylization with identity preservation |

The pipeline **detects all faces**, renders the scene through SceneNet, then composites FaceForge output into each face region using elliptical feathered blending — producing seamless, artifact-free results even with multiple people in frame.

**No face detected?** Falls back to full-scene anime automatically. Zero crashes, zero blank outputs.

---

## 🖼️ Sample Results

<table>
<tr>
<td align="center"><b>Original</b></td>
<td align="center"><b>SceneNet Only</b></td>
<td align="center"><b>Hybrid (cartoon1)</b></td>
<td align="center"><b>Hybrid (arcane)</b></td>
<td align="center"><b>Hybrid (pixar)</b></td>
</tr>
<tr>
<td><img src="assets/samples/original.jpg" width="150"/></td>
<td><img src="assets/samples/scenenet_only.jpg" width="150"/></td>
<td><img src="assets/samples/hybrid_cartoon1.jpg" width="150"/></td>
<td><img src="assets/samples/hybrid_arcane.jpg" width="150"/></td>
<td><img src="assets/samples/hybrid_pixar.jpg" width="150"/></td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (T4 or better recommended)
- ~4GB GPU VRAM minimum

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/toonforge-realtime-cartoon-stylization.git
cd toonforge-realtime-cartoon-stylization
pip install -r requirements.txt
```

### Run on a Single Image

```bash
python -m toonforge.run --input photo.jpg --style cartoon1-d --degree 0.5
```

### Run Live Webcam

```bash
python -m toonforge.run --webcam --style arcane1-d --degree 0.6 --preset balanced
```

### Run in Google Colab (Recommended for Quick Start)

Open `notebooks/ToonForge_Demo.ipynb` in Colab with a T4 GPU runtime. Everything installs and runs in ~3 minutes.

---

## 🎭 Available Styles

ToonForge ships with **14 face styles** via FaceForge, each with adjustable intensity:

| Style Family | Variants | Adjustable (`-d`) |
|-------------|----------|-------------------|
| Cartoon | `cartoon1` through `cartoon5` | ✅ |
| Arcane | `arcane1`, `arcane2` | ✅ |
| Pixar | `pixar1` | ✅ |
| Comic | `comic1`, `comic2` | ✅ |
| Caricature | `caricature1`, `caricature2` | ✅ |

**Style degree** (`--degree 0.0-1.0`): Controls how strongly the face style is applied. `0.3` = subtle enhancement, `0.7` = strong stylization.

Drop the `-d` suffix for fixed-intensity variants (e.g., `cartoon1` instead of `cartoon1-d`).

---

## ⚙️ Architecture

```
Input Frame
    │
    ├──► Face Detector (Haar Cascade, multi-face)
    │         │
    │         ├── faces found ──► Crop each face region
    │         │                       │
    │         │                       ▼
    │         │                  FaceForge Engine
    │         │                  (VToonify backbone)
    │         │                       │
    │         │                       ▼
    │         │               Elliptical feathered
    │         │                  alpha blending
    │         │                       │
    │         └── no faces ────────┐  │
    │                              │  │
    ├──► SceneNet Engine ──────────┤  │
    │    (AnimeGANv2 backbone)     │  │
    │                              ▼  ▼
    │                         Composite
    │                              │
    └──────────────────────────────┘
                                   │
                              Output Frame
```

### Key Design Decisions

1. **Face regions preserved in SceneNet pass** — Before compositing, original face pixels are blended back into the anime-rendered frame using Gaussian-blurred masks, preventing double-stylization artifacts.

2. **Per-face VToonify** — Each detected face gets its own crop → align → stylize → composite pass. Handles group photos with different face sizes.

3. **Adaptive blending** — Elliptical masks with Gaussian feathering (`kernel = 0.3 × min(w,h)`) ensure seamless edges between stylized faces and anime backgrounds.

---

## 📁 Project Structure

```
toonforge/
├── toonforge/
│   ├── __init__.py
│   ├── run.py              # CLI entry point
│   ├── pipeline.py         # Hybrid pipeline orchestrator
│   ├── scene_engine.py     # SceneNet (full-frame anime)
│   ├── face_engine.py      # FaceForge (per-face stylization)
│   ├── detector.py         # Multi-face detection
│   ├── blender.py          # Feathered alpha compositing
│   └── utils.py            # Image I/O, resolution helpers
├── configs/
│   └── default.yaml        # Pipeline configuration
├── notebooks/
│   └── ToonForge_Demo.ipynb
├── assets/
│   └── samples/
├── tests/
│   ├── test_detector.py
│   └── test_pipeline.py
├── docs/
│   └── ARCHITECTURE.md
├── requirements.txt
├── setup.py
├── LICENSE
├── CONTRIBUTING.md
├── CHANGELOG.md
└── .github/
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.md
    │   └── feature_request.md
    └── workflows/
        └── ci.yml
```

---

## ⚡ Performance

Benchmarked on Google Colab T4 GPU (640×480 input):

| Mode | FPS | Description |
|------|-----|-------------|
| `fast` | 2-3 | Anime @ 320px, lower JPEG quality |
| `balanced` | 1-2 | Anime @ 480px, good quality |
| `quality` | 0.5-1 | Anime @ 640px, maximum fidelity |

GPU memory usage: ~2.5-3.5 GB depending on style and face count.

---

## 🔧 Configuration

Edit `configs/default.yaml` or pass CLI flags:

```yaml
style: cartoon1-d
style_degree: 0.5
preset: balanced        # fast | balanced | quality
padding: [200, 200, 200, 200]  # face crop padding
max_input_res: 640      # cap input resolution
face_min_size: 60       # minimum face detection size (px)
```

---

## 🗺️ Roadmap

- [x] Multi-face hybrid pipeline
- [x] Webcam live preview
- [x] Style comparison grid
- [x] Colab notebook
- [ ] Gradio/Streamlit web UI
- [ ] Video file batch processing with ffmpeg
- [ ] ONNX export for edge deployment
- [ ] MediaPipe face mesh (replace Haar cascade)
- [ ] Custom style training guide

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Quick version:

```bash
# Fork → clone → branch
git checkout -b feature/your-feature

# Make changes, test
python -m pytest tests/

# Commit with conventional format
git commit -m "feat: add batch video processing"

# Push → open PR
git push origin feature/your-feature
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

This project builds upon excellent research and open-source work:

- [VToonify](https://github.com/williamyang1991/VToonify) — High-resolution portrait video style transfer (SIGGRAPH Asia 2022)
- [AnimeGANv2](https://github.com/bryandlee/animegan2-pytorch) — Lightweight anime-style transformation
- [OpenCV](https://opencv.org/) — Face detection via Haar cascades

---

## 📖 Citation

If you use ToonForge in your work, please consider citing:

```bibtex
@software{toonforge2025,
  title={ToonForge: Real-Time Multi-Model Cartoon Stylization},
  author={YOUR_NAME},
  year={2025},
  url={https://github.com/YOUR_USERNAME/toonforge-realtime-cartoon-stylization}
}
```

---

<p align="center">
  <b>⭐ Star this repo if you find it useful!</b><br>
  Questions? Open an <a href="https://github.com/YOUR_USERNAME/toonforge-realtime-cartoon-stylization/issues">issue</a> or start a <a href="https://github.com/YOUR_USERNAME/toonforge-realtime-cartoon-stylization/discussions">discussion</a>.
</p>
