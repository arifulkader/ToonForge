# Changelog

All notable changes to ToonForge will be documented here.

## [0.3.0] - 2025-XX-XX

### Added
- Multi-face hybrid pipeline (SceneNet + FaceForge)
- Elliptical feathered blending for seamless face compositing
- Webcam live preview with 3 performance presets
- 5-style comparison grid mode
- CLI with `--input`, `--webcam`, `--compare` modes
- Modular architecture: detector, scene_engine, face_engine, blender

### Changed
- Restructured from notebook into installable Python package
- Face regions now preserved during SceneNet pass to prevent double-stylization

## [0.2.0] - 2025-XX-XX

### Added
- AnimeGANv2 full-scene integration
- Basic face detection with Haar cascades
- Single-face VToonify stylization

## [0.1.0] - 2025-XX-XX

### Added
- Initial VToonify wrapper with Colab webcam support
- Single style (cartoon1-d) proof of concept
