# Contributing to ToonForge

Thanks for your interest in contributing! This guide covers everything you need to get started.

## Development Setup

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/toonforge-realtime-cartoon-stylization.git
cd toonforge-realtime-cartoon-stylization

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dev dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Branch Naming

Use descriptive prefixes:

- `feature/` — New functionality (e.g., `feature/gradio-web-ui`)
- `fix/` — Bug fixes (e.g., `fix/multi-face-alignment-crash`)
- `docs/` — Documentation only (e.g., `docs/add-training-guide`)
- `refactor/` — Code restructuring without behavior change
- `perf/` — Performance improvements

## Making Changes

1. **Create a branch** from `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature
   ```

2. **Write code** following the existing style:
   - Type hints on all public functions
   - Docstrings for classes and non-trivial functions
   - Keep GPU/CPU logic separate where possible

3. **Test your changes**:
   ```bash
   python -m pytest tests/ -v
   ```

4. **Commit** with [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat: add batch video processing with progress bar
   fix: handle zero-face edge case in blender module
   docs: add ONNX export instructions
   perf: reduce GPU memory by reusing face crop tensor
   ```

## Pull Request Process

1. **Push** your branch:
   ```bash
   git push origin feature/your-feature
   ```

2. **Open a PR** against `main` on GitHub.

3. **Fill out the PR template** — describe what changed, why, and how to test.

4. **Checklist before requesting review**:
   - [ ] Code runs without errors on a T4 GPU (Colab is fine)
   - [ ] No hardcoded paths or credentials
   - [ ] New features have at least basic tests
   - [ ] README updated if user-facing behavior changed
   - [ ] No print statements left in (use `logging`)

5. **Review** — maintainers will review within a few days. Address feedback by pushing new commits to the same branch.

6. **Merge** — once approved, a maintainer will squash-merge your PR.

## Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md). Include:

- Python version, PyTorch version, GPU model
- Minimal reproduction steps
- Full error traceback
- Input image/video if possible (blur faces if needed for privacy)

## Feature Requests

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md). Describe:

- What you want to achieve
- Why existing features don't cover it
- Any implementation ideas you have

## Code Style

- **Formatter**: `black` with default settings
- **Linter**: `ruff`
- **Type checking**: `mypy` (strict mode on new code)
- **Imports**: `isort` with black-compatible profile

```bash
black toonforge/ tests/
ruff check toonforge/ tests/
mypy toonforge/
```

## Architecture Guidelines

- `pipeline.py` is the orchestrator — it shouldn't contain model-specific logic
- Engine modules (`scene_engine.py`, `face_engine.py`) handle their own model loading and inference
- `blender.py` handles all compositing — keep blending math here, not in pipeline
- `detector.py` is swappable — designed so Haar can be replaced with MediaPipe later

## GPU Memory Rules

- Always use `@torch.inference_mode()` for inference functions
- Release intermediate tensors explicitly in loops
- Test with `torch.cuda.max_memory_allocated()` and document in PR

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
