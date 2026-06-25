"""
CLI entry point for ToonForge.

Usage:
    python -m toonforge.run --input photo.jpg --style cartoon1-d
    python -m toonforge.run --webcam --style arcane1-d --preset balanced
    python -m toonforge.run --compare photo.jpg   # 5-style grid comparison
"""

import argparse
import logging
import time
import sys
import os

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("toonforge")


def parse_args():
    p = argparse.ArgumentParser(
        description="ToonForge — Real-time hybrid cartoon stylization"
    )
    p.add_argument("--input", type=str, help="Path to input image")
    p.add_argument("--output", type=str, default="output.jpg", help="Output path")
    p.add_argument("--webcam", action="store_true", help="Live webcam mode")
    p.add_argument("--compare", type=str, help="Compare 5 styles on one image")
    p.add_argument("--style", type=str, default="cartoon1-d", help="Face style name")
    p.add_argument("--degree", type=float, default=0.5, help="Style degree (0.0-1.0)")
    p.add_argument(
        "--preset", type=str, default="balanced",
        choices=["fast", "balanced", "quality"],
        help="Performance preset for webcam mode",
    )
    p.add_argument(
        "--padding", type=int, nargs=4, default=[200, 200, 200, 200],
        help="Face crop padding: top bottom left right",
    )
    p.add_argument("--device", type=str, default="cuda", help="Compute device")
    p.add_argument("--list-styles", action="store_true", help="List available styles")
    return p.parse_args()


def run_single(pipeline, args):
    """Process a single image."""
    img = Image.open(args.input).convert("RGB")
    log.info("Processing: %s (%dx%d)", args.input, *img.size)

    t0 = time.time()
    result, mode = pipeline.process(img, args.degree, tuple(args.padding))
    elapsed = time.time() - t0

    result.save(args.output, quality=95)
    log.info("Done: %s | %s | %.0fms → %s", mode, args.style, elapsed * 1000, args.output)


def run_compare(pipeline, args):
    """Compare multiple styles on one image."""
    styles = ["cartoon1-d", "arcane1-d", "pixar1-d", "comic1-d", "caricature1-d"]
    img = Image.open(args.compare).convert("RGB")
    log.info("Comparing %d styles on: %s", len(styles), args.compare)

    for style in styles:
        pipeline.set_style(style)
        result, mode = pipeline.process(img, 0.5, tuple(args.padding))
        out_path = f"compare_{style}.jpg"
        result.save(out_path, quality=95)
        log.info("  %s (%s) → %s", style, mode, out_path)

    log.info("Comparison complete — %d images saved", len(styles))


def main():
    args = parse_args()

    from .pipeline import ToonForgePipeline

    pipeline = ToonForgePipeline(device=args.device)

    if args.list_styles:
        pipeline.load(args.style)
        print("Available styles:")
        for s in pipeline.available_styles:
            tag = "adjustable" if s.endswith("-d") else "fixed"
            print(f"  {s:20s} ({tag})")
        return

    pipeline.load(args.style)

    if args.input:
        run_single(pipeline, args)
    elif args.compare:
        run_compare(pipeline, args)
    elif args.webcam:
        log.info("Webcam mode — use Colab notebook for browser webcam support")
        log.info("Or implement cv2.VideoCapture(0) for local webcam")
    else:
        log.error("Specify --input, --webcam, or --compare. Use --help for options.")
        sys.exit(1)


if __name__ == "__main__":
    main()
