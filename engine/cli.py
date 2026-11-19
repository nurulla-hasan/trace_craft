"""CLI runner to vectorize images and generate verification previews."""

from __future__ import annotations
import argparse
import time
from .tracer import vectorize_image
from .verify import evaluate_svg


def main() -> None:
    parser = argparse.ArgumentParser(description="TraceCraft Vectorization Engine CLI")
    parser.add_argument("input", help="Input raster image path")
    parser.add_argument("output", help="Output SVG path")
    parser.add_argument("--mode", choices=["auto", "binary", "color"], default="auto")
    parser.add_argument("--preview", help="Optional output PNG path for top/bottom comparison")
    parser.add_argument("--no-bg", action="store_true", help="Omit background rect in SVG")
    args = parser.parse_args()

    t0 = time.perf_counter()
    stats = vectorize_image(args.input, args.output, mode=args.mode, include_bg=not args.no_bg)
    dt = time.perf_counter() - t0
    metrics = evaluate_svg(args.input, args.output, preview_path=args.preview)

    print(
        f"Done in {dt:.2f}s | mode={stats['mode']} | {stats['width']}x{stats['height']} | "
        f"regions={stats['regions']} | paths={stats['paths']} | "
        f"size={stats['bytes']/1024:.1f} KB | match={metrics['pixel_match']} | "
        f"mean_dE={metrics['mean_lab_de']}"
    )


if __name__ == "__main__":
    main()
