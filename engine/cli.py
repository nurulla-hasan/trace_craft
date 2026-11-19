"""CLI runner to vectorize images and optionally render a verification preview."""

from __future__ import annotations

import argparse
import io
import time
import cv2
import numpy as np
from PIL import Image
import resvg_py

from .pipeline import load_image, vectorize_image


def evaluate_svg(src_path: str, svg_path: str, preview_path: str | None = None) -> dict:
    orig = load_image(src_path)
    h, w = orig.shape[:2]
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_str = f.read()
    png_bytes = resvg_py.svg_to_bytes(svg_string=svg_str, width=w, height=h, background="#ffffff")
    rendered = cv2.cvtColor(np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB")), cv2.COLOR_RGB2BGR)
    mae = float(np.abs(rendered.astype(np.float32) - orig.astype(np.float32)).mean() / 255.0)
    if preview_path:
        sep = np.full((6, w, 3), 90, dtype=np.uint8)
        cv2.imwrite(preview_path, np.vstack([orig, sep, rendered]))
    return {"pixel_match": round(1.0 - mae, 4)}


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
        f"size={stats['bytes'] / 1024:.1f} KB | match={metrics['pixel_match']}"
    )


if __name__ == "__main__":
    main()
