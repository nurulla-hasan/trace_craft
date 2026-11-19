"""Stage 5: Render-Verify Loop & Diagnostic Zoom Comparison Generator."""

from __future__ import annotations
import io
import re
import cv2
import numpy as np
from PIL import Image
import resvg_py

from .color_separation import bgr_to_lab_f32
from .preprocess import load_image


def render_svg_bgr(
    svg_path: str,
    width: int,
    height: int,
    strip_bg_rect: bool = False,
    background: str | None = "#ffffff",
) -> np.ndarray:
    """Render an SVG file to a BGR uint8 numpy array using resvg."""
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_str = f.read()
    if strip_bg_rect:
        svg_str = re.sub(r'<rect\s+id="background"[^>]*/>\s*', "", svg_str, count=1)

    png_bytes = resvg_py.svg_to_bytes(
        svg_string=svg_str, width=width, height=height, background=background
    )
    pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def evaluate_svg(src_path: str, svg_path: str, preview_path: str | None = None) -> dict:
    """Compare rendered SVG against original raster and optionally save top/bottom preview."""
    orig = load_image(src_path)
    h, w = orig.shape[:2]
    rendered = render_svg_bgr(svg_path, width=w, height=h, strip_bg_rect=False, background="#ffffff")

    mae = float(np.abs(rendered.astype(np.float32) - orig.astype(np.float32)).mean() / 255.0)
    pixel_match = 1.0 - mae

    lab_o = bgr_to_lab_f32(orig)
    lab_r = bgr_to_lab_f32(rendered)
    mean_de = float(np.linalg.norm(lab_o - lab_r, axis=2).mean())

    if preview_path:
        sep = np.full((6, w, 3), 90, dtype=np.uint8)
        stacked = np.vstack([orig, sep, rendered])
        cv2.imwrite(preview_path, stacked)

    return {
        "pixel_match": round(pixel_match, 4),
        "mean_lab_de": round(mean_de, 3),
    }


def save_zoom_audit(
    src_path: str,
    svg_path: str,
    crops: list[tuple[str, tuple[int, int, int, int]]],
    out_path: str,
    zoom: int = 4,
) -> None:
    """Create a side-by-side 4x zoom audit grid: [Original | SVG on White | SVG on Dark Gray]."""
    orig = load_image(src_path)
    h, w = orig.shape[:2]

    rend_white = render_svg_bgr(
        svg_path, width=w * zoom, height=h * zoom, strip_bg_rect=True, background="#ffffff"
    )
    rend_dark = render_svg_bgr(
        svg_path, width=w * zoom, height=h * zoom, strip_bg_rect=True, background="#3a3d42"
    )

    rows: list[np.ndarray] = []
    target_w = 0
    panels: list[tuple[str, np.ndarray]] = []

    for name, (x0, y0, x1, y1) in crops:
        o_crop = cv2.resize(
            orig[y0:y1, x0:x1],
            ((x1 - x0) * zoom, (y1 - y0) * zoom),
            interpolation=cv2.INTER_NEAREST,
        )
        w_crop = rend_white[y0 * zoom : y1 * zoom, x0 * zoom : x1 * zoom]
        d_crop = rend_dark[y0 * zoom : y1 * zoom, x0 * zoom : x1 * zoom]

        sep = np.full((o_crop.shape[0], 6, 3), 120, dtype=np.uint8)
        row = np.hstack([o_crop, sep, w_crop, sep, d_crop])
        # Label banner
        banner = np.full((28, row.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(
            banner,
            f"{name}   |   LEFT: Original (1x nearest)   |   MID: SVG on White   |   RIGHT: SVG on Dark",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        panel = np.vstack([banner, row])
        target_w = max(target_w, panel.shape[1])
        panels.append((name, panel))

    for _, p in panels:
        if p.shape[1] < target_w:
            pad_r = np.full((p.shape[0], target_w - p.shape[1], 3), 30, dtype=np.uint8)
            p = np.hstack([p, pad_r])
        rows.append(p)
        rows.append(np.full((8, target_w, 3), 20, dtype=np.uint8))

    if rows:
        cv2.imwrite(out_path, np.vstack(rows))
