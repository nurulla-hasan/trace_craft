"""Stage 4: Sub-Pixel Alpha-Field Color Tracer & Binary Outline Tracer."""

from __future__ import annotations
import cv2
import numpy as np
from scipy import ndimage as ndi

from .preprocess import (
    load_image,
    detect_mode,
    denoise_1x,
    extract_background_mask,
    split_foreground_regions,
)
from .color_separation import extract_pure_palette
from .unmix import unmix_pixels
from .bezier import mask_to_svg_path_d, _fmt
from .shared_graph import extract_shared_edge_paths

SCALE = 4


def _bgr_to_hex(bgr: np.ndarray) -> str:
    b, g, r = [int(np.clip(round(float(v)), 0, 255)) for v in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def _boost_thin_strokes(alpha: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    """Preserve thin sub-pixel outer appendages (1px antennas, tentacles, spines) against background.

    Crucially restricted to TRUE thin appendages (pixels away from any thick body core),
    so it NEVER fires on the outer anti-aliased boundary of thick shapes like whales.
    """
    K = alpha.shape[0] - 1
    if K <= 0:
        return alpha

    out = alpha.copy()
    fg = 1.0 - out[K]
    h, w = fg.shape
    if h < 5 or w < 5:
        return out

    # Identify thick body cores (foreground pixels >= 2.2px inside the silhouette)
    fg_dist = ndi.distance_transform_edt(~bg_mask)
    thick_core = fg_dist > 2.2
    near_thick_core = cv2.dilate(thick_core.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0

    p = np.pad(fg, 2, mode="constant", constant_values=0.0)
    bg_th = 0.14
    c = p[2:-2, 2:-2]

    v_thin = (
        ((p[1:-3, 2:-2] < bg_th) | (p[0:-4, 2:-2] < bg_th))
        & ((p[3:-1, 2:-2] < bg_th) | (p[4:, 2:-2] < bg_th))
        & (c >= p[1:-3, 2:-2] - 0.04)
        & (c >= p[3:-1, 2:-2] - 0.04)
    )
    h_thin = (
        ((p[2:-2, 1:-3] < bg_th) | (p[2:-2, 0:-4] < bg_th))
        & ((p[2:-2, 3:-1] < bg_th) | (p[2:-2, 4:] < bg_th))
        & (c >= p[2:-2, 1:-3] - 0.04)
        & (c >= p[2:-2, 3:-1] - 0.04)
    )
    d1_thin = (
        ((p[1:-3, 1:-3] < bg_th) | (p[0:-4, 0:-4] < bg_th))
        & ((p[3:-1, 3:-1] < bg_th) | (p[4:, 4:] < bg_th))
        & (c >= p[1:-3, 1:-3] - 0.04)
        & (c >= p[3:-1, 3:-1] - 0.04)
    )
    d2_thin = (
        ((p[1:-3, 3:-1] < bg_th) | (p[0:-4, 4:] < bg_th))
        & ((p[3:-1, 1:-3] < bg_th) | (p[4:, 0:-4] < bg_th))
        & (c >= p[1:-3, 3:-1] - 0.04)
        & (c >= p[3:-1, 1:-3] - 0.04)
    )

    ridge = (
        (v_thin | h_thin | d1_thin | d2_thin)
        & (~near_thick_core)
        & (fg >= 0.13)
        & (fg < 0.62)
    )
    ry, rx = np.where(ridge)
    if len(ry) == 0:
        return out

    fg_ch = out[:K, ry, rx]
    fg_sum = np.maximum(fg_ch.sum(axis=0), 1e-6)
    new_fg = np.minimum(0.76, np.maximum(0.56, fg[ry, rx] * 2.1))
    scale_factor = new_fg / fg_sum
    out[:K, ry, rx] = fg_ch * scale_factor[None, :]
    out[K, ry, rx] = 1.0 - new_fg
    return out


def _boost_interior_lines(alpha: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    """Preserve 1px interior lines, baleen grooves, mouth creases, and tiny 1-2px eyes.

    When an interior detail color `k` is anti-aliased against a surrounding body/belly color,
    its 1x coverage along the 1px centerline is often 0.22..0.50 (so argmax would erase it!).
    Detecting 1D ridges and 2D pinpoint peaks of `alpha[k]` strictly inside the foreground
    and boosting them above 0.56 ensures they survive as crisp vector details.
    """
    K = alpha.shape[0] - 1
    if K <= 1:
        return alpha

    out = alpha.copy()
    h, w = bg_mask.shape
    if h < 5 or w < 5:
        return out

    # Strictly interior foreground (at least 1px inside foreground silhouette)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    inner_fg = (cv2.erode((~bg_mask).astype(np.uint8), cross) > 0) & (out[K] < 0.15)

    # Order colors by total coverage; boost detail/line colors (not the single largest base color)
    totals = np.array([out[k].sum() for k in range(K)])
    dom_k = int(np.argmax(totals))

    for k in range(K):
        if k == dom_k and K <= 2:
            # Even in a 2-color icon (e.g. dark blue + light blue whale), dark blue baleen lines
            # run inside the light blue belly! So check every color where it is locally a thin ridge.
            pass

        ak = out[k]
        p = np.pad(ak, 1, mode="constant", constant_values=0.0)
        c = p[1:-1, 1:-1]

        # 1D ridge check across 4 orientations: center is local maximum and drops by >= 0.10 on BOTH sides
        drop = 0.09
        v_ridge = (c - p[:-2, 1:-1] >= drop) & (c - p[2:, 1:-1] >= drop)
        h_ridge = (c - p[1:-1, :-2] >= drop) & (c - p[1:-1, 2:] >= drop)
        d1_ridge = (c - p[:-2, :-2] >= drop) & (c - p[2:, 2:] >= drop)
        d2_ridge = (c - p[:-2, 2:] >= drop) & (c - p[2:, :-2] >= drop)

        is_ridge = (
            inner_fg
            & (v_ridge | h_ridge | d1_ridge | d2_ridge)
            & (ak >= 0.20)
            & (ak < 0.62)
        )
        ry, rx = np.where(is_ridge)
        if len(ry) == 0:
            continue

        boosted = np.minimum(0.74, np.maximum(0.58, ak[ry, rx] * 1.9))
        rem = np.maximum(1.0 - ak[ry, rx], 1e-6)
        scale_other = (1.0 - boosted) / rem
        for m in range(K + 1):
            if m == k:
                out[m, ry, rx] = boosted
            else:
                out[m, ry, rx] *= scale_other

    return out


def trace_color_region(
    crop_bgr: np.ndarray,
    crop_bg_mask: np.ndarray,
    bg_color: np.ndarray,
    scale: int = SCALE,
    max_k: int = 12,
    merge_de: float = 9.0,
    fit_tol_1x: float = 0.48,
    use_shared_graph: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Trace a single flat-color region into clean SVG <path> elements."""
    den = denoise_1x(crop_bgr)
    fg_mask = ~crop_bg_mask

    palette_bgr, pure_mask = extract_pure_palette(
        den, fg_mask, bg_color, max_k=max_k, merge_de=merge_de
    )
    K = len(palette_bgr)
    if K == 0:
        return [], palette_bgr

    # Unmix on sharp 1x crop_bgr so 1px baleen lines, eyes, and antennas aren't blurred away!
    alpha = unmix_pixels(crop_bgr, crop_bg_mask, pure_mask, palette_bgr, bg_color)
    alpha = _boost_thin_strokes(alpha, crop_bg_mask)
    alpha = _boost_interior_lines(alpha, crop_bg_mask)

    h, w = crop_bgr.shape[:2]
    H4, W4 = h * scale, w * scale

    # Upscale continuous alpha fields (NOT RGB!) so zero fake colors can ever appear
    alpha_4x = np.zeros((K + 1, H4, W4), dtype=np.float32)
    for k in range(K):
        up = cv2.resize(alpha[k], (W4, H4), interpolation=cv2.INTER_CUBIC)
        up = cv2.GaussianBlur(up, (0, 0), 1.1)
        alpha_4x[k] = np.clip(up, 0.0, 1.0)

    # Smooth the background channel with slightly wider sigma (1.45 at 4x) for glass-smooth outer silhouettes
    up_bg = cv2.resize(alpha[K], (W4, H4), interpolation=cv2.INTER_CUBIC)
    up_bg = cv2.GaussianBlur(up_bg, (0, 0), 1.45)
    alpha_4x[K] = np.clip(up_bg, 0.0, 1.0)

    label_4x = alpha_4x.argmax(axis=0).astype(np.int32)

    # Remove tiny isolated outer foreground confetti specks (< 36 pixels at 4x = < 2.2px at 1x)
    fg_4x = (label_4x != K).astype(np.uint8)
    n_cc, cc_lab, cc_stats, _ = cv2.connectedComponentsWithStats(fg_4x, connectivity=8)
    for cid in range(1, n_cc):
        if cc_stats[cid, cv2.CC_STAT_AREA] < 36:
            label_4x[cc_lab == cid] = K

    # Order colors by area descending (largest base layer first, details on top)
    areas = np.array([(label_4x == k).sum() for k in range(K)])
    order = [int(k) for k in np.argsort(-areas) if areas[k] >= 12]

    paths: list[str] = []

    if use_shared_graph:
        shared_d = extract_shared_edge_paths(
            label_4x,
            bg_label=K,
            scale=float(scale),
            min_loop_area_4x=14.0,
            fit_tol_1x=fit_tol_1x,
            corner_deg=54.0,
        )
        for k in order:
            d_str = shared_d.get(k, "")
            if d_str:
                hex_c = _bgr_to_hex(palette_bgr[k])
                paths.append(
                    f'<path fill="{hex_c}" stroke="{hex_c}" stroke-width="0.2" '
                    f'stroke-linejoin="round" fill-rule="evenodd" d="{d_str}"/>'
                )
        return paths, palette_bgr

    kern3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for rank, k in enumerate(order):
        mask_k = label_4x == k
        if rank < len(order) - 1:
            higher_ids = order[rank + 1 :]
            higher_fg = np.isin(label_4x, higher_ids)
            dil_k = cv2.dilate(mask_k.astype(np.uint8), kern3) > 0
            layer_mask = mask_k | (dil_k & higher_fg)
        else:
            layer_mask = mask_k

        mask_u8 = (layer_mask.astype(np.uint8)) * 255
        d_str = mask_to_svg_path_d(
            mask_u8,
            scale=float(scale),
            min_area_4x=14.0,
            fit_tol_1x=fit_tol_1x,
            corner_deg=54.0,
        )
        if d_str:
            hex_c = _bgr_to_hex(palette_bgr[k])
            paths.append(f'<path fill="{hex_c}" fill-rule="evenodd" d="{d_str}"/>')

    return paths, palette_bgr


def trace_binary_image(
    img_bgr: np.ndarray,
    scale: int = SCALE,
    thresh: int = 130,
    fit_tol_1x: float = 0.28,
) -> str:
    """High-precision sub-pixel binary/outline vectorization (preserves exact layout & title)."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 1.1)
    bw_fg = np.where(up < thresh, 255, 0).astype(np.uint8)

    d_str = mask_to_svg_path_d(
        bw_fg,
        scale=float(scale),
        min_area_4x=10.0,
        fit_tol_1x=fit_tol_1x,
        corner_deg=50.0,
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        f'  <rect id="background" width="{w}" height="{h}" fill="#ffffff"/>\n'
        f'  <g id="artwork">\n'
        f'    <path fill="#000000" fill-rule="evenodd" d="{d_str}"/>\n'
        f'  </g>\n'
        f'</svg>\n'
    )
    return svg


def vectorize_image(
    input_path: str,
    output_svg_path: str,
    mode: str = "auto",
    include_bg: bool = True,
) -> dict:
    """End-to-end vectorization entry point for any raster image."""
    img_bgr = load_image(input_path)
    h, w = img_bgr.shape[:2]

    chosen_mode = detect_mode(img_bgr) if mode == "auto" else mode

    if chosen_mode == "binary":
        svg = trace_binary_image(img_bgr)
        with open(output_svg_path, "w", encoding="utf-8") as f:
            f.write(svg)
        return {
            "mode": "binary",
            "width": w,
            "height": h,
            "regions": 1,
            "paths": svg.count("<path"),
            "bytes": len(svg.encode("utf-8")),
        }

    bg_mask, bg_color = extract_background_mask(img_bgr, bg_tol=14)
    regions = split_foreground_regions(img_bgr, bg_mask, bg_color)

    bg_hex = _bgr_to_hex(bg_color)
    groups: list[str] = []
    total_paths = 0
    palette_sizes: list[int] = []

    for idx, reg in enumerate(regions, start=1):
        x0, y0, x1, y1 = reg["bbox"]
        pad = reg["pad"]
        paths, pal = trace_color_region(reg["crop_bgr"], reg["crop_bg_mask"], bg_color)
        if not paths:
            continue
        total_paths += len(paths)
        palette_sizes.append(len(pal))
        tx = x0 - pad
        ty = y0 - pad
        inner = "\n    ".join(paths)
        groups.append(
            f'  <g id="region-{idx:02d}" transform="translate({_fmt(tx)},{_fmt(ty)})">\n'
            f"    {inner}\n"
            f"  </g>"
        )

    bg_rect = (
        f'  <rect id="background" width="{w}" height="{h}" fill="{bg_hex}"/>\n'
        if include_bg
        else ""
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        f"{bg_rect}"
        + "\n".join(groups)
        + "\n</svg>\n"
    )

    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write(svg)

    return {
        "mode": "color",
        "width": w,
        "height": h,
        "regions": len(groups),
        "paths": total_paths,
        "avg_palette": round(float(np.mean(palette_sizes)), 2) if palette_sizes else 0.0,
        "bytes": len(svg.encode("utf-8")),
    }
