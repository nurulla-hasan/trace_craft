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

SCALE = 4


def _bgr_to_hex(bgr: np.ndarray) -> str:
    b, g, r = [int(np.clip(round(float(v)), 0, 255)) for v in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def _apply_4x_dog_interior_details(
    fg_alpha_4x: np.ndarray,
    crop_bgr: np.ndarray,
    palette_bgr: np.ndarray,
    fg_4x: np.ndarray,
    scale: int = SCALE,
) -> np.ndarray:
    """Detect 1px interior lines (baleen grooves, mouth creases, shell lines) and pinpoint eyes
    directly on the 4x bicubic-upscaled image using Difference-of-Gaussians (DoG).

    Why at 4x instead of 1x?
    At 1x, a 60-degree 1px line straddles two pixel columns (x + 0.5) every 2-3 rows, splitting
    its contrast 50/50 and causing 1x ridge detectors to drop out periodically (breaking lines
    into dashed droplets!). At 4x resolution, `4*(x + 0.5)` lies on an exact 4x grid center, so
    `DoG_4x = Blur(img_4x, sigma=1.1) - Blur(img_4x, sigma=5.2)` is translation-invariant and
    produces 100% continuous, unbroken strokes!
    """
    K = len(palette_bgr)
    if K <= 1:
        return fg_alpha_4x

    H4, W4 = fg_4x.shape
    out = fg_alpha_4x.copy()

    # Keep strictly >= 6px (1.5px at 1x) inside fg_4x so outer JPEG border ringing is never touched
    ell13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    deep_4x = cv2.erode(fg_4x.astype(np.uint8), ell13) > 0
    if not np.any(deep_4x):
        return out

    up_4x = cv2.resize(crop_bgr, (W4, H4), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    # At 4x scale: sigma=1.4 is 0.35px at 1x (line scale), sigma=12.0 is 3.0px at 1x (background scale)!
    fine_4x = cv2.GaussianBlur(up_4x, (0, 0), 1.4)
    coarse_4x = cv2.GaussianBlur(up_4x, (0, 0), 12.0)

    dog_vec = fine_4x - coarse_4x
    dog_mag = np.linalg.norm(dog_vec, axis=2)

    # 1D ridge check on dog_mag at 4x using step = 6 (1.5px in 1x units) perpendicular offset
    step = 6
    p = np.pad(dog_mag, step, mode="constant", constant_values=0.0)
    c = p[step:-step, step:-step]
    drop = 3.5

    v_line = (c - p[:-2 * step, step:-step] >= drop) & (c - p[2 * step :, step:-step] >= drop)
    h_line = (c - p[step:-step, :-2 * step] >= drop) & (c - p[step:-step, 2 * step :] >= drop)
    d1_line = (c - p[:-2 * step, :-2 * step] >= drop) & (c - p[2 * step :, 2 * step :] >= drop)
    d2_line = (c - p[:-2 * step, 2 * step :] >= drop) & (c - p[2 * step :, :-2 * step] >= drop)

    is_detail = deep_4x & (
        ((v_line | h_line | d1_line | d2_line) & (dog_mag >= 9.0))
        | (dog_mag >= 22.0)  # strong pinpoint eye / spot
    )

    ry, rx = np.where(is_detail)
    if len(ry) == 0:
        return out

    pal_f = palette_bgr.astype(np.float32)
    # Vectorized assignment of each 4x detail pixel to its matching detail palette color
    C_bg_loc = coarse_4x[ry, rx]  # (M, 3)
    P_fine = fine_4x[ry, rx]      # (M, 3)
    V_det = dog_vec[ry, rx]       # (M, 3)
    V_norm = np.maximum(np.linalg.norm(V_det, axis=1, keepdims=True), 1e-6)
    V_unit = V_det / V_norm

    d_bg_pal = np.linalg.norm(C_bg_loc[:, None, :] - pal_f[None, :, :], axis=2)  # (M, K)
    k_local_bg = d_bg_pal.argmin(axis=1)  # (M,)
    C_pal_bg = pal_f[k_local_bg]          # (M, 3)

    best_k = np.full(len(ry), -1, dtype=np.int32)
    best_score = np.full(len(ry), 1e9, dtype=np.float32)

    for k in range(K):
        dir_k = pal_f[k][None, :] - C_pal_bg  # (M, 3)
        dist_k = np.linalg.norm(dir_k, axis=1)  # (M,)
        valid = (k != k_local_bg) & (dist_k >= 12.0)
        if not np.any(valid):
            continue

        cos_sim = np.sum((dir_k / np.maximum(dist_k[:, None], 1e-6)) * V_unit, axis=1)
        valid &= cos_sim >= 0.55
        if not np.any(valid):
            continue

        t_proj = np.clip(
            np.sum((P_fine - C_pal_bg) * dir_k, axis=1) / np.maximum(dist_k * dist_k, 1e-6),
            0.12,
            1.0,
        )
        recon = C_pal_bg + t_proj[:, None] * dir_k
        err = np.linalg.norm(P_fine - recon, axis=1)
        overshoot = np.abs(dist_k - dog_mag[ry, rx] * 2.2) * 0.20
        score = err + overshoot

        better = valid & (score < best_score)
        if np.any(better):
            best_score[better] = score[better]
            best_k[better] = k

    has_match = best_k >= 0
    if np.any(has_match):
        my = ry[has_match]
        mx = rx[has_match]
        mk = best_k[has_match]
        out[mk, my, mx] = np.maximum(out[mk, my, mx], 0.85)

    # Smooth each boosted channel slightly at 4x so the extracted 4x detail strokes have sleek curves
    for k in range(K):
        out[k] = cv2.GaussianBlur(out[k], (0, 0), 0.85)

    return out


def _boost_thin_strokes(alpha: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    """Preserve thin sub-pixel outer appendages (1px antennas, tentacles, spines) against background."""
    K = alpha.shape[0] - 1
    if K <= 0:
        return alpha

    out = alpha.copy()
    fg = 1.0 - out[K]
    h, w = fg.shape
    if h < 5 or w < 5:
        return out

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
    if len(ry) > 0:
        fg_ch = out[:K, ry, rx]
        fg_sum = np.maximum(fg_ch.sum(axis=0), 1e-6)
        new_fg = np.minimum(0.76, np.maximum(0.58, fg[ry, rx] * 2.1))
        scale_factor = new_fg / fg_sum
        out[:K, ry, rx] = fg_ch * scale_factor[None, :]
        out[K, ry, rx] = 1.0 - new_fg

    return out


def _boost_interior_lines(
    alpha: np.ndarray,
    crop_bgr: np.ndarray,
    palette_bgr: np.ndarray,
    bg_mask: np.ndarray,
) -> np.ndarray:
    """Detect and preserve 1px interior lines (baleen grooves, mouth creases, shell lines) and
    pinpoint eyes directly from the 1x raster image, then bridge diagonal saddle corners so
    they interpolate into continuous, smooth strokes at 4x resolution.
    """
    K = len(palette_bgr)
    if K <= 1:
        return alpha

    out = alpha.copy()
    h, w = bg_mask.shape
    if h < 5 or w < 5:
        return out

    # Strictly interior foreground (at least 2px inside foreground silhouette)
    ell5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    deep_fg = (cv2.erode((~bg_mask).astype(np.uint8), ell5) > 0) & (out[K] < 0.08)
    if not np.any(deep_fg):
        return out

    img_f = crop_bgr.astype(np.float32)
    pal_f = palette_bgr.astype(np.float32)

    # Local background color inside deep_fg via 5x5 median filter
    med_bgr = cv2.medianBlur(crop_bgr, 5).astype(np.float32)
    diff_vec = img_f - med_bgr  # vector pointing from local background toward the detail pixel
    diff_mag = np.linalg.norm(diff_vec, axis=2)

    # Check if diff_mag forms a 1D ridge (across 1px or 2px perpendicular neighbors) OR a strong 2D eye peak
    p = np.pad(diff_mag, 2, mode="constant", constant_values=0.0)
    c = p[2:-2, 2:-2]
    drop = 7.5

    # Compare center against perpendicular neighbors (either 1px or 2px away to handle 1-2px lines)
    n_up = np.minimum(p[1:-3, 2:-2], p[0:-4, 2:-2])
    n_dn = np.minimum(p[3:-1, 2:-2], p[4:, 2:-2])
    n_lt = np.minimum(p[2:-2, 1:-3], p[2:-2, 0:-4])
    n_rt = np.minimum(p[2:-2, 3:-1], p[2:-2, 4:])
    n_ul = np.minimum(p[1:-3, 1:-3], p[0:-4, 0:-4])
    n_dr = np.minimum(p[3:-1, 3:-1], p[4:, 4:])
    n_ur = np.minimum(p[1:-3, 3:-1], p[0:-4, 4:])
    n_dl = np.minimum(p[3:-1, 1:-3], p[4:, 0:-4])

    v_line = (c - n_up >= drop) & (c - n_dn >= drop)
    h_line = (c - n_lt >= drop) & (c - n_rt >= drop)
    d1_line = (c - n_ul >= drop) & (c - n_dr >= drop)
    d2_line = (c - n_ur >= drop) & (c - n_dl >= drop)

    is_detail = deep_fg & (
        ((v_line | h_line | d1_line | d2_line) & (diff_mag >= 13.0))
        | (diff_mag >= 28.0)  # strong pinpoint detail like an eye
    )

    ry, rx = np.where(is_detail)
    if len(ry) > 0:
        # For each detected detail pixel, identify the local background palette color `k_bg`
        # and the best detail palette color `k_line != k_bg` whose direction from `k_bg` aligns
        # with `img_f - med_bgr` (and is closest in RGB).
        d_med_pal = np.linalg.norm(med_bgr[ry, rx][:, None, :] - pal_f[None, :, :], axis=2)
        k_local_bg = d_med_pal.argmin(axis=1)

        P_det = img_f[ry, rx]
        V_det = diff_vec[ry, rx]
        V_norm = np.maximum(np.linalg.norm(V_det, axis=1, keepdims=True), 1e-6)
        V_unit = V_det / V_norm

        for idx in range(len(ry)):
            y, x = int(ry[idx]), int(rx[idx])
            k_bg = int(k_local_bg[idx])
            c_bg = pal_f[k_bg]
            p_rgb = P_det[idx]
            v_u = V_unit[idx]

            best_k = -1
            best_score = 1e9
            for k in range(K):
                if k == k_bg:
                    continue
                dir_k = pal_f[k] - c_bg
                dist_k = float(np.linalg.norm(dir_k))
                if dist_k < 12.0:
                    continue
                # Must point in the same direction as the detail departure from local median
                cos_sim = float(np.dot(dir_k / dist_k, v_u))
                if cos_sim < 0.55:
                    continue
                # Project p_rgb onto segment [c_bg, pal_f[k]]
                t_proj = np.clip(float(np.dot(p_rgb - c_bg, dir_k)) / (dist_k * dist_k), 0.15, 1.0)
                recon = c_bg + t_proj * dir_k
                err = float(np.linalg.norm(p_rgb - recon))
                # Prefer closest palette color along that ray (so teal groove picks teal or medium blue, not black!)
                overshoot = abs(dist_k - float(diff_mag[y, x]) * 1.6) * 0.25
                score = err + overshoot
                if score < best_score:
                    best_score = score
                    best_k = k

            if best_k >= 0:
                boost_val = min(0.76, max(0.60, float(diff_mag[y, x]) / 32.0))
                if out[best_k, y, x] < boost_val:
                    out[:, y, x] *= 1.0 - boost_val
                    out[best_k, y, x] = boost_val
                    s = float(out[:, y, x].sum())
                    if s > 1e-6:
                        out[:, y, x] /= s

    # Bridge 8-connected diagonal saddle corners for all non-background colors so 4x bicubic
    # never pinches diagonal baleen grooves or antennas into disconnected droplets!
    for k in range(K):
        out[k] = _bridge_diagonal_saddles(out[k], thresh=0.35)

    s_all = out.sum(axis=0, keepdims=True)
    out = np.divide(out, np.maximum(s_all, 1e-6))
    return out


def _clean_outer_border_blotches(
    fg_label_4x: np.ndarray,
    fg_4x: np.ndarray,
    palette_bgr: np.ndarray,
    dom_k: int,
    K: int,
) -> np.ndarray:
    """Remove low-contrast JPEG-undershoot blotches and thin 1px border ribbons that touch the
    outer silhouette border of a thick body, while preserving compact thick fins and high-contrast eyes.
    """
    cleaned = fg_label_4x.copy()
    outer_ring_4x = fg_4x & (
        cv2.dilate((~fg_4x).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    )
    dist_4x = ndi.distance_transform_edt(fg_4x)
    near_thick_4x = (
        cv2.dilate((dist_4x > 9.0).astype(np.uint8), np.ones((19, 19), np.uint8)) > 0
    )

    kern5 = np.ones((5, 5), np.uint8)
    for k in range(K):
        if k == dom_k:
            continue
        mask_k = ((cleaned == k) & fg_4x).astype(np.uint8)
        n_cc, cc_lab, stats, _ = cv2.connectedComponentsWithStats(mask_k, connectivity=8)
        for cid in range(1, n_cc):
            area = stats[cid, cv2.CC_STAT_AREA]
            if area >= 900:
                continue
            comp = cc_lab == cid
            if np.any(comp & outer_ring_4x) and np.any(comp & near_thick_4x):
                # Measure maximum internal thickness of this component at 4x
                # A real fin/flipper has max_thick >= 8.0 at 4x (>= 2.0px at 1x);
                # A JPEG border ribbon/blotch has max_thick < 6.5 at 4x (< 1.6px at 1x).
                x0 = stats[cid, cv2.CC_STAT_LEFT]
                y0 = stats[cid, cv2.CC_STAT_TOP]
                cw = stats[cid, cv2.CC_STAT_WIDTH]
                ch = stats[cid, cv2.CC_STAT_HEIGHT]
                sub_comp = comp[y0 : y0 + ch, x0 : x0 + cw]
                max_thick = float(ndi.distance_transform_edt(sub_comp).max()) if np.any(sub_comp) else 0.0
                if max_thick >= 6.8 and area >= 180:
                    continue

                dil = (cv2.dilate(comp.astype(np.uint8), kern5) > 0) & fg_4x & (~comp)
                nb_labels = cleaned[dil]
                if len(nb_labels) > 0:
                    vals, counts = np.unique(nb_labels, return_counts=True)
                    nb_k = int(vals[np.argmax(counts)])
                else:
                    nb_k = dom_k
                c_diff = float(np.linalg.norm(palette_bgr[k] - palette_bgr[nb_k]))
                if c_diff < 52.0 or area < 32:
                    cleaned[comp] = nb_k
    return cleaned


def trace_color_region(
    crop_bgr: np.ndarray,
    crop_bg_mask: np.ndarray,
    bg_color: np.ndarray,
    scale: int = SCALE,
    max_k: int = 12,
    merge_de: float = 12.0,
    fit_tol_1x: float = 0.52,
) -> tuple[list[str], np.ndarray]:
    """Trace a single flat-color region into clean, smooth hierarchical SVG <path> elements."""
    den = denoise_1x(crop_bgr)
    fg_mask = ~crop_bg_mask

    palette_bgr, pure_mask = extract_pure_palette(
        den, fg_mask, bg_color, max_k=max_k, merge_de=merge_de
    )
    K = len(palette_bgr)
    if K == 0:
        return [], palette_bgr

    alpha = unmix_pixels(crop_bgr, crop_bg_mask, pure_mask, palette_bgr, bg_color)
    alpha = _boost_thin_strokes(alpha, crop_bg_mask)

    h, w = crop_bgr.shape[:2]
    H4, W4 = h * scale, w * scale

    # 1. Total foreground coverage -> smooth, T-junction-free outer silhouette at 4x
    fg_1x = np.clip(1.0 - alpha[K], 0.0, 1.0)
    fg_up = cv2.resize(fg_1x, (W4, H4), interpolation=cv2.INTER_CUBIC)
    fg_up = cv2.GaussianBlur(fg_up, (0, 0), 1.55)
    fg_4x = fg_up > 0.50

    n_cc, cc_lab, cc_stats, _ = cv2.connectedComponentsWithStats(
        fg_4x.astype(np.uint8), connectivity=8
    )
    for cid in range(1, n_cc):
        if cc_stats[cid, cv2.CC_STAT_AREA] < 36:
            fg_4x[cc_lab == cid] = False

    if not np.any(fg_4x):
        return [], palette_bgr

    # 2. Upscale foreground color channels and apply 4x DoG translation-invariant interior line detector
    fg_alpha_4x = np.zeros((K, H4, W4), dtype=np.float32)
    for k in range(K):
        up = cv2.resize(alpha[k], (W4, H4), interpolation=cv2.INTER_CUBIC)
        up = cv2.GaussianBlur(up, (0, 0), 1.05)
        fg_alpha_4x[k] = np.clip(up, 0.0, 1.0)

    fg_alpha_4x = _apply_4x_dog_interior_details(
        fg_alpha_4x, crop_bgr, palette_bgr, fg_4x, scale=scale
    )

    fg_label_4x = fg_alpha_4x.argmax(axis=0).astype(np.int32)

    areas_init = np.array([((fg_label_4x == k) & fg_4x).sum() for k in range(K)])
    dom_k = int(np.argmax(areas_init))

    # Clean outer border JPEG-undershoot blotches (preserving high-contrast eyes)
    fg_label_4x = _clean_outer_border_blotches(fg_label_4x, fg_4x, palette_bgr, dom_k, K)

    areas = np.array([((fg_label_4x == k) & fg_4x).sum() for k in range(K)])
    order = [int(k) for k in np.argsort(-areas) if areas[k] >= 12]

    paths: list[str] = []
    kern_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for rank, k in enumerate(order):
        mask_k = (fg_label_4x == k) & fg_4x
        # Light morphological closing (5x5 at 4x = 1.25px at 1x) stitches any micro-gaps along thin grooves
        mask_k = (cv2.morphologyEx(mask_k.astype(np.uint8), cv2.MORPH_CLOSE, kern_close) > 0) & fg_4x

        if rank < len(order) - 1:
            higher_ids = order[rank + 1 :]
            higher_fg = np.isin(fg_label_4x, higher_ids) & fg_4x
            bridged = (cv2.dilate(mask_k.astype(np.uint8), kern_bridge) > 0) & higher_fg
            raw_layer = (mask_k | bridged).astype(np.float32)
            smooth_layer = (cv2.GaussianBlur(raw_layer, (0, 0), 1.4) > 0.48) & fg_4x
            layer_mask = smooth_layer | mask_k
        else:
            layer_mask = mask_k

        mask_u8 = (layer_mask.astype(np.uint8)) * 255
        d_str = mask_to_svg_path_d(
            mask_u8,
            scale=float(scale),
            min_area_4x=14.0,
            fit_tol_1x=fit_tol_1x,
            corner_deg=56.0,
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
