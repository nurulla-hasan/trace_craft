"""TraceCraft Vectorization Engine — v2 (trace_color9 + regularize_labels + vtracer).

Adapted from `new/vectorizer_pipeline.py` and `new/make_sheet_svg.py` for:
- Cross-platform (Windows / Linux / Docker) & thread-safe temp files (no hardcoded /tmp)
- Unicode-safe image loading on Windows
- Automatic single-image OR multi-icon sheet region splitting (no hardcoded 10x5 grid)
- Auto / Color / Binary modes + optional transparent background (`include_bg`)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import re
import tempfile
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import vtracer

SCALE = 4
MAX_IMAGE_PIXELS = 36_000_000  # 36 Megapixels safeguard against decompression bombs


def load_image(path: str) -> np.ndarray:
    """Load an image as BGR uint8 (compositing alpha over white if present)."""
    data = np.fromfile(path, dtype=np.uint8)
    raw = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size > 0 else None
    if raw is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # Guard against memory exhaustion / decompression bombs
    h_raw, w_raw = raw.shape[:2]
    total_px = h_raw * w_raw
    if total_px > MAX_IMAGE_PIXELS:
        scale_down = (MAX_IMAGE_PIXELS / total_px) ** 0.5
        new_w, new_h = max(1, int(w_raw * scale_down)), max(1, int(h_raw * scale_down))
        raw = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.shape[2] == 4:
        alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
        bgr = raw[:, :, :3].astype(np.float32)
        comp = bgr * alpha + 255.0 * (1.0 - alpha)
        return np.clip(comp, 0, 255).astype(np.uint8)
    return raw


def detect_mode(img_bgr: np.ndarray) -> str:
    """Auto-detect whether an image is binary/outline art or flat-color illustration."""
    diff_bg = np.abs(img_bgr.astype(np.int16) - 255).max(axis=2)
    fg = diff_bg > 20
    if not np.any(fg):
        return "binary"
    fg_px = img_bgr[fg]
    chroma = fg_px.max(axis=1).astype(np.int16) - fg_px.min(axis=1).astype(np.int16)
    colorful_ratio = float((chroma > 22).mean())
    return "binary" if colorful_ratio < 0.02 else "color"


def _inner_svg(svg: str) -> str:
    start = svg.index(">", svg.index("<svg")) + 1
    end = svg.rindex("</svg>")
    return svg[start:end]


def _fix_svg(svg: str, w: int, h: int) -> str:
    return re.sub(
        r'width="[\d.]+"\s+height="[\d.]+"',
        f'width="{w}" height="{h}" viewBox="0 0 {w * SCALE} {h * SCALE}"',
        svg,
        count=1,
    )


def regularize_labels(
    lab_s: np.ndarray,
    mask: np.ndarray,
    palette: np.ndarray,
    el,
    rim_px: int = 10,
    hue_tol: float = 35,
    min_chroma: float = 15,
) -> np.ndarray:
    """Make the colour map look designed rather than scanned:
    (a) every thin free-standing structure (antenna, tentacle, spine) gets ONE colour (its dominant one)
    (b) the 1-2px band hugging the silhouette is merged into the interior colour when it has the same hue
        (neutral outlines - black/white/grey - are real artwork and are left alone).
    """
    K = len(palette)
    if K <= 1:
        return lab_s

    pal_L = (
        cv2.cvtColor(palette.reshape(1, K, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
        .reshape(K, 3)[:, 0]
        .astype(np.float32)
    )
    m8 = mask.astype(np.uint8)
    opened = cv2.morphologyEx(m8, cv2.MORPH_OPEN, el(5))
    thin = (m8 > 0) & ~(cv2.dilate(opened, el(2)) > 0)
    nth, cth, st, _ = cv2.connectedComponentsWithStats(thin.astype(np.uint8), connectivity=8)
    for i in range(1, nth):
        if st[i, 4] < 40:
            continue
        mm = cth == i
        cnt = np.bincount(lab_s[mm], minlength=K)
        cand_l = np.where(cnt >= 0.15 * cnt.sum())[0]
        if len(cand_l) > 0:
            # a 1px line's pale pixels are anti-alias artefacts; its real colour is the darkest one it really uses
            lab_s[mm] = cand_l[np.argmin(pal_L[cand_l])]

    # (b) rim merge
    d_in = ndi.distance_transform_edt(mask)
    deep = mask & (d_in > rim_px) & ~thin
    if deep.any():
        _, (ky, kx) = ndi.distance_transform_edt(~deep, return_indices=True)
        cand = lab_s[ky, kx]
        labc = (
            cv2.cvtColor(palette.reshape(1, K, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
            .reshape(K, 3)
            .astype(np.float32)
        )
        a_ = labc[:, 1] - 128
        b_ = labc[:, 2] - 128
        hue = np.degrees(np.arctan2(b_, a_))
        chroma = np.hypot(a_, b_)
        dh = np.abs(hue[:, None] - hue[None])
        dh = np.minimum(dh, 360 - dh)
        ok = (dh < hue_tol) & (chroma[:, None] > min_chroma) & (chroma[None] > min_chroma)
        band = mask & (d_in <= rim_px) & ~thin
        lab_s = np.where(band & ok[lab_s, cand], cand, lab_s)
    return lab_s


def trace_color9(
    crop_bgr: np.ndarray,
    k: int = 12,
    merge: float = 9.0,
    core_r: int = 2,
    sigma: float = 1.8,
    speckle: int = 16,
    layer_diff: int = 10,
    cp: int = 8,
    prec: int = 1,
    glow_px: int = 14,
    pale_ratio: float = 0.55,
    ghost_tint: int = 55,
    ghost_half: int = 14,
) -> tuple[str, tuple[int, int], np.ndarray]:
    """Clean flat-colour trace (trace_color9 from new/vectorizer_pipeline.py).

    1. Silhouette from a soft alpha (un-mixed against the nearest solid colour), blurred + thresholded at 0.5
    2. Palette (Lab k-means) from solid interior only; edge pixels inherit the label of the nearest interior pixel
    3. Every label map is Gaussian-smoothed then arg-maxed -> smooth region boundaries, no speckle
    4. Label regularization + vtracer spline tracing
    """
    pad = 8
    crop = cv2.copyMakeBorder(
        crop_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    up = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    den = cv2.bilateralFilter(up, 9, 25, 9)
    diff = np.abs(den.astype(int) - 255).max(axis=2)

    # Adaptive background tolerance: dark/medium icons absorb the glow band next to the edge;
    # very light icons (beluga, pale shells) need a tight tolerance or they would be eaten
    med = float(np.median(diff[diff > 14])) if (diff > 14).any() else 0.0
    _, lab_cc = cv2.connectedComponents((diff < 14).astype(np.uint8))
    border = set(
        np.unique(np.concatenate([lab_cc[0], lab_cc[-1], lab_cc[:, 0], lab_cc[:, -1]]))
    ) - {0}
    bg = np.isin(lab_cc, list(border))

    if med >= 60:
        _, lab2 = cv2.connectedComponents((diff < 28).astype(np.uint8))
        b2 = set(
            np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]]))
        ) - {0}
        wide = np.isin(lab2, list(b2))
        bg = bg | (wide & (ndi.distance_transform_edt(~bg) <= glow_px))

        # Enclosed near-white gaps (between legs, bucket handles, claw slits...) are background
        enc = ((diff < 48) & ~bg).astype(np.uint8)
        _, ecc = cv2.connectedComponents(enc, connectivity=4)
        dtm = ndi.distance_transform_edt(enc)
        for i, sl in enumerate(ndi.find_objects(ecc), 1):
            if sl is None:
                continue
            mm = ecc[sl] == i
            if np.median(diff[sl][mm]) < ghost_tint and dtm[sl][mm].max() < ghost_half:
                bg[sl] |= mm

    fg0 = cv2.morphologyEx((~bg).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if not np.any(fg0):
        empty_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{crop.shape[1]}" height="{crop.shape[0]}" viewBox="0 0 {crop.shape[1]*SCALE} {crop.shape[0]*SCALE}"></svg>'
        return empty_svg, (crop.shape[1], crop.shape[0]), np.full_like(up, 255)

    el = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    # Glow removal: pale band thinner than ~4px along outline is anti-alias/glow fringe
    D = np.abs(den.astype(int) - 255).max(axis=2).astype(np.uint8)
    Dm = cv2.dilate(np.where(fg0 > 0, D, 0).astype(np.uint8), el(10))
    pale = ((fg0 > 0) & (D < pale_ratio * Dm)).astype(np.uint8)
    wide = cv2.morphologyEx(pale, cv2.MORPH_OPEN, el(8)) > 0
    near_bg = ndi.distance_transform_edt(fg0 > 0) <= 16
    glow = (pale > 0) & ~wide & near_bg
    fg0 = cv2.morphologyEx(((fg0 > 0) & ~glow).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    if not np.any(fg0):
        empty_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{crop.shape[1]}" height="{crop.shape[0]}" viewBox="0 0 {crop.shape[1]*SCALE} {crop.shape[0]*SCALE}"></svg>'
        return empty_svg, (crop.shape[1], crop.shape[0]), np.full_like(up, 255)

    core = cv2.erode(fg0, el(core_r)) > 0
    if int(core.sum()) < 200:
        core = fg0 > 0
    dcore = ndi.distance_transform_edt(~core)
    core |= (fg0 > 0) & (dcore > 6)  # thin parts (antennae etc.) keep their own colour
    _, (iy, ix) = ndi.distance_transform_edt(~core, return_indices=True)

    # Soft alpha: how far is each pixel from white, relative to the nearest solid colour
    cref = den[iy, ix].astype(np.float32)
    v = 255.0 - den.astype(np.float32)
    u = 255.0 - cref
    uu = (u * u).sum(2)
    alpha = np.clip((v * u).sum(2) / (uu + 1e-6), 0, 1)
    alpha[uu < 900] = 1.0
    alpha[core] = 1.0
    alpha[fg0 == 0] = 0.0
    mask = cv2.GaussianBlur(alpha, (0, 0), 1.5) > 0.5
    mask &= cv2.dilate(fg0, el(2)) > 0

    if not np.any(mask):
        empty_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{crop.shape[1]}" height="{crop.shape[0]}" viewBox="0 0 {crop.shape[1]*SCALE} {crop.shape[0]*SCALE}"></svg>'
        return empty_svg, (crop.shape[1], crop.shape[0]), np.full_like(up, 255)

    # Palette from interior pixels, Lab space, merge near-duplicates
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB).astype(np.float32)
    cl = lab[core]
    kk = min(k, max(1, min(len(cl), max(2, len(cl) // 400))))
    if len(cl) == 0:
        empty_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{crop.shape[1]}" height="{crop.shape[0]}" viewBox="0 0 {crop.shape[1]*SCALE} {crop.shape[0]*SCALE}"></svg>'
        return empty_svg, (crop.shape[1], crop.shape[0]), np.full_like(up, 255)

    samp = cl[np.random.RandomState(0).choice(len(cl), min(len(cl), 40000), replace=False)]
    kk = min(kk, len(samp))
    _, _, C = cv2.kmeans(
        samp,
        kk,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.3),
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    C_list = list(C)
    changed = True
    while changed and len(C_list) > 1:
        changed = False
        for a in range(len(C_list)):
            for b in range(a + 1, len(C_list)):
                if np.linalg.norm(C_list[a] - C_list[b]) < merge:
                    C_list[a] = (C_list[a] + C_list[b]) / 2
                    C_list.pop(b)
                    changed = True
                    break
            if changed:
                break
    C_arr = np.array(C_list, np.float32)

    ys, xs = np.where(core)
    d = ((lab[ys, xs][:, None, :] - C_arr[None]) ** 2).sum(2)
    lab_core = np.zeros(core.shape, np.int32)
    lab_core[ys, xs] = d.argmin(1)
    labels = lab_core[iy, ix]  # every pixel inherits nearest interior label

    K = len(C_arr)
    palette = np.zeros((K, 3), np.uint8)
    for l in range(K):
        sel = core & (labels == l)
        palette[l] = den[sel].mean(0) if sel.any() else 255

    # Smooth label boundaries
    probs = np.stack(
        [cv2.GaussianBlur(((labels == l) & mask).astype(np.float32), (0, 0), sigma) for l in range(K)]
    )
    lab_s = probs.argmax(0)
    lab_s = regularize_labels(lab_s, mask, palette, el)
    out = palette[lab_s]
    out[~mask] = 255
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    with tempfile.TemporaryDirectory() as tmpdir:
        in_png = os.path.join(tmpdir, "in.png")
        out_svg = os.path.join(tmpdir, "out.svg")
        Image.fromarray(np.dstack([rgb, (mask * 255).astype(np.uint8)])).save(in_png)
        vtracer.convert_image_to_svg_py(
            in_png,
            out_svg,
            colormode="color",
            hierarchical="stacked",
            mode="spline",
            filter_speckle=speckle,
            color_precision=cp,
            layer_difference=layer_diff,
            corner_threshold=60,
            length_threshold=6.0,
            splice_threshold=45,
            path_precision=prec,
        )
        with open(out_svg, "r", encoding="utf-8") as f:
            svg = f.read()

    return _fix_svg(svg, crop.shape[1], crop.shape[0]), (crop.shape[1], crop.shape[0]), out


def split_regions_masked(
    img_bgr: np.ndarray,
    bg_thresh: int = 12,
    min_region_area: int = 250,
    merge_dist: int = 14,
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Split a multi-icon sheet into independent masked crops (or return 1 region for a single logo/icon).

    Every pixel belonging to a neighboring icon inside the crop bbox is painted white so overlapping
    bounding boxes never pollute each other.
    """
    h, w = img_bgr.shape[:2]
    diff = np.abs(img_bgr.astype(np.int16) - 255).max(axis=2)
    fg = (diff > bg_thresh).astype(np.uint8)

    n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n_cc <= 1:
        return []

    major_ids = [i for i in range(1, n_cc) if stats[i, cv2.CC_STAT_AREA] >= min_region_area]
    minor_ids = [i for i in range(1, n_cc) if 4 <= stats[i, cv2.CC_STAT_AREA] < min_region_area]

    # Single cohesive illustration / icon
    if len(major_ids) <= 1:
        ys, xs = np.where(fg > 0)
        if len(xs) == 0:
            return []
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        return [((x0, y0, x1, y1), img_bgr[y0:y1, x0:x1].copy())]

    # Assign minor components (detached eyes, dots, sparkles) to nearest major component
    major_mask = np.isin(lab, major_ids)
    dist_to_major, nearest_idx = ndi.distance_transform_edt(~major_mask, return_indices=True)

    groups: dict[int, list[int]] = {mid: [mid] for mid in major_ids}
    for sid in minor_ids:
        ys, xs = np.where(lab == sid)
        if float(dist_to_major[ys, xs].min()) <= merge_dist:
            best = int(np.argmin(dist_to_major[ys, xs]))
            ny = nearest_idx[0, ys[best], xs[best]]
            nx = nearest_idx[1, ys[best], xs[best]]
            parent = int(lab[ny, nx])
            if parent in groups:
                groups[parent].append(sid)
            else:
                groups[sid] = [sid]
        else:
            groups[sid] = [sid]

    regions: list[tuple[float, float, tuple[int, int, int, int], np.ndarray]] = []
    kern_own = np.ones((7, 7), np.uint8)
    for member_ids in groups.values():
        own_fg = np.isin(lab, member_ids)
        ys, xs = np.where(own_fg)
        if len(xs) == 0:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        own_slice = own_fg[y0:y1, x0:x1]
        own_dil = cv2.dilate(own_slice.astype(np.uint8), kern_own) > 0
        other_fg = (fg[y0:y1, x0:x1] > 0) & (~own_slice)
        crop = img_bgr[y0:y1, x0:x1].copy()
        crop[other_fg | (~own_dil)] = 255
        regions.append(((y0 + y1) / 2.0, (x0 + x1) / 2.0, (x0, y0, x1, y1), crop))

    regions.sort(key=lambda r: (round(r[0] / 60.0), r[1]))
    return [(bbox, crop) for _, _, bbox, crop in regions]


def trace_binary_sheet(img_bgr: np.ndarray, include_bg: bool = True) -> str:
    """Black outline sheet / line art: trace in binary spline mode via vtracer.
    Uses dynamic scale to prevent RAM exhaustion on large sheets.
    """
    H, W = img_bgr.shape[:2]
    max_dim = max(H, W)
    dyn_scale = SCALE if max_dim <= 1600 else (2 if max_dim <= 3200 else 1)

    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(g, None, fx=dyn_scale, fy=dyn_scale, interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 1.2)
    bw = np.where(up < 130, 0, 255).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        in_png = os.path.join(tmpdir, "sheet.png")
        out_svg = os.path.join(tmpdir, "sheet.svg")
        Image.fromarray(np.dstack([bw, bw, bw, np.full_like(bw, 255)])).save(in_png)
        vtracer.convert_image_to_svg_py(
            in_png,
            out_svg,
            colormode="binary",
            mode="spline",
            filter_speckle=6,
            corner_threshold=55,
            length_threshold=6.0,
            splice_threshold=40,
            path_precision=2,
        )
        with open(out_svg, "r", encoding="utf-8") as f:
            body = _inner_svg(f.read())

    bg_rect = (
        f'<rect id="background" width="{W * dyn_scale}" height="{H * dyn_scale}" fill="#ffffff"/>\n'
        if include_bg
        else ""
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W * dyn_scale} {H * dyn_scale}">\n'
        f"{bg_rect}"
        f'<g id="artwork">{body}</g>\n'
        f"</svg>\n"
    )


def _trace_single_region(
    item: tuple[int, tuple[tuple[int, int, int, int], np.ndarray]]
) -> tuple[int, str]:
    idx, ((x0, y0, x1, y1), crop) = item
    svg_part, _, _ = trace_color9(crop)
    body = _inner_svg(svg_part).strip()
    if not body:
        return idx, ""
    part = (
        f'  <g id="icon-{idx:02d}" transform="translate({x0 - 8},{y0 - 8}) scale({1 / SCALE})">'
        f"{body}</g>"
    )
    return idx, part


def vectorize_image(
    input_path: str,
    output_svg_path: str,
    mode: str = "auto",
    include_bg: bool = True,
) -> dict:
    """End-to-end vectorization entry point for any raster image (single icon or multi-icon sheet)."""
    img_bgr = load_image(input_path)
    H, W = img_bgr.shape[:2]
    chosen_mode = detect_mode(img_bgr) if mode == "auto" else mode

    if chosen_mode == "binary":
        svg = trace_binary_sheet(img_bgr, include_bg=include_bg)
        with open(output_svg_path, "w", encoding="utf-8") as f:
            f.write(svg)
        return {
            "mode": "binary",
            "width": W,
            "height": H,
            "regions": 1,
            "paths": svg.count("<path"),
            "bytes": len(svg.encode("utf-8")),
        }

    regions = split_regions_masked(img_bgr, bg_thresh=12)
    if not regions:
        # Fallback for empty/blank images
        bg_rect = (
            f'<rect id="background" width="{W}" height="{H}" fill="#ffffff"/>\n'
            if include_bg
            else ""
        )
        empty_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n{bg_rect}</svg>\n'
        with open(output_svg_path, "w", encoding="utf-8") as f:
            f.write(empty_svg)
        return {
            "mode": "color",
            "width": W,
            "height": H,
            "regions": 0,
            "paths": 0,
            "bytes": len(empty_svg.encode("utf-8")),
        }

    # Parallelize multi-icon tracing across available CPU hardware threads
    workers = min(8, max(1, os.cpu_count() or 4))
    if len(regions) > 1 and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_trace_single_region, enumerate(regions, 1)))
        results.sort(key=lambda x: x[0])
        parts = [elem for _, elem in results if elem]
    else:
        parts = []
        for item in enumerate(regions, 1):
            _, elem = _trace_single_region(item)
            if elem:
                parts.append(elem)

    bg_rect = (
        f'  <rect id="background" width="{W}" height="{H}" fill="#ffffff"/>\n'
        if include_bg
        else ""
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
        f"{bg_rect}"
        + "\n".join(parts)
        + "\n</svg>\n"
    )
    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write(svg)

    return {
        "mode": "color",
        "width": W,
        "height": H,
        "regions": len(parts),
        "paths": svg.count("<path"),
        "bytes": len(svg.encode("utf-8")),
    }
