"""Stage 2: 1x Pure-Pixel Color Separation, Spatial Blend-Color Rejection & Interior Line Rescue."""

from __future__ import annotations
import cv2
import numpy as np


def bgr_to_lab_f32(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 (or float32) image to perceptual CIELAB float32 (L:0..100, a/b:-128..127)."""
    if img_bgr.dtype != np.uint8:
        img_u8 = np.clip(img_bgr, 0, 255).astype(np.uint8)
    else:
        img_u8 = img_bgr
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] = lab[..., 0] * (100.0 / 255.0)
    lab[..., 1] = lab[..., 1] - 128.0
    lab[..., 2] = lab[..., 2] - 128.0
    return lab


def find_pure_pixels(
    img_bgr: np.ndarray,
    fg_mask: np.ndarray,
    grad_thresh: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify strictly pure interior foreground pixels at 1x resolution."""
    lab = bgr_to_lab_f32(img_bgr)

    pad_lab = np.pad(lab, ((1, 1), (1, 1), (0, 0)), mode="edge")
    d_up = np.linalg.norm(lab - pad_lab[:-2, 1:-1], axis=2)
    d_dn = np.linalg.norm(lab - pad_lab[2:, 1:-1], axis=2)
    d_lt = np.linalg.norm(lab - pad_lab[1:-1, :-2], axis=2)
    d_rt = np.linalg.norm(lab - pad_lab[1:-1, 2:], axis=2)
    max_nb_de = np.maximum(np.maximum(d_up, d_dn), np.maximum(d_lt, d_rt))

    # Erode by 2px at 1x where possible so JPEG edge-undershoot ring (1-2px inside outer border)
    # cannot pollute pure interior palette discovery!
    ell5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded2 = cv2.erode(fg_mask.astype(np.uint8), ell5) > 0
    pure = eroded2 & (max_nb_de < grad_thresh)

    if int(pure.sum()) < 25:
        cross3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        eroded1 = cv2.erode(fg_mask.astype(np.uint8), cross3) > 0
        pure = eroded1 & (max_nb_de < grad_thresh)
    if int(pure.sum()) < 20:
        pure = eroded1 & (max_nb_de < grad_thresh * 2.0)
    if int(pure.sum()) < 15:
        pure = eroded1
    if int(pure.sum()) < 10:
        pure = fg_mask.copy()

    return pure, lab


def _cluster_pure_lab(
    pure_lab: np.ndarray,
    max_k: int = 12,
    merge_de: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster pure pixels in CIELAB and merge perceptually close clusters (suppressing JPEG noise)."""
    n = len(pure_lab)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    k_init = min(max_k, max(1, n // 25))
    if k_init == 1:
        return pure_lab.mean(axis=0, keepdims=True).astype(np.float32), np.array([n], dtype=np.int32)

    rng = np.random.RandomState(42)
    sample = pure_lab[rng.choice(n, 20000, replace=False)] if n > 20000 else pure_lab

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(
        sample.astype(np.float32), k_init, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.ravel(), minlength=k_init).astype(np.float32)

    valid = counts > 0
    centers = centers[valid]
    counts = counts[valid]

    centers_list = [c for c in centers]
    counts_list = [float(w) for w in counts]
    while len(centers_list) > 1:
        best_d = 1e9
        best_pair = (-1, -1)
        for i in range(len(centers_list)):
            for j in range(i + 1, len(centers_list)):
                d = float(np.linalg.norm(centers_list[i] - centers_list[j]))
                if d < best_d:
                    best_d = d
                    best_pair = (i, j)
        if best_d >= merge_de:
            break
        i, j = best_pair
        wi, wj = counts_list[i], counts_list[j]
        centers_list[i] = (centers_list[i] * wi + centers_list[j] * wj) / (wi + wj)
        counts_list[i] = wi + wj
        centers_list.pop(j)
        counts_list.pop(j)

    C = np.array(centers_list, dtype=np.float32)
    dists = np.linalg.norm(pure_lab[:, None, :] - C[None, :, :], axis=2)
    nearest = dists.argmin(axis=1)
    exact_counts = np.bincount(nearest, minlength=len(C))

    min_px = max(6, int(0.004 * n))
    order = np.argsort(-exact_counts)
    kept: list[int] = []
    for idx in order:
        if exact_counts[idx] == 0:
            continue
        if exact_counts[idx] >= min_px or len(kept) == 0:
            kept.append(int(idx))
        else:
            min_de = min(float(np.linalg.norm(C[idx] - C[k])) for k in kept)
            if min_de > 22.0 and exact_counts[idx] >= 3:
                kept.append(int(idx))

    return C[kept], exact_counts[kept].astype(np.int32)


def _is_linear_blend(
    c_mid: np.ndarray,
    c_a: np.ndarray,
    c_b: np.ndarray,
    max_err: float = 16.0,
) -> bool:
    """Check if RGB color c_mid lies on the linear blend segment between c_a and c_b."""
    ab = c_a - c_b
    denom = float(np.dot(ab, ab))
    if denom < 100.0:
        return False
    t = float(np.dot(c_mid - c_b, ab)) / denom
    if not (0.06 < t < 0.94):
        return False
    proj = c_b + t * ab
    err = float(np.linalg.norm(c_mid - proj))
    return err < max_err


def _reject_blend_colors(
    img_bgr: np.ndarray,
    fg_mask: np.ndarray,
    pure_mask: np.ndarray,
    palette_bgr: np.ndarray,
    bg_color: np.ndarray,
) -> np.ndarray:
    """Reject candidate palette colors that are spatially sitting on the transition border
    between two regions (e.g., salmon rim between orange & white, or peach antenna step).

    Crucially preserves thin interior detail lines (like teal baleen grooves embedded INSIDE a
    light-blue whale belly) where pixels are surrounded by ONE region rather than sitting on
    the boundary between two regions!
    """
    pal = palette_bgr.astype(np.float32).copy()
    bg = bg_color.astype(np.float32)

    if len(pal) <= 1:
        return pal

    ell3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kern5 = np.ones((5, 5), dtype=np.uint8)
    fg_f = img_bgr.astype(np.float32)

    changed = True
    while changed and len(pal) > 1:
        changed = False
        all_c = np.vstack([pal, bg[None, :]])
        d = np.linalg.norm(fg_f[fg_mask][:, None, :] - all_c[None, :, :], axis=2)
        fg_lab = d.argmin(axis=1)
        label_map = np.full(fg_mask.shape, len(pal), dtype=np.int32)
        label_map[fg_mask] = fg_lab

        # Precompute 2px dilated neighborhood presence for each label (0..len(pal))
        dil_masks = [
            cv2.dilate((label_map == i).astype(np.uint8), kern5) > 0
            for i in range(len(pal) + 1)
        ]

        counts = np.array([(label_map == i).sum() for i in range(len(pal))])
        test_order = np.argsort(counts)

        for m in test_order:
            mask_m = (label_map == m).astype(np.uint8)
            total_m = int(mask_m.sum())
            if total_m == 0:
                pal = np.delete(pal, m, axis=0)
                changed = True
                break

            opened = cv2.morphologyEx(mask_m, cv2.MORPH_OPEN, ell3)
            solid_m = int(opened.sum())
            if solid_m >= 22 and (solid_m / max(1, total_m)) >= 0.36:
                continue

            pure_m = int((pure_mask & (label_map == m)).sum())
            if pure_m >= 45 and solid_m >= 14:
                continue

            m_bool = mask_m > 0
            candidates = [i for i in range(len(pal)) if i != m] + [len(pal)]
            is_blend = False
            for idx_a in range(len(candidates)):
                ia = candidates[idx_a]
                ca = all_c[ia]
                for idx_b in range(idx_a + 1, len(candidates)):
                    ib = candidates[idx_b]
                    cb = all_c[ib]
                    if ia < len(pal) and ib < len(pal):
                        if max(counts[ia], counts[ib]) < total_m:
                            continue
                    if not _is_linear_blend(pal[m], ca, cb, max_err=16.0):
                        continue

                    # Spatial check: is color m actually sitting on the border BETWEEN ia and ib
                    # (or touching background ib == len(pal))?
                    # If ib is background and m touches background -> outer rim or antenna step -> REJECT!
                    if ib == len(pal):
                        touch_bg = float((m_bool & dil_masks[len(pal)]).sum()) / total_m
                        if touch_bg > 0.25:
                            is_blend = True
                            break
                    # If both ia and ib are foreground colors, m is a boundary artifact ONLY if most of
                    # m's pixels lie near BOTH ia and ib (not embedded deep inside just one of them!)
                    both_near = float((m_bool & dil_masks[ia] & dil_masks[ib]).sum()) / total_m
                    if both_near > 0.45:
                        is_blend = True
                        break
                if is_blend:
                    break

            if is_blend:
                pal = np.delete(pal, m, axis=0)
                changed = True
                break

    return pal


def _min_segment_residual(
    pixels_bgr: np.ndarray,
    colors_bgr: np.ndarray,
) -> np.ndarray:
    """Compute minimum Euclidean RGB distance from each pixel to any segment [Ci, Cj] in colors_bgr."""
    n = len(pixels_bgr)
    m = len(colors_bgr)
    if n == 0 or m == 0:
        return np.zeros(n, dtype=np.float32)
    if m == 1:
        return np.linalg.norm(pixels_bgr - colors_bgr[0:1], axis=1)

    best_err = np.full(n, 1e9, dtype=np.float32)
    for i in range(m):
        ci = colors_bgr[i]
        d_i = np.linalg.norm(pixels_bgr - ci[None, :], axis=1)
        best_err = np.minimum(best_err, d_i)
        for j in range(i + 1, m):
            cj = colors_bgr[j]
            diff = ci - cj
            denom = float(np.dot(diff, diff))
            if denom < 1e-3:
                continue
            t = np.clip(((pixels_bgr - cj[None, :]) @ diff) / denom, 0.0, 1.0)
            recon = cj[None, :] + t[:, None] * diff[None, :]
            err = np.linalg.norm(pixels_bgr - recon, axis=1)
            best_err = np.minimum(best_err, err)
    return best_err


def _rescue_thin_feature_colors(
    img_bgr: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    pure_mask: np.ndarray,
    palette_bgr: np.ndarray,
    bg_color: np.ndarray,
    max_rescues: int = 4,
    residual_thresh: float = 24.0,
) -> np.ndarray:
    """Rescue true thin-feature colors:
    1. High-residual thin appendages & outlines (shrimp dark brown antennas/legs, clownfish black outline)
    2. Interior 1px detail lines/grooves & eyes embedded inside a solid region (whale baleen grooves,
       mouth creases, eyes, turtle shell lines, whale shark spots) — excluding the 2px outer border
       ring where JPEG Gibbs edge-undershoot lives!
    """
    from scipy import ndimage as ndi

    pal = palette_bgr.astype(np.float32).copy()
    bg = bg_color.astype(np.float32)
    fg_f = img_bgr.astype(np.float32)

    near_bg_2px = cv2.dilate(bg_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    fg_deep = fg_mask & (~near_bg_2px)

    # Local detail contrast: true interior lines/grooves/spots/eyes stand out from their own
    # 5x5 local median neighborhood (> 12 RGB units), whereas smooth JPEG block drift does not!
    med5 = cv2.medianBlur(img_bgr, 5).astype(np.float32)
    local_detail_diff = np.linalg.norm(fg_f - med5, axis=2)

    if np.any(pure_mask):
        dist_to_pure = ndi.distance_transform_edt(~pure_mask)
    else:
        dist_to_pure = np.full(fg_mask.shape, 999.0, dtype=np.float32)

    fg_appendage = fg_mask & near_bg_2px & (dist_to_pure > 1.8)
    fg_interior_detail = fg_deep & (local_detail_diff > 12.0)

    for _ in range(max_rescues):
        if len(pal) == 0:
            break

        res_map = np.zeros(fg_mask.shape, dtype=np.float32)

        ay, ax = np.where(fg_appendage)
        if len(ay) > 0:
            pal_with_bg = np.vstack([pal, bg[None, :]])
            res_map[ay, ax] = _min_segment_residual(fg_f[ay, ax], pal_with_bg)

        dy, dx = np.where(fg_interior_detail)
        if len(dy) > 0:
            d_single = np.linalg.norm(fg_f[dy, dx][:, None, :] - pal[None, :, :], axis=2).min(axis=1)
            res_map[dy, dx] = d_single

        dist_bg = np.linalg.norm(fg_f - bg[None, None, :], axis=2)
        unexplained = fg_mask & (res_map > residual_thresh) & (dist_bg > 28.0)

        uy, ux = np.where(unexplained)
        if len(uy) < 2:
            break

        u_px = fg_f[uy, ux]
        u_res = res_map[uy, ux]

        top_k = max(2, int(len(u_res) * 0.45))
        top_idx = np.argsort(-u_res)[:top_k]
        cand_px = u_px[top_idx]

        cand_lab = bgr_to_lab_f32(cand_px.reshape(1, -1, 3))[0]
        d_mat = np.linalg.norm(cand_lab[:, None, :] - cand_lab[None, :, :], axis=2)
        density = (d_mat < 15.0).sum(axis=1)
        seed_idx = int(np.argmax(density))
        in_cluster = d_mat[seed_idx] < 16.0
        cluster_px = cand_px[in_cluster]

        if len(cluster_px) < 2:
            break

        d_from_bg = np.linalg.norm(cluster_px - bg[None, :], axis=1)
        keep_n = max(1, (len(cluster_px) + 1) // 2)
        pure_est = np.median(cluster_px[np.argsort(-d_from_bg)[:keep_n]], axis=0).astype(np.float32)

        # Suppress all pixels close to pure_est from future rescue iterations
        near_est = np.linalg.norm(fg_f - pure_est[None, None, :], axis=2) < 26.0
        fg_appendage = fg_appendage & (~near_est)
        fg_interior_detail = fg_interior_detail & (~near_est)

        min_d = float(np.linalg.norm(pal - pure_est[None, :], axis=1).min())
        if min_d < 28.0:
            continue

        pal = np.vstack([pal, pure_est[None, :]])

    return pal


def _drop_near_bg_fringe_colors(
    img_bgr: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    palette_bgr: np.ndarray,
    bg_color: np.ndarray,
) -> np.ndarray:
    """Remove any near-white / near-background color that leaked into palette due to JPEG noise."""
    if len(palette_bgr) <= 1:
        return palette_bgr

    bg = bg_color.astype(np.float32)
    keep = [k for k in range(len(palette_bgr)) if float(np.linalg.norm(palette_bgr[k] - bg)) >= 28.0]
    if not keep:
        dists = np.linalg.norm(palette_bgr - bg[None, :], axis=1)
        return palette_bgr[int(np.argmax(dists)) : int(np.argmax(dists)) + 1]
    return palette_bgr[keep]


def extract_pure_palette(
    img_bgr: np.ndarray,
    fg_mask: np.ndarray,
    bg_color: np.ndarray,
    max_k: int = 12,
    merge_de: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Stage 2: Pure-pixel detection + Lab clustering + spatial blend rejection + detail rescue."""
    bg_mask = ~fg_mask
    pure_mask, lab = find_pure_pixels(img_bgr, fg_mask)
    pure_lab = lab[pure_mask]
    pure_bgr = img_bgr[pure_mask].astype(np.float32)

    centers_lab, _ = _cluster_pure_lab(pure_lab, max_k=max_k, merge_de=merge_de)
    if len(centers_lab) == 0:
        return np.zeros((0, 3), dtype=np.float32), pure_mask

    dists = np.linalg.norm(pure_lab[:, None, :] - centers_lab[None, :, :], axis=2)
    assign = dists.argmin(axis=1)
    palette_bgr = np.zeros((len(centers_lab), 3), dtype=np.float32)
    for k in range(len(centers_lab)):
        sel = assign == k
        if np.any(sel):
            palette_bgr[k] = np.median(pure_bgr[sel], axis=0)

    palette_bgr = _drop_near_bg_fringe_colors(img_bgr, fg_mask, bg_mask, palette_bgr, bg_color)
    palette_bgr = _reject_blend_colors(img_bgr, fg_mask, pure_mask, palette_bgr, bg_color)

    palette_bgr = _rescue_thin_feature_colors(
        img_bgr, fg_mask, bg_mask, pure_mask, palette_bgr, bg_color, max_rescues=4, residual_thresh=24.0
    )

    palette_bgr = _drop_near_bg_fringe_colors(img_bgr, fg_mask, bg_mask, palette_bgr, bg_color)
    palette_bgr = _reject_blend_colors(img_bgr, fg_mask, pure_mask, palette_bgr, bg_color)

    if len(palette_bgr) > 0:
        pal_lab = bgr_to_lab_f32(palette_bgr.reshape(1, -1, 3))[0]
        dists = np.linalg.norm(pure_lab[:, None, :] - pal_lab[None, :, :], axis=2)
        assign = dists.argmin(axis=1)
        for k in range(len(palette_bgr)):
            sel = (assign == k) & (dists[:, k] < 11.0)
            if int(sel.sum()) >= 10:
                palette_bgr[k] = np.median(pure_bgr[sel], axis=0)

    palette_bgr = _drop_near_bg_fringe_colors(img_bgr, fg_mask, bg_mask, palette_bgr, bg_color)
    return palette_bgr, pure_mask
