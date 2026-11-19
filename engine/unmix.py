"""Stage 3: Full 2-Color Pixel Unmixing (Sub-pixel Coverage Maps & Outer-Edge Halo Prevention)."""

from __future__ import annotations
import cv2
import numpy as np
from scipy import ndimage as ndi
from .color_separation import bgr_to_lab_f32


def unmix_pixels(
    img_bgr: np.ndarray,
    bg_mask: np.ndarray,
    pure_mask: np.ndarray,
    palette_bgr: np.ndarray,
    bg_color: np.ndarray,
) -> np.ndarray:
    """Compute continuous sub-pixel coverage map alpha of shape (K+1, H, W).

    Channels 0..K-1 correspond to palette_bgr colors; channel K corresponds to background.
    - Outer boundary pixels adjacent to a solid interior region unmix strictly between their
      nearest pure interior color and background (preventing a pale belly color from matching
      the anti-aliased edge of a dark back color!).
    - Interior transition pixels and thin appendages unmix between locally present palette pairs.
    """
    h, w = img_bgr.shape[:2]
    K = len(palette_bgr)
    if K == 0:
        alpha = np.zeros((1, h, w), dtype=np.float32)
        alpha[0] = 1.0
        return alpha

    img_f = img_bgr.astype(np.float32)
    bg_f = bg_color.astype(np.float32)
    all_c = np.vstack([palette_bgr.astype(np.float32), bg_f[None, :]])  # (K+1, 3)
    alpha = np.zeros((K + 1, h, w), dtype=np.float32)

    # 1. Hard anchor: pure foreground pixels -> nearest pure palette color in CIELAB
    lab = bgr_to_lab_f32(img_bgr)
    pal_lab = bgr_to_lab_f32(palette_bgr.reshape(1, -1, 3))[0]  # (K, 3)

    hard_label = np.full((h, w), -1, dtype=np.int32)
    py, px = np.where(pure_mask)
    if len(py) > 0:
        d_lab = np.linalg.norm(lab[py, px][:, None, :] - pal_lab[None, :, :], axis=2)
        p_assign = d_lab.argmin(axis=1)
        hard_label[py, px] = p_assign
        alpha[p_assign, py, px] = 1.0

    # 2. Hard anchor: background pixels strictly >1px away from foreground
    fg_mask = ~bg_mask
    fg_dil = cv2.dilate(fg_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    far_bg = bg_mask & (~fg_dil)
    hard_label[far_bg] = K
    alpha[K, far_bg] = 1.0

    # Distance to pure interior pixels and nearest pure pixel coordinates
    if np.any(pure_mask):
        dist_to_pure, (nearest_py, nearest_px) = ndi.distance_transform_edt(
            ~pure_mask, return_indices=True
        )
        nearest_pure_color = hard_label[nearest_py, nearest_px]
    else:
        dist_to_pure = np.full((h, w), 999.0, dtype=np.float32)
        nearest_pure_color = np.zeros((h, w), dtype=np.int32)

    # Identify outer boundary pixels that border background AND are close to a pure interior core
    # (<= 2.6px from pure_mask). These are outer edges of solid bodies (like a whale back/belly),
    # NOT long thin appendages (like shrimp antennas where dist_to_pure > 3).
    near_bg_1px = cv2.dilate(bg_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    solid_outer_edge = (hard_label < 0) & near_bg_1px & (dist_to_pure <= 2.6)

    # Check if pixel actually looks like a blend of nearest_pure_color and bg_color
    # (or if it's a thin dark outline/eye right on the border)
    sy, sx = np.where(solid_outer_edge)
    if len(sy) > 0:
        P_s = img_f[sy, sx]
        k_ref = nearest_pure_color[sy, sx]
        c_ref = all_c[k_ref]
        diff_bg = c_ref - bg_f[None, :]
        denom = np.sum(diff_bg * diff_bg, axis=1)
        t_bg = np.clip(np.sum((P_s - bg_f[None, :]) * diff_bg, axis=1) / np.maximum(denom, 1e-3), 0.0, 1.0)
        recon_bg = bg_f[None, :] + t_bg[:, None] * diff_bg
        err_bg = np.linalg.norm(P_s - recon_bg, axis=1)

        # If well-explained as anti-aliased edge of nearest interior color + background (err < 22),
        # lock it directly to (k_ref, bg)! This makes it IMPOSSIBLE for a pale belly color to
        # appear as confetti on a dark back's anti-aliased edge!
        good_outer = err_bg < 22.0
        gy, gx = sy[good_outer], sx[good_outer]
        gk = k_ref[good_outer]
        gt = t_bg[good_outer]
        # Suppress faint outer JPEG ringing (< 0.25 coverage)
        gt = np.where(gt < 0.25, 0.0, gt)
        alpha[gk, gy, gx] = gt
        alpha[K, gy, gx] = 1.0 - gt
        hard_label[gy, gx] = K  # mark as resolved

    # 3. Local spatial presence maps for remaining unresolved pixels (interior transitions & thin strokes)
    # Seed local_w ONLY from pure interior pixels + rescued thin-feature pixels that are NOT on
    # resolved solid outer edges.
    local_w = np.zeros((K + 1, h, w), dtype=np.float32)
    unresolved_fg = fg_mask & (~solid_outer_edge)
    for k in range(K):
        direct_k = unresolved_fg & (np.linalg.norm(img_f - all_c[k][None, None, :], axis=2) < 22.0)
        m = ((pure_mask & (hard_label == k)) | direct_k).astype(np.float32)
        local_w[k] = cv2.GaussianBlur(m, (7, 7), 1.5)
    local_w[K] = cv2.GaussianBlur(bg_mask.astype(np.float32), (5, 5), 1.1)

    near_bg_2px = cv2.dilate(bg_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    # 4. Vectorized 2-color unmixing for remaining boundary/interior/thin pixels
    todo = hard_label < 0
    ty, tx = np.where(todo)
    if len(ty) == 0:
        return alpha

    P = img_f[ty, tx]  # (N, 3)
    N = len(ty)
    is_near_bg = near_bg_2px[ty, tx]

    pairs = [(i, j) for i in range(K + 1) for j in range(i + 1, K + 1)]

    best_cost = np.full(N, 1e9, dtype=np.float32)
    best_i = np.full(N, K, dtype=np.int32)
    best_j = np.full(N, K, dtype=np.int32)
    best_t = np.zeros(N, dtype=np.float32)

    lw_todo = local_w[:, ty, tx]  # (K+1, N)
    max_fg_lw = lw_todo[:K].max(axis=0)

    for i, j in pairs:
        ci = all_c[i]
        cj = all_c[j]
        diff = ci - cj
        denom = float(np.dot(diff, diff))
        if denom < 1e-3:
            continue

        t = np.clip(((P - cj[None, :]) @ diff) / denom, 0.0, 1.0)
        recon = cj[None, :] + t[:, None] * diff[None, :]
        err = np.linalg.norm(P - recon, axis=1)

        has_i = lw_todo[i] > 0.02
        has_j = lw_todo[j] > 0.02
        pen = np.where(has_i & has_j, 0.0, np.where(has_i | has_j, 10.0, 26.0)).astype(np.float32)

        if j == K:
            # Disallow background (K) from mixing into strictly deep interior pixels
            pen = np.where(is_near_bg, pen, pen + 90.0)
            dist_i_bg = float(np.sqrt(denom))
            if dist_i_bg < 90.0:
                not_dom = lw_todo[i] < (max_fg_lw * 0.65)
                pale_pen = float((90.0 - dist_i_bg) * 0.25)
                pen = np.where(not_dom, pen + pale_pen, pen)

        cost = err + pen

        better = cost < best_cost
        if np.any(better):
            best_cost[better] = cost[better]
            best_i[better] = i
            best_j[better] = j
            best_t[better] = t[better]

    alpha[best_i, ty, tx] += best_t
    alpha[best_j, ty, tx] += 1.0 - best_t

    s = alpha.sum(axis=0, keepdims=True)
    alpha = np.divide(alpha, np.maximum(s, 1e-6))
    return alpha
