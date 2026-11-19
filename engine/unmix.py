"""Stage 3: Full 2-Color Pixel Unmixing with Strict Local Anti-Aliasing Physics."""

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

    Enforces two key physical laws of raster anti-aliasing:
      1. Outer skin of solid regions (within 2.5px of pure_mask and bordering background)
         can ONLY blend the nearest pure interior color with background (absorbing JPEG
         Gibbs edge-undershoot instead of turning it into dark border blotches!).
      2. Two foreground colors (Ci, Cj) can ONLY blend at pixels within ~2.5px of BOTH regions
         (preventing a distant dark back color from blending with a light belly color to erase
         interior teal baleen grooves!).
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

    if np.any(pure_mask):
        dist_to_pure, (nearest_py, nearest_px) = ndi.distance_transform_edt(
            ~pure_mask, return_indices=True
        )
        nearest_pure_color = hard_label[nearest_py, nearest_px]
    else:
        dist_to_pure = np.full((h, w), 999.0, dtype=np.float32)
        nearest_pure_color = np.zeros((h, w), dtype=np.int32)

    # 3. Outer skin of solid regions: lock to (nearest_pure_color, bg) — including JPEG Gibbs
    # edge-undershoot pixels (which dip up to ~38 RGB units darker than c_ref right inside the border!)
    near_bg_2px = cv2.dilate(bg_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    solid_outer_edge = (hard_label < 0) & near_bg_2px & (dist_to_pure <= 2.6)

    sy, sx = np.where(solid_outer_edge)
    if len(sy) > 0:
        P_s = img_f[sy, sx]
        k_ref = nearest_pure_color[sy, sx]
        c_ref = all_c[k_ref]
        diff_bg = c_ref - bg_f[None, :]
        denom = np.sum(diff_bg * diff_bg, axis=1)
        t_bg = np.clip(
            np.sum((P_s - bg_f[None, :]) * diff_bg, axis=1) / np.maximum(denom, 1e-3),
            0.0,
            1.0,
        )
        recon_bg = bg_f[None, :] + t_bg[:, None] * diff_bg
        err_bg = np.linalg.norm(P_s - recon_bg, axis=1)
        dist_cref = np.linalg.norm(P_s - c_ref, axis=1)

        # Match either smooth anti-aliased transition to bg (err_bg < 25) OR JPEG Gibbs undershoot
        # of the local solid color (dist_cref < 40)
        good_outer = (err_bg < 25.0) | (dist_cref < 40.0)
        gy, gx = sy[good_outer], sx[good_outer]
        gk = k_ref[good_outer]
        gt = t_bg[good_outer]
        gt = np.where(gt < 0.25, 0.0, gt)
        alpha[gk, gy, gx] = gt
        alpha[K, gy, gx] = 1.0 - gt
        hard_label[gy, gx] = K

    # 4. Local spatial presence maps for remaining interior / thin-stroke pixels
    local_w = np.zeros((K + 1, h, w), dtype=np.float32)
    unresolved_fg = fg_mask & (~solid_outer_edge)
    for k in range(K):
        direct_k = unresolved_fg & (np.linalg.norm(img_f - all_c[k][None, None, :], axis=2) < 20.0)
        m = ((pure_mask & (hard_label == k)) | direct_k).astype(np.float32)
        local_w[k] = cv2.GaussianBlur(m, (5, 5), 1.2)
    local_w[K] = cv2.GaussianBlur(bg_mask.astype(np.float32), (5, 5), 1.1)

    todo = hard_label < 0
    ty, tx = np.where(todo)
    if len(ty) == 0:
        return alpha

    P = img_f[ty, tx]  # (N, 3)
    N = len(ty)
    is_near_bg = near_bg_2px[ty, tx]

    best_cost = np.full(N, 1e9, dtype=np.float32)
    best_i = np.full(N, K, dtype=np.int32)
    best_j = np.full(N, K, dtype=np.int32)
    best_t = np.zeros(N, dtype=np.float32)

    lw_todo = local_w[:, ty, tx]  # (K+1, N)
    max_fg_lw = lw_todo[:K].max(axis=0)

    # First check direct single-color match to each foreground color k
    # Require strong local presence (or high contrast) so a dark shadow color from one part of an
    # icon (e.g. whale's dark back flipper) cannot teleport onto JPEG noise on the tail or back!
    for k in range(K):
        ck = all_c[k]
        err_single = np.linalg.norm(P - ck[None, :], axis=1)
        has_k = lw_todo[k] > 0.025
        pen_single = np.where(has_k, 0.0, 24.0).astype(np.float32)
        cost_single = err_single + pen_single
        better = cost_single < best_cost
        if np.any(better):
            best_cost[better] = cost_single[better]
            best_i[better] = k
            best_j[better] = k
            best_t[better] = 1.0

    # Then check 2-color blends (i, j)
    pairs = [(i, j) for i in range(K + 1) for j in range(i + 1, K + 1)]
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

        has_i = lw_todo[i] > 0.025
        has_j = lw_todo[j] > 0.025

        if j < K:
            # Two foreground colors can ONLY blend if BOTH are locally present at this boundary!
            # Prevents distant dark-blue back + light-blue belly from blending inside the belly
            # to swallow teal baleen lines!
            pen = np.where(has_i & has_j, 1.5, np.where(has_i | has_j, 24.0, 45.0)).astype(np.float32)
        else:
            # Blending with background (j == K)
            pen = np.where(has_i & has_j, 0.0, np.where(has_i | has_j, 10.0, 26.0)).astype(np.float32)
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
