"""Stage 1: Preprocessing, mode detection, background segmentation, and region splitting."""

from __future__ import annotations
import cv2
import numpy as np
from scipy import ndimage as ndi


def load_image(path: str) -> np.ndarray:
    """Load an image as BGR uint8 (compositing alpha over white if present)."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read image: {path}")
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


def denoise_1x(img_bgr: np.ndarray) -> np.ndarray:
    """Edge-preserving denoise at 1x resolution.

    Two-pass strategy:
    1. Non-local means (fastNlMeansDenoisingColored, h=6) – suppresses JPEG 8x8 block ringing
       without smearing sharp 1px edges. Slower but crucial for clean palette discovery.
    2. Light bilateral (d=5, sigmaColor=12) – smooths residual high-frequency salt in flat areas.
    """
    # fastNlMeansDenoising: h=6 is conservative enough to preserve thin strokes & eyes
    nlm = cv2.fastNlMeansDenoisingColored(img_bgr, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=15)
    return cv2.bilateralFilter(nlm, d=5, sigmaColor=12, sigmaSpace=5)


def extract_background_mask(
    img_bgr: np.ndarray, bg_tol: int = 14
) -> tuple[np.ndarray, np.ndarray]:
    """Detect uniform background via border flood-fill (keeps interior whites intact).

    Returns:
        bg_mask: bool array (H, W), True where background.
        bg_color: float32 BGR color (3,) of detected background.
    """
    h, w = img_bgr.shape[:2]
    border_px = np.concatenate(
        [img_bgr[0, :], img_bgr[-1, :], img_bgr[:, 0], img_bgr[:, -1]], axis=0
    )
    bg_color = np.median(border_px, axis=0).astype(np.float32)

    diff = np.abs(img_bgr.astype(np.float32) - bg_color[None, None, :]).max(axis=2)
    cand = (diff <= bg_tol).astype(np.uint8)
    _, labels = cv2.connectedComponents(cand, connectivity=4)

    border_labels = set(
        np.unique(
            np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
        ).tolist()
    ) - {0}

    bg_mask = np.isin(labels, list(border_labels))
    return bg_mask, bg_color


def split_foreground_regions(
    img_bgr: np.ndarray,
    bg_mask: np.ndarray,
    bg_color: np.ndarray,
    min_region_area: int = 250,
    merge_dist: int = 14,
    pad: int = 6,
) -> list[dict]:
    """Split a multi-object sheet into independent foreground regions automatically.

    Each returned dict has:
      - bbox: (x0, y0, x1, y1) in original 1x coordinates (before padding)
      - pad: padding added around crop
      - crop_bgr: padded crop with neighboring-object pixels masked to bg_color
      - crop_bg_mask: bool background mask for the padded crop
    If the image is a single cohesive illustration, returns 1 region covering it.
    """
    h, w = img_bgr.shape[:2]
    fg = (~bg_mask).astype(np.uint8)
    # Remove 1-2px isolated JPEG dust specks from foreground mask
    n_cc, lab, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)

    if n_cc <= 1:
        return []

    major_ids = [i for i in range(1, n_cc) if stats[i, cv2.CC_STAT_AREA] >= min_region_area]
    minor_ids = [
        i
        for i in range(1, n_cc)
        if 4 <= stats[i, cv2.CC_STAT_AREA] < min_region_area
    ]

    # If everything is small or only 1 major region, treat whole foreground as single region
    if len(major_ids) <= 1:
        ys, xs = np.where(fg > 0)
        if len(xs) == 0:
            return []
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        crop = img_bgr[y0:y1, x0:x1].copy()
        c_bg = bg_mask[y0:y1, x0:x1].copy()
        crop_p = cv2.copyMakeBorder(
            crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg_color.tolist()
        )
        c_bg_p = cv2.copyMakeBorder(
            c_bg.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=1
        ).astype(bool)
        return [{"bbox": (x0, y0, x1, y1), "pad": pad, "crop_bgr": crop_p, "crop_bg_mask": c_bg_p}]

    # Assign each minor component to the nearest major component if within merge_dist
    major_mask = np.isin(lab, major_ids)
    dist_to_major, nearest_idx = ndi.distance_transform_edt(~major_mask, return_indices=True)

    groups: dict[int, list[int]] = {mid: [mid] for mid in major_ids}
    orphan_minors: list[int] = []
    for sid in minor_ids:
        ys, xs = np.where(lab == sid)
        d_min = dist_to_major[ys, xs].min()
        if d_min <= merge_dist:
            best_pos = int(np.argmin(dist_to_major[ys, xs]))
            ny = nearest_idx[0, ys[best_pos], xs[best_pos]]
            nx = nearest_idx[1, ys[best_pos], xs[best_pos]]
            parent = int(lab[ny, nx])
            if parent in groups:
                groups[parent].append(sid)
            else:
                orphan_minors.append(sid)
        else:
            orphan_minors.append(sid)

    for oid in orphan_minors:
        groups[oid] = [oid]

    # Sort regions top-to-bottom, left-to-right (deterministic order)
    regions = []
    for root_id, member_ids in groups.items():
        own_fg = np.isin(lab, member_ids)
        ys, xs = np.where(own_fg)
        if len(xs) == 0:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        # Keep 2px anti-aliased margin around own_fg, mask out pixels belonging to OTHER objects
        own_dil = cv2.dilate(own_fg[y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        other_fg = (~bg_mask[y0:y1, x0:x1]) & (~own_fg[y0:y1, x0:x1])

        crop = img_bgr[y0:y1, x0:x1].copy()
        crop[other_fg | (~own_dil)] = bg_color.astype(np.uint8)
        c_bg = bg_mask[y0:y1, x0:x1] | other_fg | (~own_dil)

        crop_p = cv2.copyMakeBorder(
            crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg_color.tolist()
        )
        c_bg_p = cv2.copyMakeBorder(
            c_bg.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=1
        ).astype(bool)

        regions.append(
            {
                "bbox": (x0, y0, x1, y1),
                "pad": pad,
                "crop_bgr": crop_p,
                "crop_bg_mask": c_bg_p,
                "cy": (y0 + y1) / 2.0,
                "cx": (x0 + x1) / 2.0,
            }
        )

    # Sort cleanly by row bands then x
    regions.sort(key=lambda r: (round(r["cy"] / 60.0), r["cx"]))
    return regions
