"""Cubic Bezier curve fitting (Philip Schneider algorithm) and SVG path generation."""

from __future__ import annotations
import cv2
import numpy as np


def _fmt(v: float) -> str:
    s = f"{v:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def _smooth_closed_contour(
    pts: np.ndarray,
    iters: int = 10,
    pinned_indices: set[int] | None = None,
) -> np.ndarray:
    """Corner-preserving closed-loop Gaussian smoothing that eliminates the 4-pixel-period
    1x staircase wave at 4x resolution while keeping true sharp corners pinned.
    """
    p = pts.astype(np.float64).copy()
    n = len(p)
    if n < 5:
        return p
    pinned_mask = np.zeros(n, dtype=bool)
    if pinned_indices:
        for idx in pinned_indices:
            if 0 <= idx < n:
                pinned_mask[idx] = True
    orig = p.copy()
    for _ in range(iters):
        prev_p = np.roll(p, 1, axis=0)
        next_p = np.roll(p, -1, axis=0)
        p = 0.25 * prev_p + 0.50 * p + 0.25 * next_p
        if pinned_indices:
            p[pinned_mask] = orig[pinned_mask]
    return p


def _detect_corners(pts: np.ndarray, step: int = 4, angle_deg: float = 52.0) -> list[int]:
    """Detect sharp corners along a closed contour using chord turning angles."""
    n = len(pts)
    if n < step * 3:
        return []

    prev_p = np.roll(pts, step, axis=0)
    next_p = np.roll(pts, -step, axis=0)
    v1 = pts - prev_p
    v2 = next_p - pts

    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    valid = denom > 1e-6

    cos_a = np.ones(n, dtype=np.float64)
    cos_a[valid] = np.clip(np.sum(v1[valid] * v2[valid], axis=1) / denom[valid], -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_a))

    # Non-maximum suppression within +/- step window
    corners: list[int] = []
    for i in range(n):
        if angles[i] < angle_deg:
            continue
        is_max = True
        for off in range(-step, step + 1):
            if off == 0:
                continue
            j = (i + off) % n
            if angles[j] > angles[i] or (angles[j] == angles[i] and j < i):
                is_max = False
                break
        if is_max:
            corners.append(i)
    return corners


def _chord_length_params(pts: np.ndarray) -> np.ndarray:
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    u = np.zeros(len(pts), dtype=np.float64)
    u[1:] = np.cumsum(seg_lens)
    total = u[-1]
    if total > 1e-9:
        u /= total
    return u


def _eval_bezier(b0: np.ndarray, b1: np.ndarray, b2: np.ndarray, b3: np.ndarray, u: np.ndarray) -> np.ndarray:
    mt = 1.0 - u
    return (
        (mt ** 3)[:, None] * b0
        + (3.0 * (mt ** 2) * u)[:, None] * b1
        + (3.0 * mt * (u ** 2))[:, None] * b2
        + (u ** 3)[:, None] * b3
    )


def _generate_bezier(
    pts: np.ndarray, u: np.ndarray, t_hat1: np.ndarray, t_hat2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares cubic Bezier fit with fixed endpoints & end tangents."""
    p0 = pts[0]
    p3 = pts[-1]
    mt = 1.0 - u

    a1 = (3.0 * (mt ** 2) * u)[:, None] * t_hat1[None, :]
    a2 = (3.0 * mt * (u ** 2))[:, None] * t_hat2[None, :]

    c00 = float(np.sum(a1 * a1))
    c01 = float(np.sum(a1 * a2))
    c11 = float(np.sum(a2 * a2))

    # Residual from baseline cubic with alpha1 = alpha2 = 0
    base = ((mt ** 3 + 3.0 * (mt ** 2) * u)[:, None] * p0) + ((3.0 * mt * (u ** 2) + u ** 3)[:, None] * p3)
    tmp = pts - base

    x0 = float(np.sum(a1 * tmp))
    x1 = float(np.sum(a2 * tmp))

    det = c00 * c11 - c01 * c01
    if abs(det) > 1e-9:
        alpha_l = (x0 * c11 - x1 * c01) / det
        alpha_r = (c00 * x1 - c01 * x0) / det
    else:
        alpha_l = 0.0
        alpha_r = 0.0

    seg_len = float(np.linalg.norm(p3 - p0))
    eps = 1e-6 * seg_len
    if alpha_l < eps or alpha_r < eps or alpha_l > seg_len * 1.5 or alpha_r > seg_len * 1.5:
        d = seg_len / 3.0
        return p0, p0 + t_hat1 * d, p3 + t_hat2 * d, p3

    return p0, p0 + t_hat1 * alpha_l, p3 + t_hat2 * alpha_r, p3


def _compute_max_error(
    pts: np.ndarray,
    bez: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    u: np.ndarray,
) -> tuple[float, int]:
    fitted = _eval_bezier(bez[0], bez[1], bez[2], bez[3], u)
    dists = np.linalg.norm(fitted - pts, axis=1)
    idx = int(np.argmax(dists))
    return float(dists[idx]), idx


def _unit_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0], dtype=np.float64)


def _fit_cubic_recursive(
    pts: np.ndarray,
    t_hat1: np.ndarray,
    t_hat2: np.ndarray,
    tol: float,
    depth: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    n = len(pts)
    if n <= 2:
        p0, p3 = pts[0], pts[-1]
        d = float(np.linalg.norm(p3 - p0)) / 3.0
        return [(p0, p0 + t_hat1 * d, p3 + t_hat2 * d, p3)]

    # Straight line check
    p0, p3 = pts[0], pts[-1]
    line_vec = p3 - p0
    line_len = float(np.linalg.norm(line_vec))
    if line_len > 1e-6:
        # Distance from each point to line segment p0->p3
        rel = pts - p0
        cross = np.abs(rel[:, 0] * line_vec[1] - rel[:, 1] * line_vec[0]) / line_len
        if float(cross.max()) <= tol * 0.65:
            return [(p0, p0 + line_vec / 3.0, p3 - line_vec / 3.0, p3)]

    u = _chord_length_params(pts)
    bez = _generate_bezier(pts, u, t_hat1, t_hat2)
    max_err, split_idx = _compute_max_error(pts, bez, u)

    if max_err <= tol or depth >= 8 or n <= 4:
        return [bez]

    # Split at worst error point (clamp away from endpoints)
    split_idx = max(1, min(n - 2, split_idx))
    t_center = _unit_vec(pts[split_idx + 1] - pts[split_idx - 1])

    left = _fit_cubic_recursive(pts[: split_idx + 1], t_hat1, -t_center, tol, depth + 1)
    right = _fit_cubic_recursive(pts[split_idx:], t_center, t_hat2, tol, depth + 1)
    return left + right


def contour_to_svg_subpath(
    contour_4x: np.ndarray,
    scale: float = 4.0,
    fit_tol_1x: float = 0.35,
    corner_deg: float = 52.0,
) -> str:
    """Convert a single OpenCV contour (at `scale`x resolution) into a smooth SVG subpath string."""
    pts_raw = contour_4x.reshape(-1, 2).astype(np.float64)
    if len(pts_raw) < 3:
        return ""

    # Detect true sharp corners at 4x resolution using a ~2.5px 1x chord (step=10 at 4x)
    # BEFORE smoothing, then smooth the 1x staircase while keeping corners pinned!
    n_raw = len(pts_raw)
    step_raw = max(3, min(10, n_raw // 5))
    raw_corners = _detect_corners(pts_raw, step=step_raw, angle_deg=corner_deg)
    pts_smooth = _smooth_closed_contour(pts_raw, iters=10, pinned_indices=set(raw_corners))
    pts = pts_smooth / float(scale)

    # Deduplicate consecutive near-identical points
    diffs = np.linalg.norm(np.diff(pts, axis=0, append=pts[:1]), axis=1)
    pts = pts[diffs > 1e-4]
    n = len(pts)
    if n < 3:
        return ""

    step = max(3, min(9, n // 5))
    corners = _detect_corners(pts, step=step, angle_deg=corner_deg)

    segments: list[np.ndarray] = []
    if len(corners) == 0:
        # Smooth closed loop: split into 2 halves at farthest points for stable fitting
        half = n // 2
        segments.append(pts[: half + 1])
        segments.append(np.vstack([pts[half:], pts[:1]]))
    else:
        for idx_c in range(len(corners)):
            a = corners[idx_c]
            b = corners[(idx_c + 1) % len(corners)]
            if b > a:
                seg = pts[a : b + 1]
            else:
                seg = np.vstack([pts[a:], pts[: b + 1]])
            if len(seg) >= 2:
                segments.append(seg)

    beziers: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for seg in segments:
        k = min(3, len(seg) - 1)
        t1 = _unit_vec(seg[k] - seg[0])
        t2 = _unit_vec(seg[-1 - k] - seg[-1])
        beziers.extend(_fit_cubic_recursive(seg, t1, t2, tol=fit_tol_1x))

    if not beziers:
        return ""

    parts = [f"M{_fmt(beziers[0][0][0])} {_fmt(beziers[0][0][1])}"]
    for b0, b1, b2, b3 in beziers:
        # Check if control points are nearly collinear with chord -> emit compact L
        chord = b3 - b0
        clen = float(np.linalg.norm(chord))
        if clen < 1e-4:
            continue
        d1 = abs((b1[0] - b0[0]) * chord[1] - (b1[1] - b0[1]) * chord[0]) / clen
        d2 = abs((b2[0] - b0[0]) * chord[1] - (b2[1] - b0[1]) * chord[0]) / clen
        if d1 < 0.08 and d2 < 0.08:
            parts.append(f"L{_fmt(b3[0])} {_fmt(b3[1])}")
        else:
            parts.append(
                f"C{_fmt(b1[0])} {_fmt(b1[1])} {_fmt(b2[0])} {_fmt(b2[1])} {_fmt(b3[0])} {_fmt(b3[1])}"
            )
    parts.append("Z")
    return "".join(parts)


def mask_to_svg_path_d(
    mask_4x: np.ndarray,
    scale: float = 4.0,
    min_area_4x: float = 16.0,
    fit_tol_1x: float = 0.35,
    corner_deg: float = 52.0,
) -> str:
    """Convert a binary uint8 mask (0/255) at `scale`x resolution into a compound SVG path `d` string.

    Uses RETR_CCOMP + fill-rule="evenodd" so outer contours and holes combine cleanly.
    """
    contours, _ = cv2.findContours(mask_4x, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    subpaths: list[str] = []
    for cnt in contours:
        if len(cnt) < 4:
            continue
        area = cv2.contourArea(cnt)
        if area < min_area_4x:
            continue
        sp = contour_to_svg_subpath(cnt, scale=scale, fit_tol_1x=fit_tol_1x, corner_deg=corner_deg)
        if sp:
            subpaths.append(sp)
    return " ".join(subpaths)
