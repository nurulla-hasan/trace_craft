"""Phase 2: Shared-Edge Topological Graph & Shared Cubic Bezier Fitter.

Extracts the exact planar partition graph of a 2D label map (`label_4x`), identifies all
triple/multi-label junction vertices, fits smooth Philip Schneider cubic Beziers ONCE per
shared boundary chain, and stitches exact closed SVG paths for every foreground color.
"""

from __future__ import annotations
from collections import defaultdict
import numpy as np

from .bezier import (
    _fmt,
    _detect_corners,
    _fit_cubic_recursive,
    _unit_vec,
    _smooth_closed_contour,
)


def _smooth_open_chain(
    pts: np.ndarray,
    iters: int = 10,
    pinned_indices: set[int] | None = None,
) -> np.ndarray:
    """Smooth an open polyline while keeping its two junction endpoints and any interior sharp corners pinned."""
    p = pts.astype(np.float64).copy()
    n = len(p)
    if n < 4:
        return p
    orig = p.copy()
    pinned_mask = np.zeros(n, dtype=bool)
    pinned_mask[0] = True
    pinned_mask[-1] = True
    if pinned_indices:
        for idx in pinned_indices:
            if 0 <= idx < n:
                pinned_mask[idx] = True
    for _ in range(iters):
        inner = 0.25 * p[:-2] + 0.50 * p[1:-1] + 0.25 * p[2:]
        p[1:-1] = inner
        p[pinned_mask] = orig[pinned_mask]
    return p


def _fit_chain_beziers(
    pts_4x: np.ndarray,
    is_closed: bool,
    scale: float = 4.0,
    fit_tol_1x: float = 0.48,
    corner_deg: float = 54.0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Fit a list of cubic Beziers (b0, b1, b2, b3) in 1x coordinates to a boundary chain."""
    if is_closed:
        core = pts_4x[:-1] if np.allclose(pts_4x[0], pts_4x[-1]) else pts_4x
        n_core = len(core)
        if n_core < 3:
            return []
        step_raw = max(3, min(10, n_core // 5))
        raw_corners = _detect_corners(core, step=step_raw, angle_deg=corner_deg)
        pts_s = _smooth_closed_contour(core, iters=10, pinned_indices=set(raw_corners)) / float(scale)
        diffs = np.linalg.norm(np.diff(pts_s, axis=0, append=pts_s[:1]), axis=1)
        pts = pts_s[diffs > 1e-4]
        n = len(pts)
        if n < 3:
            return []
        step = max(3, min(9, n // 5))
        corners = _detect_corners(pts, step=step, angle_deg=corner_deg)
        segments: list[np.ndarray] = []
        if not corners:
            half = n // 2
            segments.append(pts[: half + 1])
            segments.append(np.vstack([pts[half:], pts[:1]]))
        else:
            for idx_c in range(len(corners)):
                a = corners[idx_c]
                b = corners[(idx_c + 1) % len(corners)]
                seg = pts[a : b + 1] if b > a else np.vstack([pts[a:], pts[: b + 1]])
                if len(seg) >= 2:
                    segments.append(seg)
    else:
        n_raw = len(pts_4x)
        if n_raw < 2:
            return []
        # Detect interior corners along open chain using a ~2.5px 1x chord (step=10 at 4x)
        interior_corners: list[int] = []
        step_open = min(10, max(3, n_raw // 4))
        if n_raw >= step_open * 2 + 3:
            for i in range(step_open, n_raw - step_open):
                v1 = pts_4x[i] - pts_4x[i - step_open]
                v2 = pts_4x[i + step_open] - pts_4x[i]
                l1 = float(np.linalg.norm(v1))
                l2 = float(np.linalg.norm(v2))
                if l1 > 1e-6 and l2 > 1e-6:
                    ca = np.clip(float(np.dot(v1, v2)) / (l1 * l2), -1.0, 1.0)
                    ang = float(np.degrees(np.arccos(ca)))
                    if ang >= corner_deg:
                        if not interior_corners or (i - interior_corners[-1]) >= step_open:
                            interior_corners.append(i)

        pts = _smooth_open_chain(pts_4x, iters=10, pinned_indices=set(interior_corners)) / float(scale)
        # Split open chain at detected interior corners so sharp fin/tail tips stay sharp!
        splits = [0] + interior_corners + [len(pts) - 1]
        segments = []
        for idx_s in range(len(splits) - 1):
            a, b = splits[idx_s], splits[idx_s + 1]
            seg = pts[a : b + 1]
            if len(seg) > 2:
                keep = [0]
                for i in range(1, len(seg) - 1):
                    if float(np.linalg.norm(seg[i] - seg[keep[-1]])) > 1e-4:
                        keep.append(i)
                keep.append(len(seg) - 1)
                seg = seg[keep]
            if len(seg) >= 2:
                segments.append(seg)

    beziers: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for seg in segments:
        if len(seg) < 2:
            continue
        k = min(3, len(seg) - 1)
        t1 = _unit_vec(seg[k] - seg[0])
        t2 = _unit_vec(seg[-1 - k] - seg[-1])
        beziers.extend(_fit_cubic_recursive(seg, t1, t2, tol=fit_tol_1x))
    return beziers


def _beziers_to_svg_cmds(
    beziers: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    emit_move: bool = True,
) -> str:
    if not beziers:
        return ""
    parts: list[str] = []
    if emit_move:
        parts.append(f"M{_fmt(beziers[0][0][0])} {_fmt(beziers[0][0][1])}")
    for b0, b1, b2, b3 in beziers:
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
    return "".join(parts)


def _reverse_beziers(
    beziers: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    return [(b3, b2, b1, b0) for (b0, b1, b2, b3) in reversed(beziers)]


def extract_shared_edge_paths(
    label_4x: np.ndarray,
    bg_label: int,
    scale: float = 4.0,
    min_loop_area_4x: float = 14.0,
    fit_tol_1x: float = 0.32,
    corner_deg: float = 52.0,
) -> dict[int, str]:
    """Build the shared-edge planar partition graph of `label_4x` and return {color_idx: svg_d}.

    Every shared boundary between two colors is fitted ONCE as cubic Beziers, and both adjacent
    regions reuse the exact same curve (forward / reversed), eliminating duplicate nodes and
    ensuring topologically exact boundaries.
    """
    H, W = label_4x.shape
    # Pad by 1 pixel with bg_label so every foreground region is strictly closed inside the grid
    L = np.pad(label_4x, 1, mode="constant", constant_values=bg_label)
    Hp, Wp = L.shape

    # Extract all directed unit crack edges where left_label != right_label.
    # Convention: walking from u -> v has `left_label` on the LEFT (interior of CCW loop in y-down).
    # Horizontal cracks between row y-1 (above) and row y (below):
    #   Edge from (x, y) -> (x+1, y) has (y, x) on RIGHT (below in y-down) and (y-1, x) on LEFT (above).
    #   So walking (x+1, y) -> (x, y) has below=(y, x) on LEFT!
    # Vertical cracks between col x-1 (left) and col x (right):
    #   Edge from (x, y) -> (x, y+1) has (y, x) on LEFT (in y-down: dx=0, dy=+1 -> left is +x)!
    # Let's store undirected canonical crack edges keyed by sorted label pair (la, lb) with la < lb,
    # oriented so `la` is ALWAYS on the LEFT.

    # Vectorized extraction of crack edges:
    # 1. Horizontal boundaries between L[y-1, x] and L[y, x], for y in 1..Hp-1, x in 0..Wp-1
    hy, hx = np.where(L[:-1, :] != L[1:, :])
    # Grid vertex coords (in padded grid, offset by -1 to match original 4x coords):
    # top pixel = L[hy, hx], bot pixel = L[hy+1, hx]
    # Walking (hx, hy+1) -> (hx+1, hy+1): dx=+1, dy=0 -> LEFT in y-down is (0, -1) = top pixel!
    # Walking (hx+1, hy+1) -> (hx, hy+1): LEFT in y-down is bot pixel!

    # 2. Vertical boundaries between L[y, x-1] and L[y, x], for y in 0..Hp-1, x in 1..Wp-1
    vy, vx = np.where(L[:, :-1] != L[:, 1:])
    # left pixel = L[vy, vx], right pixel = L[vy, vx+1]
    # Walking (vx+1, vy) -> (vx+1, vy+1): dx=0, dy=+1 -> LEFT in y-down is (+1, 0) = right pixel!
    # Walking (vx+1, vy+1) -> (vx+1, vy): LEFT in y-down is left pixel!

    # Map each vertex (gx, gy) -> count of incident undirected crack edges (to find degree != 2 junctions)
    # Also store directed edges grouped by canonical pair (min_l, max_l)
    pair_edges: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]] = defaultdict(list)
    v_deg: dict[tuple[int, int], int] = defaultdict(int)

    top_l = L[hy, hx]
    bot_l = L[hy + 1, hx]
    for x, y, tl, bl in zip(hx.tolist(), hy.tolist(), top_l.tolist(), bot_l.tolist()):
        u = (x - 1, y)
        v = (x, y)
        v_deg[u] += 1
        v_deg[v] += 1
        if tl < bl:
            pair_edges[(tl, bl)].append((u, v))  # tl on LEFT
        else:
            pair_edges[(bl, tl)].append((v, u))  # bl on LEFT

    left_l = L[vy, vx]
    right_l = L[vy, vx + 1]
    for x, y, ll, rl in zip(vx.tolist(), vy.tolist(), left_l.tolist(), right_l.tolist()):
        u = (x, y - 1)
        v = (x, y)
        v_deg[u] += 1
        v_deg[v] += 1
        if rl < ll:
            pair_edges[(rl, ll)].append((u, v))  # rl on LEFT
        else:
            pair_edges[(ll, rl)].append((v, u))  # ll on LEFT

    # Trace maximal non-branching chains for each canonical pair (la, lb)
    # Every chain runs between two junction vertices (v_deg != 2) or forms a closed loop.
    # Store fitted chains per color: color -> list of (start_v, end_v, beziers, raw_pts)
    color_chains: dict[int, list[tuple[tuple[int, int], tuple[int, int], list, bool]]] = defaultdict(list)

    for (la, lb), edges in pair_edges.items():
        if la == bg_label and lb == bg_label:
            continue
        adj: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, (u, v) in enumerate(edges):
            adj[u].append(idx)

        used = [False] * len(edges)

        # Helper to trace a chain starting from edge `e_idx`
        def trace_from(e_idx: int) -> tuple[list[tuple[int, int]], bool]:
            used[e_idx] = True
            u, v = edges[e_idx]
            pts = [u, v]
            curr = v
            while curr != u and v_deg[curr] == 2:
                nxt_list = adj.get(curr)
                if not nxt_list:
                    break
                moved = False
                for ne in nxt_list:
                    if not used[ne]:
                        used[ne] = True
                        _, nv = edges[ne]
                        pts.append(nv)
                        curr = nv
                        moved = True
                        break
                if not moved:
                    break
            return pts, (curr == u and v_deg[u] == 2)

        # 1. First trace all open chains starting at junction vertices (v_deg != 2)
        for idx, (u, v) in enumerate(edges):
            if not used[idx] and v_deg[u] != 2:
                pts, is_closed = trace_from(idx)
                pts_arr = np.array(pts, dtype=np.float64)
                bez = _fit_chain_beziers(
                    pts_arr, is_closed=False, scale=scale, fit_tol_1x=fit_tol_1x, corner_deg=corner_deg
                )
                if bez:
                    if la != bg_label:
                        color_chains[la].append((pts[0], pts[-1], bez, False))
                    if lb != bg_label:
                        color_chains[lb].append((pts[-1], pts[0], _reverse_beziers(bez), False))

        # 2. Then trace any remaining closed loops (all vertices have degree == 2)
        for idx in range(len(edges)):
            if not used[idx]:
                pts, _ = trace_from(idx)
                pts_arr = np.array(pts, dtype=np.float64)
                # Area filter for tiny closed loops
                x = pts_arr[:, 0]
                y = pts_arr[:, 1]
                area = 0.5 * abs(float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])))
                if area < min_loop_area_4x:
                    continue
                bez = _fit_chain_beziers(
                    pts_arr, is_closed=True, scale=scale, fit_tol_1x=fit_tol_1x, corner_deg=corner_deg
                )
                if bez:
                    if la != bg_label:
                        color_chains[la].append((pts[0], pts[0], bez, True))
                    if lb != bg_label:
                        color_chains[lb].append((pts[0], pts[0], _reverse_beziers(bez), True))

    # Stitch open chains into closed loops per color
    result: dict[int, str] = {}
    for color, chains in color_chains.items():
        subpaths: list[str] = []
        open_chains: list[tuple[tuple[int, int], tuple[int, int], list]] = []

        for s_v, e_v, bez, is_closed in chains:
            if is_closed:
                cmd = _beziers_to_svg_cmds(bez, emit_move=True)
                if cmd:
                    subpaths.append(cmd + "Z")
            else:
                open_chains.append((s_v, e_v, bez))

        # Stitch open chains by matching end vertex -> start vertex
        if open_chains:
            start_map: dict[tuple[int, int], list[int]] = defaultdict(list)
            for idx, (s_v, e_v, _) in enumerate(open_chains):
                start_map[s_v].append(idx)
            used_c = [False] * len(open_chains)

            for idx in range(len(open_chains)):
                if used_c[idx]:
                    continue
                used_c[idx] = True
                s0, curr_v, bez0 = open_chains[idx]
                loop_bez = list(bez0)
                while curr_v != s0:
                    cands = start_map.get(curr_v, [])
                    picked = -1
                    for ci in cands:
                        if not used_c[ci]:
                            picked = ci
                            break
                    if picked < 0:
                        break
                    used_c[picked] = True
                    _, curr_v, b_next = open_chains[picked]
                    loop_bez.extend(b_next)

                # Filter tiny stitched loops by bounding box / signed area of Bezier endpoints
                if len(loop_bez) >= 2:
                    ep = np.array([b[0] for b in loop_bez], dtype=np.float64)
                    area_1x = 0.5 * abs(
                        float(
                            np.dot(ep[:, 0], np.roll(ep[:, 1], -1))
                            - np.dot(ep[:, 1], np.roll(ep[:, 0], -1))
                        )
                    )
                    if area_1x * (scale * scale) >= min_loop_area_4x:
                        cmd = _beziers_to_svg_cmds(loop_bez, emit_move=True)
                        if cmd:
                            subpaths.append(cmd + "Z")

        if subpaths:
            result[color] = " ".join(subpaths)

    return result
