from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyvista as pv
import vtk


@dataclass
class ShrinkStats:
    iterations_done: int
    accepted_steps: int
    rejected_steps: int
    final_step: float


def _tri_faces(mesh: pv.PolyData) -> np.ndarray:
    tri = mesh.triangulate().clean()
    faces = tri.faces.reshape((-1, 4))
    if np.any(faces[:, 0] != 3):
        raise ValueError("Mesh contains non-triangle faces after triangulate()")
    return faces[:, 1:4]


def _vertex_adjacency(n_points: int, tris: np.ndarray) -> list[list[int]]:
    adj = [set() for _ in range(n_points)]
    for a, b, c in tris:
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)
    return [sorted(x) for x in adj]


def _triangle_areas(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    p0 = points[tris[:, 0]]
    p1 = points[tris[:, 1]]
    p2 = points[tris[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _min_edge_lengths_per_vertex(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    n = len(points)
    out = np.full(n, np.inf, dtype=float)

    for a, b, c in tris:
        pab = np.linalg.norm(points[a] - points[b])
        pbc = np.linalg.norm(points[b] - points[c])
        pca = np.linalg.norm(points[c] - points[a])

        out[a] = min(out[a], pab, pca)
        out[b] = min(out[b], pab, pbc)
        out[c] = min(out[c], pbc, pca)

    out[~np.isfinite(out)] = 0.0
    return out


def _build_target_locator(target: pv.PolyData) -> vtk.vtkStaticCellLocator:
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(target)
    locator.BuildLocator()
    return locator


def _closest_points_and_dist2(points: np.ndarray, locator: vtk.vtkStaticCellLocator) -> tuple[np.ndarray, np.ndarray]:
    cp = [0.0, 0.0, 0.0]
    cell_id = vtk.mutable(0)
    sub_id = vtk.mutable(0)
    dist2 = vtk.mutable(0.0)

    closest = np.empty_like(points, dtype=float)
    d2 = np.empty(len(points), dtype=float)

    for i, p in enumerate(points):
        locator.FindClosestPoint(p, cp, cell_id, sub_id, dist2)
        closest[i] = cp
        d2[i] = float(dist2)

    return closest, d2


def _laplacian_smooth_points(points: np.ndarray, adjacency: list[list[int]], alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return points.copy()

    out = points.copy()
    for i, nbrs in enumerate(adjacency):
        if not nbrs:
            continue
        avg = np.mean(points[nbrs], axis=0)
        out[i] = (1.0 - alpha) * points[i] + alpha * avg
    return out


def _quality_ok(
    old_points: np.ndarray,
    new_points: np.ndarray,
    tris: np.ndarray,
    min_area_ratio: float,
    max_area_growth: float,
) -> bool:
    old_areas = _triangle_areas(old_points, tris)
    new_areas = _triangle_areas(new_points, tris)

    old_positive = old_areas > 0.0
    if np.any(old_positive):
        ratio = np.zeros_like(new_areas)
        ratio[old_positive] = new_areas[old_positive] / old_areas[old_positive]

        if np.any(ratio[old_positive] < min_area_ratio):
            return False

        if np.any(ratio[old_positive] > max_area_growth):
            return False

    if np.any(new_areas <= 0.0):
        return False

    return True


def shrink_wrap(
    shell: pv.PolyData,
    target: pv.PolyData,
    iterations: int = 30,
    step: float = 0.25,
    smooth_alpha: float = 0.10,
    constraint_mul: float = 0.35,
    min_step: float = 1e-4,
    step_decay_on_reject: float = 0.5,
    min_area_ratio: float = 0.15,
    max_area_growth: float = 4.0,
    return_stats: bool = False,
) -> pv.PolyData | tuple[pv.PolyData, ShrinkStats]:
    """
    Стягивает начальную оболочку к target.
    Основные отличия от простой версии:
    - ограничение смещения через локальную длину ребра
    - rejection/rollback плохого шага
    - адаптивное уменьшение шага
    - мягкое лапласово сглаживание после принятого шага
    """

    work = shell.triangulate().clean()
    tgt = target.triangulate().clean()

    if work.n_points == 0 or work.n_cells == 0:
        stats = ShrinkStats(
            iterations_done=0,
            accepted_steps=0,
            rejected_steps=0,
            final_step=step,
        )
        return (work, stats) if return_stats else work

    tris = _tri_faces(work)
    adjacency = _vertex_adjacency(work.n_points, tris)
    locator = _build_target_locator(tgt)

    points = np.asarray(work.points, dtype=float).copy()

    accepted = 0
    rejected = 0
    cur_step = float(step)
    done = 0

    for it in range(iterations):
        done = it + 1

        closest, _dist2 = _closest_points_and_dist2(points, locator)
        direction = closest - points

        min_edges = _min_edge_lengths_per_vertex(points, tris)
        max_move = constraint_mul * min_edges

        norms = np.linalg.norm(direction, axis=1)
        move = cur_step * direction

        move_norms = np.linalg.norm(move, axis=1)
        clamp_mask = (move_norms > max_move) & (move_norms > 0.0)
        if np.any(clamp_mask):
            move[clamp_mask] *= (max_move[clamp_mask] / move_norms[clamp_mask])[:, None]

        candidate = points + move

        if smooth_alpha > 0.0:
            candidate = _laplacian_smooth_points(candidate, adjacency, smooth_alpha)

        ok = _quality_ok(
            old_points=points,
            new_points=candidate,
            tris=tris,
            min_area_ratio=min_area_ratio,
            max_area_growth=max_area_growth,
        )

        if ok:
            points = candidate
            accepted += 1
        else:
            rejected += 1
            cur_step *= step_decay_on_reject
            if cur_step < min_step:
                break
            continue

        if np.max(norms) < 1e-9:
            break

    out = work.copy(deep=True)
    out.points = points
    out = out.clean().triangulate()

    stats = ShrinkStats(
        iterations_done=done,
        accepted_steps=accepted,
        rejected_steps=rejected,
        final_step=cur_step,
    )

    return (out, stats) if return_stats else out