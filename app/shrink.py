from typing import Dict, List, Set, Tuple

import numpy as np
import pyvista as pv
import vtk


Tri = Tuple[int, int, int]
Edge = Tuple[int, int]


def build_shell_adjacency(shell: pv.PolyData) -> List[Set[int]]:
    n = shell.n_points
    adj: List[Set[int]] = [set() for _ in range(n)]
    faces = shell.faces.reshape((-1, 4))
    for f in faces:
        a, b, c = int(f[1]), int(f[2]), int(f[3])
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)
    return adj


def compute_min_edge_lengths(points: np.ndarray, adj: List[Set[int]]) -> np.ndarray:
    n = points.shape[0]
    out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        nei = adj[i]
        if not nei:
            out[i] = 0.0
            continue

        pi = points[i]
        vals = [np.linalg.norm(points[j] - pi) for j in nei]
        out[i] = float(min(vals)) if vals else 0.0

    return out


def build_surface_locator(mesh: pv.PolyData) -> vtk.vtkStaticCellLocator:
    poly = mesh.extract_surface(algorithm="dataset_surface").triangulate().clean()
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    return locator


def closest_points(locator: vtk.vtkStaticCellLocator, pts: np.ndarray) -> np.ndarray:
    out = np.empty_like(pts)
    cp = [0.0, 0.0, 0.0]
    cid = vtk.mutable(0)
    sid = vtk.mutable(0)
    dist2 = vtk.mutable(0.0)

    for i in range(pts.shape[0]):
        locator.FindClosestPoint(pts[i].tolist(), cp, cid, sid, dist2)
        out[i, 0] = cp[0]
        out[i, 1] = cp[1]
        out[i, 2] = cp[2]

    return out


def _build_triangles(shell: pv.PolyData) -> List[Tri]:
    faces = shell.faces.reshape((-1, 4))
    tris: List[Tri] = []
    for f in faces:
        tris.append((int(f[1]), int(f[2]), int(f[3])))
    return tris


def _build_unique_edges(tris: List[Tri]) -> List[Edge]:
    edges = set()
    for a, b, c in tris:
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return sorted(edges)


def _laplacian_displacements(points: np.ndarray, adj: List[Set[int]]) -> np.ndarray:
    disp = np.zeros_like(points)
    for i in range(points.shape[0]):
        nei = list(adj[i])
        if not nei:
            continue
        avg = np.mean(points[nei], axis=0)
        disp[i] = avg - points[i]
    return disp


def shrink_shell_to_mesh(
    shell: pv.PolyData,
    target_mesh: pv.PolyData,
    iters: int,
    step: float,
    constraint_mul: float,
    lap_iters_per_step: int,
    lap_relax: float,
    progress_cb=None,
) -> pv.PolyData:
    if iters <= 0:
        return shell

    out = shell.copy(deep=True).triangulate().clean()
    if out.n_points == 0 or out.n_cells == 0:
        return out

    pts = np.asarray(out.points, dtype=np.float64).copy()
    tris = _build_triangles(out)
    edges = _build_unique_edges(tris)
    adj = build_shell_adjacency(out)
    locator = build_surface_locator(target_mesh)

    ref_pts = pts.copy()

    ref_edge_len: Dict[Edge, float] = {}
    for i, j in edges:
        ref_edge_len[(i, j)] = float(np.linalg.norm(ref_pts[j] - ref_pts[i]))

    k_edge = 0.35
    k_attr = 1.20
    k_lap = 0.20

    for it in range(int(iters)):
        if progress_cb is not None:
            progress_cb(it + 1, int(iters))

        forces = np.zeros_like(pts)

        # 1. Пружинные силы по рёбрам
        for i, j in edges:
            e = pts[j] - pts[i]
            l = float(np.linalg.norm(e))
            if l <= 1e-15:
                continue

            l0 = ref_edge_len[(i, j)]
            d = l - l0
            dir_ij = e / l
            f = k_edge * d * dir_ij

            forces[i] += f
            forces[j] -= f

        # 2. Притяжение к поверхности
        cp = closest_points(locator, pts)
        attr = cp - pts
        forces += k_attr * attr

        # 3. Локальная регуляризация
        lap = _laplacian_displacements(pts, adj)
        forces += k_lap * lap_relax * lap

        # 4. Ограничение шага по локальному размеру
        disp = step * forces
        min_edge = compute_min_edge_lengths(pts, adj)
        max_shift = min_edge * float(constraint_mul)

        disp_len = np.linalg.norm(disp, axis=1)
        scale = np.ones_like(disp_len)
        mask = disp_len > 0.0
        scale[mask] = np.minimum(1.0, max_shift[mask] / disp_len[mask])

        pts = pts + disp * scale[:, None]

        # 5. Дополнительная мягкая релаксация
        for _k in range(int(lap_iters_per_step)):
            lap2 = _laplacian_displacements(pts, adj)
            pts = pts + lap2 * float(lap_relax) * 0.25

    out.points = pts
    out = out.clean().triangulate()
    return out