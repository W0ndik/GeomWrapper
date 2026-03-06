from typing import List, Set
import numpy as np
import pyvista as pv
import vtk


def build_shell_adjacency(shell: pv.PolyData) -> List[Set[int]]:
    n = shell.n_points
    adj: List[Set[int]] = [set() for _ in range(n)]
    faces = shell.faces.reshape((-1, 4))
    for f in faces:
        a, b, c = int(f[1]), int(f[2]), int(f[3])
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    return adj


def compute_min_edge_lengths(points: np.ndarray, adj: List[Set[int]]) -> np.ndarray:
    n = points.shape[0]
    minlen = np.full(n, np.inf, dtype=np.float64)
    for i in range(n):
        nei = adj[i]
        if not nei:
            minlen[i] = 0.0
            continue
        pi = points[i]
        ml = np.inf
        for j in nei:
            d = np.linalg.norm(points[j] - pi)
            if d < ml:
                ml = d
        minlen[i] = 0.0 if not np.isfinite(ml) else ml
    return minlen


def build_surface_locator(mesh: pv.PolyData) -> vtk.vtkStaticCellLocator:
    poly = mesh.extract_surface().triangulate().clean(tolerance=1e-12)
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    return locator


def closest_points(locator: vtk.vtkStaticCellLocator, pts: np.ndarray) -> np.ndarray:
    out = np.empty_like(pts)
    closest = [0.0, 0.0, 0.0]
    cid = vtk.mutable(0)
    sid = vtk.mutable(0)
    dist2 = vtk.mutable(0.0)
    for i in range(pts.shape[0]):
        p = pts[i]
        locator.FindClosestPoint(p.tolist(), closest, cid, sid, dist2)
        out[i, 0] = closest[0]
        out[i, 1] = closest[1]
        out[i, 2] = closest[2]
    return out


def laplacian_displacements(points: np.ndarray, adj: List[Set[int]]) -> np.ndarray:
    disp = np.zeros_like(points)
    for i in range(points.shape[0]):
        nei = adj[i]
        if not nei:
            continue
        avg = np.mean(points[list(nei)], axis=0)
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
) -> pv.PolyData:
    if iters <= 0:
        return shell

    out = shell.copy(deep=True).triangulate().clean(tolerance=1e-12)
    adj = build_shell_adjacency(out)
    locator = build_surface_locator(target_mesh)

    pts = np.asarray(out.points, dtype=np.float64)

    for _ in range(int(iters)):
        min_edge = compute_min_edge_lengths(pts, adj)
        constraints = min_edge * float(constraint_mul)

        cp = closest_points(locator, pts)
        disp = (cp - pts) * float(step)

        lens = np.linalg.norm(disp, axis=1)
        mask = lens > 1e-15
        scale = np.ones_like(lens)
        scale[mask] = np.minimum(1.0, constraints[mask] / lens[mask])
        disp = disp * scale[:, None]

        pts = pts + disp

        for _k in range(int(lap_iters_per_step)):
            lap = laplacian_displacements(pts, adj)
            pts = pts + lap * float(lap_relax)

        out.points = pts
        out = out.clean(tolerance=1e-12).triangulate()
        pts = np.asarray(out.points, dtype=np.float64)

    return out