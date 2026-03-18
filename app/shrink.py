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
        if not adj[i]:
            out[i] = 0.0
            continue

        pi = points[i]
        vals = [np.linalg.norm(points[j] - pi) for j in adj[i]]
        out[i] = float(min(vals)) if vals else 0.0

    return out


def build_surface_locator(mesh: pv.PolyData) -> vtk.vtkStaticCellLocator:
    poly = mesh.extract_surface().triangulate().clean()
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


def _build_edge_to_tris(tris: List[Tri]) -> Dict[Edge, List[int]]:
    out: Dict[Edge, List[int]] = {}
    for ti, (a, b, c) in enumerate(tris):
        for e in (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))):
            out.setdefault(e, []).append(ti)
    return out


def _triangle_area_vec(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> np.ndarray:
    return np.cross(pb - pa, pc - pa)


def _triangle_area(points: np.ndarray, tri: Tri) -> float:
    a, b, c = tri
    av = _triangle_area_vec(points[a], points[b], points[c])
    return 0.5 * float(np.linalg.norm(av))


def _triangle_unit_normal(points: np.ndarray, tri: Tri) -> np.ndarray:
    a, b, c = tri
    n = _triangle_area_vec(points[a], points[b], points[c])
    ln = float(np.linalg.norm(n))
    if ln == 0.0:
        return np.zeros(3, dtype=np.float64)
    return n / ln


def _triangle_heights(points: np.ndarray, tri: Tri) -> np.ndarray:
    a, b, c = tri
    pa = points[a]
    pb = points[b]
    pc = points[c]

    lab = float(np.linalg.norm(pb - pa))
    lbc = float(np.linalg.norm(pc - pb))
    lca = float(np.linalg.norm(pa - pc))
    area = _triangle_area(points, tri)

    ha = 0.0 if lbc == 0.0 else 2.0 * area / lbc
    hb = 0.0 if lca == 0.0 else 2.0 * area / lca
    hc = 0.0 if lab == 0.0 else 2.0 * area / lab

    return np.array([ha, hb, hc], dtype=np.float64)


def _dihedral_angle(points: np.ndarray, tri0: Tri, tri1: Tri) -> float:
    n0 = _triangle_unit_normal(points, tri0)
    n1 = _triangle_unit_normal(points, tri1)

    ln0 = float(np.linalg.norm(n0))
    ln1 = float(np.linalg.norm(n1))
    if ln0 == 0.0 or ln1 == 0.0:
        return 0.0

    cs = float(np.dot(n0, n1))
    cs = max(-1.0, min(1.0, cs))
    return float(np.arccos(cs))


def _laplacian_displacements(points: np.ndarray, adj: List[Set[int]]) -> np.ndarray:
    disp = np.zeros_like(points)
    for i in range(points.shape[0]):
        nei = list(adj[i])
        if not nei:
            continue
        avg = np.mean(points[nei], axis=0)
        disp[i] = avg - points[i]
    return disp


def _segment_triangle_intersection(
    p0: np.ndarray,
    p1: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
) -> bool:
    d = p1 - p0
    e1 = t1 - t0
    e2 = t2 - t0
    h = np.cross(d, e2)
    a = float(np.dot(e1, h))

    if a == 0.0:
        return False

    f = 1.0 / a
    s = p0 - t0
    u = f * float(np.dot(s, h))
    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, e1)
    v = f * float(np.dot(d, q))
    if v < 0.0 or (u + v) > 1.0:
        return False

    t = f * float(np.dot(e2, q))
    return 0.0 <= t <= 1.0


def _build_vertex_to_tris(n_points: int, tris: List[Tri]) -> List[Set[int]]:
    out: List[Set[int]] = [set() for _ in range(n_points)]
    for ti, (a, b, c) in enumerate(tris):
        out[a].add(ti)
        out[b].add(ti)
        out[c].add(ti)
    return out


def _is_dangerous_displacement(
    vid: int,
    old_pts: np.ndarray,
    new_pt: np.ndarray,
    tris: List[Tri],
    vertex_to_tris: List[Set[int]],
) -> bool:
    p0 = old_pts[vid]
    p1 = new_pt
    incident = vertex_to_tris[vid]

    for ti, tri in enumerate(tris):
        if ti in incident:
            continue
        a, b, c = tri
        if _segment_triangle_intersection(p0, p1, old_pts[a], old_pts[b], old_pts[c]):
            return True

    return False


def _energy_edges(points: np.ndarray, edges: List[Edge], ref_edge_len: Dict[Edge, float]) -> float:
    e = 0.0
    for i, j in edges:
        l = float(np.linalg.norm(points[j] - points[i]))
        l0 = ref_edge_len[(i, j)]
        d = l - l0
        e += 0.5 * d * d
    return e


def _energy_heights(points: np.ndarray, tris: List[Tri], ref_heights: Dict[Tuple[int, int], float]) -> float:
    e = 0.0
    for ti, tri in enumerate(tris):
        h = _triangle_heights(points, tri)
        for local_idx in range(3):
            d = float(h[local_idx] - ref_heights[(ti, local_idx)])
            e += 0.5 * d * d
    return e


def _energy_bending(
    points: np.ndarray,
    tris: List[Tri],
    edge_to_tris: Dict[Edge, List[int]],
    ref_dihedral: Dict[Edge, float],
) -> float:
    e = 0.0
    for edge, tlist in edge_to_tris.items():
        if len(tlist) != 2:
            continue
        t0, t1 = tlist
        phi = _dihedral_angle(points, tris[t0], tris[t1])
        phi0 = ref_dihedral[edge]
        d = phi - phi0
        e += 0.5 * d * d
    return e


def _vertex_energy_gradient(
    points: np.ndarray,
    vid: int,
    energy_fn,
    h: float,
) -> np.ndarray:
    grad = np.zeros(3, dtype=np.float64)

    for ax in range(3):
        p_plus = points.copy()
        p_minus = points.copy()

        p_plus[vid, ax] += h
        p_minus[vid, ax] -= h

        e_plus = float(energy_fn(p_plus))
        e_minus = float(energy_fn(p_minus))

        grad[ax] = (e_plus - e_minus) / (2.0 * h)

    return grad


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

    out = shell.copy(deep=True).triangulate().clean()
    if out.n_points == 0 or out.n_cells == 0:
        return out

    pts = np.asarray(out.points, dtype=np.float64).copy()
    tris = _build_triangles(out)
    edges = _build_unique_edges(tris)
    edge_to_tris = _build_edge_to_tris(tris)
    adj = build_shell_adjacency(out)
    vertex_to_tris = _build_vertex_to_tris(out.n_points, tris)

    locator = build_surface_locator(target_mesh)

    ref_pts = pts.copy()

    ref_edge_len: Dict[Edge, float] = {}
    for i, j in edges:
        ref_edge_len[(i, j)] = float(np.linalg.norm(ref_pts[j] - ref_pts[i]))

    ref_heights: Dict[Tuple[int, int], float] = {}
    for ti, tri in enumerate(tris):
        hvec = _triangle_heights(ref_pts, tri)
        for local_idx in range(3):
            ref_heights[(ti, local_idx)] = float(hvec[local_idx])

    ref_dihedral: Dict[Edge, float] = {}
    for edge, tlist in edge_to_tris.items():
        if len(tlist) == 2:
            t0, t1 = tlist
            ref_dihedral[edge] = _dihedral_angle(ref_pts, tris[t0], tris[t1])

    k_edge = 1.0
    k_height = 0.30
    k_bend = 0.12
    k_attr = 1.20

    grad_h = max(step * 0.1, 1e-4)

    edge_energy_fn = lambda p: _energy_edges(p, edges, ref_edge_len)
    height_energy_fn = lambda p: _energy_heights(p, tris, ref_heights)
    bend_energy_fn = lambda p: _energy_bending(p, tris, edge_to_tris, ref_dihedral)

    for _ in range(int(iters)):
        forces = np.zeros_like(pts)

        cp = closest_points(locator, pts)
        attr = cp - pts

        attr_norm = np.linalg.norm(attr, axis=1)
        attr_dir = np.zeros_like(attr)
        nz = attr_norm > 0.0
        attr_dir[nz] = attr[nz] / attr_norm[nz][:, None]

        ambiguous = np.zeros(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            nei = list(adj[i])
            if len(nei) < 2:
                continue

            base = attr_dir[i]
            if float(np.linalg.norm(base)) == 0.0:
                continue

            dots = []
            for j in nei:
                dj = attr_dir[j]
                if float(np.linalg.norm(dj)) == 0.0:
                    continue
                dots.append(float(np.dot(base, dj)))

            if dots and min(dots) < -0.25:
                ambiguous[i] = True

        candidate_pts = pts + step * attr
        dangerous = np.zeros(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            if attr_norm[i] == 0.0:
                continue
            dangerous[i] = _is_dangerous_displacement(i, pts, candidate_pts[i], tris, vertex_to_tris)

        lap = _laplacian_displacements(pts, adj)

        for vid in range(pts.shape[0]):
            g_edge = _vertex_energy_gradient(pts, vid, edge_energy_fn, grad_h)
            g_height = _vertex_energy_gradient(pts, vid, height_energy_fn, grad_h)
            g_bend = _vertex_energy_gradient(pts, vid, bend_energy_fn, grad_h)

            forces[vid] -= k_edge * g_edge
            forces[vid] -= k_height * g_height
            forces[vid] -= k_bend * g_bend

            if ambiguous[vid] or dangerous[vid]:
                forces[vid] += k_attr * lap_relax * lap[vid]
            else:
                forces[vid] += k_attr * attr[vid]

        disp = step * forces

        min_edge = compute_min_edge_lengths(pts, adj)
        max_shift = min_edge * float(constraint_mul)

        disp_len = np.linalg.norm(disp, axis=1)
        scale = np.ones_like(disp_len)
        mask = disp_len > 0.0
        scale[mask] = np.minimum(1.0, max_shift[mask] / disp_len[mask])

        pts = pts + disp * scale[:, None]

        for _k in range(int(lap_iters_per_step)):
            lap2 = _laplacian_displacements(pts, adj)
            pts = pts + lap2 * float(lap_relax) * 0.25

    out.points = pts
    out = out.clean().triangulate()
    return out