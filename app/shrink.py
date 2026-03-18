from typing import Dict, List, Set, Tuple

import numpy as np
import pyvista as pv
import vtk


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


def _safe_normalize(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lens = np.linalg.norm(v, axis=1)
    out = np.zeros_like(v)
    mask = lens > 1e-15
    out[mask] = v[mask] / lens[mask][:, None]
    return out, lens


def _triangle_area_normal(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return np.cross(p1 - p0, p2 - p0)


def _triangle_area(points: np.ndarray, tri: Tuple[int, int, int]) -> float:
    a, b, c = tri
    n = _triangle_area_normal(points[a], points[b], points[c])
    return 0.5 * float(np.linalg.norm(n))


def _triangle_unit_normal(points: np.ndarray, tri: Tuple[int, int, int]) -> np.ndarray:
    a, b, c = tri
    n = _triangle_area_normal(points[a], points[b], points[c])
    ln = np.linalg.norm(n)
    if ln <= 1e-15:
        return np.zeros(3, dtype=np.float64)
    return n / ln


def _build_triangles(shell: pv.PolyData) -> List[Tuple[int, int, int]]:
    faces = shell.faces.reshape((-1, 4))
    tris: List[Tuple[int, int, int]] = []
    for f in faces:
        tris.append((int(f[1]), int(f[2]), int(f[3])))
    return tris


def _build_unique_edges(tris: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    edges = set()
    for a, b, c in tris:
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return sorted(edges)


def _build_edge_to_tris(tris: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], List[int]]:
    edge_to_tris: Dict[Tuple[int, int], List[int]] = {}
    for ti, (a, b, c) in enumerate(tris):
        for e in (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))):
            edge_to_tris.setdefault(e, []).append(ti)
    return edge_to_tris


def _triangle_heights(points: np.ndarray, tri: Tuple[int, int, int]) -> np.ndarray:
    a, b, c = tri
    pa = points[a]
    pb = points[b]
    pc = points[c]

    lab = np.linalg.norm(pb - pa)
    lbc = np.linalg.norm(pc - pb)
    lca = np.linalg.norm(pa - pc)

    area = _triangle_area(points, tri)

    ha = 0.0 if lbc <= 1e-15 else (2.0 * area / lbc)
    hb = 0.0 if lca <= 1e-15 else (2.0 * area / lca)
    hc = 0.0 if lab <= 1e-15 else (2.0 * area / lab)

    return np.array([ha, hb, hc], dtype=np.float64)


def _triangle_height_for_vertex(points: np.ndarray, tri: Tuple[int, int, int], local_idx: int) -> float:
    return float(_triangle_heights(points, tri)[local_idx])


def _dihedral_angle(points: np.ndarray, tri1: Tuple[int, int, int], tri2: Tuple[int, int, int]) -> float:
    n1 = _triangle_unit_normal(points, tri1)
    n2 = _triangle_unit_normal(points, tri2)

    ln1 = np.linalg.norm(n1)
    ln2 = np.linalg.norm(n2)
    if ln1 <= 1e-15 or ln2 <= 1e-15:
        return 0.0

    c = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    return float(np.arccos(c))


def _laplacian_displacements(points: np.ndarray, adj: List[Set[int]]) -> np.ndarray:
    disp = np.zeros_like(points)
    for i in range(points.shape[0]):
        nei = adj[i]
        if not nei:
            continue
        avg = np.mean(points[list(nei)], axis=0)
        disp[i] = avg - points[i]
    return disp


def _segment_triangle_intersection(
    p0: np.ndarray,
    p1: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
) -> bool:
    eps = 1e-12
    d = p1 - p0
    e1 = t1 - t0
    e2 = t2 - t0
    h = np.cross(d, e2)
    a = np.dot(e1, h)

    if -eps < a < eps:
        return False

    f = 1.0 / a
    s = p0 - t0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, e1)
    v = f * np.dot(d, q)
    if v < 0.0 or (u + v) > 1.0:
        return False

    t = f * np.dot(e2, q)
    return 0.0 <= t <= 1.0


def _build_vertex_to_tris(n_points: int, tris: List[Tuple[int, int, int]]) -> List[Set[int]]:
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
    tris: List[Tuple[int, int, int]],
    vertex_to_tris: List[Set[int]],
) -> bool:
    p0 = old_pts[vid]
    p1 = new_pt

    incident = vertex_to_tris[vid]
    for ti, tri in enumerate(tris):
        if ti in incident:
            continue
        a, b, c = tri
        t0 = old_pts[a]
        t1 = old_pts[b]
        t2 = old_pts[c]
        if _segment_triangle_intersection(p0, p1, t0, t1, t2):
            return True

    return False


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
    if out.n_points == 0 or out.n_cells == 0:
        return out

    adj = build_shell_adjacency(out)
    locator = build_surface_locator(target_mesh)
    pts = np.asarray(out.points, dtype=np.float64).copy()

    tris = _build_triangles(out)
    edges = _build_unique_edges(tris)
    edge_to_tris = _build_edge_to_tris(tris)
    vertex_to_tris = _build_vertex_to_tris(out.n_points, tris)

    ref_pts = pts.copy()

    ref_edge_len: Dict[Tuple[int, int], float] = {}
    for i, j in edges:
        ref_edge_len[(i, j)] = float(np.linalg.norm(ref_pts[j] - ref_pts[i]))

    ref_heights: Dict[Tuple[int, int], float] = {}
    for ti, tri in enumerate(tris):
        h = _triangle_heights(ref_pts, tri)
        ref_heights[(ti, 0)] = float(h[0])
        ref_heights[(ti, 1)] = float(h[1])
        ref_heights[(ti, 2)] = float(h[2])

    ref_dihedral: Dict[Tuple[int, int], float] = {}
    for e, tlist in edge_to_tris.items():
        if len(tlist) == 2:
            t0, t1 = tlist
            ref_dihedral[e] = _dihedral_angle(ref_pts, tris[t0], tris[t1])

    # коэффициенты жёсткости
    k_edge = 1.0
    k_height = 0.35
    k_bend = 0.10
    k_attr = 1.25

    critical_angle = np.deg2rad(8.0)

    for _ in range(int(iters)):
        forces = np.zeros_like(pts)

        # 1. Силы растяжения/сжатия рёбер по закону Гука
        for i, j in edges:
            e = pts[j] - pts[i]
            l = float(np.linalg.norm(e))
            if l <= 1e-15:
                continue

            l0 = ref_edge_len[(i, j)]
            dir_ij = e / l
            f = k_edge * (l - l0) * dir_ij

            forces[i] += f
            forces[j] -= f

        # 2. Силы сохранения высоты треугольников
        # Реализация через численный градиент отклонения высоты от эталонной.
        # Это тяжелее, чем закрытая формула, зато не ломает индексацию и соответствует смыслу модели.
        eps = 1e-6
        for ti, tri in enumerate(tris):
            ids = [tri[0], tri[1], tri[2]]
            for local_idx, vid in enumerate(ids):
                h_cur = _triangle_height_for_vertex(pts, tri, local_idx)
                h_ref = ref_heights[(ti, local_idx)]
                diff0 = h_cur - h_ref

                grad = np.zeros(3, dtype=np.float64)
                for ax in range(3):
                    tmp = pts.copy()
                    tmp[vid, ax] += eps
                    h_eps = _triangle_height_for_vertex(tmp, tri, local_idx)
                    diff_eps = h_eps - h_ref
                    energy_eps = 0.5 * diff_eps * diff_eps
                    energy_0 = 0.5 * diff0 * diff0
                    grad[ax] = (energy_eps - energy_0) / eps

                forces[vid] -= k_height * grad

        # 3. Силы изгиба по внутренним рёбрам через двугранный угол
        # Упрощённо распределяем по четырём вершинам пары смежных треугольников.
        for e, tlist in edge_to_tris.items():
            if len(tlist) != 2:
                continue

            i, j = e
            t0, t1 = tlist
            tri0 = tris[t0]
            tri1 = tris[t1]

            phi = _dihedral_angle(pts, tri0, tri1)
            phi0 = ref_dihedral.get(e, phi)
            dphi = phi - phi0

            if abs(phi0) < critical_angle and abs(phi) < critical_angle:
                continue

            allv = list(set([tri0[0], tri0[1], tri0[2], tri1[0], tri1[1], tri1[2]]))
            if len(allv) < 4:
                continue

            edge_vec = pts[j] - pts[i]
            edge_len = float(np.linalg.norm(edge_vec))
            if edge_len <= 1e-15:
                continue

            edge_dir = edge_vec / edge_len
            n0 = _triangle_unit_normal(pts, tri0)
            n1 = _triangle_unit_normal(pts, tri1)

            bend_dir0 = np.cross(edge_dir, n0)
            bend_dir1 = np.cross(edge_dir, n1)

            # вершины, не лежащие на общем ребре
            k0 = [v for v in tri0 if v not in e]
            k1 = [v for v in tri1 if v not in e]
            if not k0 or not k1:
                continue

            v0 = k0[0]
            v1 = k1[0]

            fb0 = -k_bend * dphi * bend_dir0
            fb1 =  k_bend * dphi * bend_dir1

            forces[v0] += fb0
            forces[v1] += fb1
            forces[i] -= 0.5 * (fb0 + fb1)
            forces[j] -= 0.5 * (fb0 + fb1)

        # 4. Силы притяжения к исходной геометрии
        cp = closest_points(locator, pts)
        attr = cp - pts

        # проверка неоднозначных/опасных смещений
        # неоднозначность грубо детектируем по сильному разбросу направлений у соседей
        attr_dir, attr_len = _safe_normalize(attr)
        ambiguous = np.zeros(pts.shape[0], dtype=bool)

        for i in range(pts.shape[0]):
            nei = list(adj[i])
            if len(nei) < 2:
                continue

            base = attr_dir[i]
            if np.linalg.norm(base) <= 1e-15:
                continue

            dots = []
            for j in nei:
                dj = attr_dir[j]
                if np.linalg.norm(dj) <= 1e-15:
                    continue
                dots.append(float(np.dot(base, dj)))

            if dots and min(dots) < -0.25:
                ambiguous[i] = True

        candidate_pts = pts + step * attr
        dangerous = np.zeros(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            if attr_len[i] <= 1e-15:
                continue
            dangerous[i] = _is_dangerous_displacement(i, pts, candidate_pts[i], tris, vertex_to_tris)

        # для неоднозначных и опасных вершин вместо прямого притяжения используем локальную регуляризацию
        lap = _laplacian_displacements(pts, adj)
        for i in range(pts.shape[0]):
            if ambiguous[i] or dangerous[i]:
                forces[i] += k_attr * lap_relax * lap[i]
            else:
                forces[i] += k_attr * attr[i]

        # 5. Ограничение смещения по локальной шкале
        disp = step * forces
        min_edge = compute_min_edge_lengths(pts, adj)
        max_shift = min_edge * float(constraint_mul)

        disp_len = np.linalg.norm(disp, axis=1)
        mask = disp_len > 1e-15
        scale = np.ones_like(disp_len)
        scale[mask] = np.minimum(1.0, max_shift[mask] / disp_len[mask])
        disp = disp * scale[:, None]

        # 6. Обновление
        pts = pts + disp

        # 7. Дополнительная мягкая релаксация как численная стабилизация
        for _k in range(int(lap_iters_per_step)):
            lap2 = _laplacian_displacements(pts, adj)
            pts = pts + lap2 * float(lap_relax) * 0.25

    out.points = pts
    out = out.clean(tolerance=1e-12).triangulate()
    return out