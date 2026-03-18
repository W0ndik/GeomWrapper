from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pyvista as pv
import vtk

from app.octree import Leaf


GridPt = Tuple[int, int, int]
UVPt = Tuple[int, int]
Seg2D = Tuple[UVPt, UVPt]


def build_shell_from_front_leaves(
    front_leaves: list[Leaf],
    outside: np.ndarray,
    origin,
    pitch: float,
) -> pv.PolyData:
    if not front_leaves:
        return pv.PolyData()

    nx, ny, nz = outside.shape

    point_map: Dict[GridPt, int] = {}
    points_xyz: List[Tuple[float, float, float]] = []
    tris: List[Tuple[int, int, int]] = []

    def key3(gx: int, gy: int, gz: int) -> GridPt:
        return (int(gx), int(gy), int(gz))

    def to_xyz(gx: int, gy: int, gz: int) -> Tuple[float, float, float]:
        return (
            origin[0] + gx * pitch,
            origin[1] + gy * pitch,
            origin[2] + gz * pitch,
        )

    def add_point3(gx: int, gy: int, gz: int) -> int:
        k = key3(gx, gy, gz)
        idx = point_map.get(k)
        if idx is not None:
            return idx

        idx = len(points_xyz)
        point_map[k] = idx
        points_xyz.append(to_xyz(gx, gy, gz))
        return idx

    def add_tri3(a: GridPt, b: GridPt, c: GridPt):
        ia = add_point3(*a)
        ib = add_point3(*b)
        ic = add_point3(*c)

        if ia == ib or ib == ic or ia == ic:
            return

        tris.append((ia, ib, ic))

    def canon_seg2(a: UVPt, b: UVPt) -> Seg2D:
        return (a, b) if a <= b else (b, a)

    def square_edges_2d(u: int, v: int) -> List[Tuple[UVPt, UVPt]]:
        p00 = (u, v)
        p10 = (u + 1, v)
        p11 = (u + 1, v + 1)
        p01 = (u, v + 1)
        return [
            (p00, p10),
            (p10, p11),
            (p11, p01),
            (p01, p00),
        ]

    def face_uv_to_grid(face_id: int, fixed: int, u: int, v: int) -> GridPt:
        if face_id in (0, 1):
            return (fixed, u, v)
        if face_id in (2, 3):
            return (u, fixed, v)
        return (u, v, fixed)

    def open_face_unit_squares(lf: Leaf, face_id: int) -> Tuple[int, List[Tuple[int, int]]]:
        i0, j0, k0 = lf.i0, lf.j0, lf.k0
        i1 = min(i0 + lf.size, nx)
        j1 = min(j0 + lf.size, ny)
        k1 = min(k0 + lf.size, nz)

        squares: List[Tuple[int, int]] = []

        if face_id == 0:
            if i0 <= 0:
                return i0, []
            fixed = i0
            for y in range(j0, j1):
                for z in range(k0, k1):
                    if outside[i0 - 1, y, z]:
                        squares.append((y, z))
            return fixed, squares

        if face_id == 1:
            if i1 >= nx:
                return i1, []
            fixed = i1
            for y in range(j0, j1):
                for z in range(k0, k1):
                    if outside[i1, y, z]:
                        squares.append((y, z))
            return fixed, squares

        if face_id == 2:
            if j0 <= 0:
                return j0, []
            fixed = j0
            for x in range(i0, i1):
                for z in range(k0, k1):
                    if outside[x, j0 - 1, z]:
                        squares.append((x, z))
            return fixed, squares

        if face_id == 3:
            if j1 >= ny:
                return j1, []
            fixed = j1
            for x in range(i0, i1):
                for z in range(k0, k1):
                    if outside[x, j1, z]:
                        squares.append((x, z))
            return fixed, squares

        if face_id == 4:
            if k0 <= 0:
                return k0, []
            fixed = k0
            for x in range(i0, i1):
                for y in range(j0, j1):
                    if outside[x, y, k0 - 1]:
                        squares.append((x, y))
            return fixed, squares

        if k1 >= nz:
            return k1, []

        fixed = k1
        for x in range(i0, i1):
            for y in range(j0, j1):
                if outside[x, y, k1]:
                    squares.append((x, y))
        return fixed, squares

    def build_patch_loops(square_uv: List[Tuple[int, int]]) -> List[List[UVPt]]:
        if not square_uv:
            return []

        edge_count: Dict[Seg2D, int] = defaultdict(int)
        edge_dir: Dict[Seg2D, Tuple[UVPt, UVPt]] = {}

        for u, v in square_uv:
            for a, b in square_edges_2d(u, v):
                k = canon_seg2(a, b)
                edge_count[k] += 1
                if k not in edge_dir:
                    edge_dir[k] = (a, b)

        boundary_edges: List[Tuple[UVPt, UVPt]] = []
        for k, cnt in edge_count.items():
            if cnt == 1:
                boundary_edges.append(edge_dir[k])

        if not boundary_edges:
            return []

        undirected_neighbors: Dict[UVPt, List[UVPt]] = defaultdict(list)
        for a, b in boundary_edges:
            undirected_neighbors[a].append(b)
            undirected_neighbors[b].append(a)

        used: Set[Seg2D] = set()
        loops: List[List[UVPt]] = []

        for a, b in boundary_edges:
            ek = canon_seg2(a, b)
            if ek in used:
                continue

            loop = [a]
            prev = a
            cur = b
            used.add(ek)

            guard = 0
            while guard < 100000:
                guard += 1
                loop.append(cur)

                nbrs = undirected_neighbors[cur]
                next_pt = None

                for cand in nbrs:
                    if cand == prev:
                        continue
                    cand_key = canon_seg2(cur, cand)
                    if cand_key not in used:
                        next_pt = cand
                        break

                if next_pt is None:
                    if cur == loop[0]:
                        break

                    for cand in nbrs:
                        cand_key = canon_seg2(cur, cand)
                        if cand_key not in used:
                            next_pt = cand
                            break

                if next_pt is None:
                    break

                prev, cur = cur, next_pt
                used.add(canon_seg2(prev, cur))

                if cur == loop[0]:
                    break

            if len(loop) >= 4 and loop[0] == loop[-1]:
                loops.append(loop[:-1])

        return loops

    def signed_area_2d(loop: List[UVPt]) -> float:
        s = 0.0
        n = len(loop)
        for i in range(n):
            x1, y1 = loop[i]
            x2, y2 = loop[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return 0.5 * s

    def add_tri_oriented(face_id: int, a: GridPt, b: GridPt, c: GridPt):
        pa = np.array(a, dtype=np.float64)
        pb = np.array(b, dtype=np.float64)
        pc = np.array(c, dtype=np.float64)

        n = np.cross(pb - pa, pc - pa)

        if face_id == 0:
            want = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        elif face_id == 1:
            want = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif face_id == 2:
            want = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        elif face_id == 3:
            want = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        elif face_id == 4:
            want = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            want = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        if float(np.dot(n, want)) < 0.0:
            add_tri3(a, c, b)
        else:
            add_tri3(a, b, c)

    def triangulate_patch(face_id: int, fixed: int, loops_uv: List[List[UVPt]]):
        if not loops_uv:
            return

        # VTK contour triangulator ожидает внешний контур и, при наличии, внутренние.
        # Для ортогональных loop этого обычно хватает, если дать замкнутые polyline.
        loops_uv_sorted = sorted(loops_uv, key=lambda lp: abs(signed_area_2d(lp)), reverse=True)

        vtk_points = vtk.vtkPoints()
        vtk_lines = vtk.vtkCellArray()
        local_map: Dict[GridPt, int] = {}

        def get_local_id(gp: GridPt) -> int:
            lid = local_map.get(gp)
            if lid is not None:
                return lid

            lid = vtk_points.InsertNextPoint(*to_xyz(*gp))
            local_map[gp] = lid
            return lid

        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_points)

        for loop in loops_uv_sorted:
            if len(loop) < 3:
                continue

            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(len(loop) + 1)

            for idx, uv in enumerate(loop):
                gp = face_uv_to_grid(face_id, fixed, uv[0], uv[1])
                pid = get_local_id(gp)
                polyline.GetPointIds().SetId(idx, pid)

            gp0 = face_uv_to_grid(face_id, fixed, loop[0][0], loop[0][1])
            polyline.GetPointIds().SetId(len(loop), get_local_id(gp0))
            vtk_lines.InsertNextCell(polyline)

        poly.SetLines(vtk_lines)

        triangulator = vtk.vtkContourTriangulator()
        triangulator.SetInputData(poly)
        triangulator.Update()

        out = triangulator.GetOutput()
        if out is None or out.GetNumberOfCells() == 0:
            return

        reverse_map = {lid: gp for gp, lid in local_map.items()}

        for ci in range(out.GetNumberOfCells()):
            cell = out.GetCell(ci)
            if cell is None or cell.GetNumberOfPoints() != 3:
                continue

            p0 = cell.GetPointId(0)
            p1 = cell.GetPointId(1)
            p2 = cell.GetPointId(2)

            g0 = reverse_map[p0]
            g1 = reverse_map[p1]
            g2 = reverse_map[p2]

            add_tri_oriented(face_id, g0, g1, g2)

    for lf in front_leaves:
        for face_id in range(6):
            fixed, square_uv = open_face_unit_squares(lf, face_id)
            if not square_uv:
                continue

            loops_uv = build_patch_loops(square_uv)
            if not loops_uv:
                continue

            triangulate_patch(face_id, fixed, loops_uv)

    if not points_xyz or not tris:
        return pv.PolyData()

    points = np.asarray(points_xyz, dtype=np.float64)
    faces = np.empty((len(tris), 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = np.asarray(tris, dtype=np.int64)

    shell = pv.PolyData(points, faces.reshape(-1))
    return shell.clean(tolerance=1e-12).triangulate()