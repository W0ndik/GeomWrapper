import numpy as np
import pyvista as pv
from app.octree import Leaf


def build_shell_from_front_leaves(
    front_leaves: list[Leaf],
    outside: np.ndarray,
    origin,
    pitch: float,
) -> pv.PolyData:
    if not front_leaves:
        return pv.PolyData()

    nx, ny, nz = outside.shape
    outside_pad = np.pad(outside, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=True)

    def is_out_cell(i, j, k) -> bool:
        return bool(outside_pad[i + 1, j + 1, k + 1])

    point_map = {}
    points = []
    tris = []

    def key(gx, gy, gz):
        return (int(gx), int(gy), int(gz))

    def to_xyz(gx, gy, gz):
        return (origin[0] + gx * pitch, origin[1] + gy * pitch, origin[2] + gz * pitch)

    def add_point(gx, gy, gz):
        k = key(gx, gy, gz)
        idx = point_map.get(k)
        if idx is not None:
            return idx
        idx = len(points)
        point_map[k] = idx
        points.append(to_xyz(gx, gy, gz))
        return idx

    def add_tri(a, b, c):
        ia = add_point(*a)
        ib = add_point(*b)
        ic = add_point(*c)
        tris.append((ia, ib, ic))

    def add_face_subgrid(face, i0, i1, j0, j1, k0, k1):
        if face == 0:
            gx = i0
            for y in range(j0, j1):
                for z in range(k0, k1):
                    add_tri((gx, y, z), (gx, y, z + 1), (gx, y + 1, z + 1))
                    add_tri((gx, y + 1, z + 1), (gx, y + 1, z), (gx, y, z))
        elif face == 1:
            gx = i1
            for y in range(j0, j1):
                for z in range(k0, k1):
                    add_tri((gx, y, z), (gx, y + 1, z), (gx, y + 1, z + 1))
                    add_tri((gx, y + 1, z + 1), (gx, y, z + 1), (gx, y, z))
        elif face == 2:
            gy = j0
            for x in range(i0, i1):
                for z in range(k0, k1):
                    add_tri((x, gy, z), (x + 1, gy, z), (x + 1, gy, z + 1))
                    add_tri((x + 1, gy, z + 1), (x, gy, z + 1), (x, gy, z))
        elif face == 3:
            gy = j1
            for x in range(i0, i1):
                for z in range(k0, k1):
                    add_tri((x, gy, z), (x, gy, z + 1), (x + 1, gy, z + 1))
                    add_tri((x + 1, gy, z + 1), (x + 1, gy, z), (x, gy, z))
        elif face == 4:
            gz = k0
            for x in range(i0, i1):
                for y in range(j0, j1):
                    add_tri((x, y, gz), (x, y + 1, gz), (x + 1, y + 1, gz))
                    add_tri((x + 1, y + 1, gz), (x + 1, y, gz), (x, y, gz))
        elif face == 5:
            gz = k1
            for x in range(i0, i1):
                for y in range(j0, j1):
                    add_tri((x, y, gz), (x + 1, y, gz), (x + 1, y + 1, gz))
                    add_tri((x + 1, y + 1, gz), (x, y + 1, gz), (x, y, gz))

    for lf in front_leaves:
        i0, j0, k0 = lf.i0, lf.j0, lf.k0
        i1 = min(i0 + lf.size, nx)
        j1 = min(j0 + lf.size, ny)
        k1 = min(k0 + lf.size, nz)

        if i0 > 0 and np.any(outside[i0 - 1, j0:j1, k0:k1]):
            add_face_subgrid(0, i0, i1, j0, j1, k0, k1)
        if i1 < nx and np.any(outside[i1, j0:j1, k0:k1]):
            add_face_subgrid(1, i0, i1, j0, j1, k0, k1)
        if j0 > 0 and np.any(outside[i0:i1, j0 - 1, k0:k1]):
            add_face_subgrid(2, i0, i1, j0, j1, k0, k1)
        if j1 < ny and np.any(outside[i0:i1, j1, k0:k1]):
            add_face_subgrid(3, i0, i1, j0, j1, k0, k1)
        if k0 > 0 and np.any(outside[i0:i1, j0:j1, k0 - 1]):
            add_face_subgrid(4, i0, i1, j0, j1, k0, k1)
        if k1 < nz and np.any(outside[i0:i1, j0:j1, k1]):
            add_face_subgrid(5, i0, i1, j0, j1, k0, k1)

    if not points or not tris:
        return pv.PolyData()

    points = np.asarray(points, dtype=np.float64)
    faces = np.empty((len(tris), 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = np.asarray(tris, dtype=np.int64)

    shell = pv.PolyData(points, faces.reshape(-1))
    return shell.clean(tolerance=1e-12).triangulate()