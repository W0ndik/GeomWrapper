import numpy as np
import pyvista as pv


def build_shell_from_front(front: np.ndarray, outside: np.ndarray, origin, pitch: float) -> pv.PolyData:
    nx, ny, nz = front.shape
    out_pad = np.pad(outside, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=True)

    def is_out(i, j, k):
        return bool(out_pad[i + 1, j + 1, k + 1])

    def key(gx, gy, gz):
        return (int(gx), int(gy), int(gz))

    def to_xyz(gx, gy, gz):
        ox, oy, oz = origin
        return (ox + gx * pitch, oy + gy * pitch, oz + gz * pitch)

    point_map = {}
    points = []
    tris = []

    def add_point(gx, gy, gz):
        k = key(gx, gy, gz)
        idx = point_map.get(k)
        if idx is not None:
            return idx
        idx = len(points)
        point_map[k] = idx
        points.append(to_xyz(gx, gy, gz))
        return idx

    def add_quad(a, b, c, d):
        ia = add_point(*a)
        ib = add_point(*b)
        ic = add_point(*c)
        idd = add_point(*d)
        tris.append((ia, ib, ic))
        tris.append((ic, idd, ia))

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not front[i, j, k]:
                    continue

                gx0, gx1 = i, i + 1
                gy0, gy1 = j, j + 1
                gz0, gz1 = k, k + 1

                if is_out(i - 1, j, k):
                    add_quad((gx0, gy0, gz0), (gx0, gy0, gz1), (gx0, gy1, gz1), (gx0, gy1, gz0))
                if is_out(i + 1, j, k):
                    add_quad((gx1, gy0, gz0), (gx1, gy1, gz0), (gx1, gy1, gz1), (gx1, gy0, gz1))
                if is_out(i, j - 1, k):
                    add_quad((gx0, gy0, gz0), (gx1, gy0, gz0), (gx1, gy0, gz1), (gx0, gy0, gz1))
                if is_out(i, j + 1, k):
                    add_quad((gx0, gy1, gz0), (gx0, gy1, gz1), (gx1, gy1, gz1), (gx1, gy1, gz0))
                if is_out(i, j, k - 1):
                    add_quad((gx0, gy0, gz0), (gx0, gy1, gz0), (gx1, gy1, gz0), (gx1, gy0, gz0))
                if is_out(i, j, k + 1):
                    add_quad((gx0, gy0, gz1), (gx1, gy0, gz1), (gx1, gy1, gz1), (gx0, gy1, gz1))

    if not points or not tris:
        return pv.PolyData()

    points = np.asarray(points, dtype=np.float64)
    faces = np.empty((len(tris), 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = np.asarray(tris, dtype=np.int64)

    shell = pv.PolyData(points, faces.reshape(-1))
    shell = shell.clean(tolerance=1e-12).triangulate()
    return shell