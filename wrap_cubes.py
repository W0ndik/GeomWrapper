import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import pyvista as pv
import vtk


@dataclass
class Params:
    pitch: float
    padding: float
    band: float
    max_dim: int
    out_shell: str


def load_mesh(path: str) -> pv.PolyData:
    m = pv.read(path)
    m = m.triangulate().clean(tolerance=1e-12)
    m = m.compute_normals(
        cell_normals=False,
        point_normals=True,
        auto_orient_normals=True,
        consistent_normals=True,
        split_vertices=False,
    )
    return m


def padded_bounds(bounds, padding: float):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (xmin - padding, xmax + padding, ymin - padding, ymax + padding, zmin - padding, zmax + padding)


def choose_dims(model_bounds, pitch: float, max_dim: int):
    xmin, xmax, ymin, ymax, zmin, zmax = model_bounds
    sx = max(xmax - xmin, pitch)
    sy = max(ymax - ymin, pitch)
    sz = max(zmax - zmin, pitch)

    nx = int(math.ceil(sx / pitch))
    ny = int(math.ceil(sy / pitch))
    nz = int(math.ceil(sz / pitch))

    clamped = False
    if max(nx, ny, nz) > max_dim:
        scale = max(nx, ny, nz) / max_dim
        pitch = pitch * scale
        nx = int(math.ceil(sx / pitch))
        ny = int(math.ceil(sy / pitch))
        nz = int(math.ceil(sz / pitch))
        clamped = True

    nx = max(nx, 1)
    ny = max(ny, 1)
    nz = max(nz, 1)

    return nx, ny, nz, pitch, clamped


def sample_sdf_on_points(mesh: pv.PolyData, bounds, dims_points):
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh)

    sampler = vtk.vtkSampleFunction()
    sampler.SetImplicitFunction(imp)
    sampler.SetModelBounds(bounds)
    sampler.SetSampleDimensions(*dims_points)
    sampler.ComputeNormalsOff()
    sampler.Update()

    img = pv.wrap(sampler.GetOutput())
    scalar_name = img.array_names[0] if img.array_names else None
    if scalar_name is None:
        raise RuntimeError("No scalar array in sampled volume")
    return img, scalar_name


def point_to_cell_average(img: pv.ImageData, point_scalar_name: str) -> np.ndarray:
    dims = img.dimensions  # (nx+1, ny+1, nz+1) for points
    nxp, nyp, nzp = dims
    nx, ny, nz = nxp - 1, nyp - 1, nzp - 1

    s = np.asarray(img.point_data[point_scalar_name], dtype=np.float64)
    s = s.reshape((nxp, nyp, nzp), order="F")

    c = (
        s[:-1, :-1, :-1]
        + s[1:, :-1, :-1]
        + s[:-1, 1:, :-1]
        + s[1:, 1:, :-1]
        + s[:-1, :-1, 1:]
        + s[1:, :-1, 1:]
        + s[:-1, 1:, 1:]
        + s[1:, 1:, 1:]
    ) * 0.125

    return c  # shape (nx, ny, nz)


def compute_front(geom: np.ndarray, outside: np.ndarray) -> np.ndarray:
    nx, ny, nz = geom.shape
    out_pad = np.pad(outside, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=False)

    nb_xm = out_pad[0:nx, 1:ny+1, 1:nz+1]
    nb_xp = out_pad[2:nx+2, 1:ny+1, 1:nz+1]
    nb_ym = out_pad[1:nx+1, 0:ny, 1:nz+1]
    nb_yp = out_pad[1:nx+1, 2:ny+2, 1:nz+1]
    nb_zm = out_pad[1:nx+1, 1:ny+1, 0:nz]
    nb_zp = out_pad[1:nx+1, 1:ny+1, 2:nz+2]

    adj_out = nb_xm | nb_xp | nb_ym | nb_yp | nb_zm | nb_zp
    return geom & adj_out


def build_shell_from_front(front: np.ndarray, outside: np.ndarray, origin, pitch: float) -> pv.PolyData:
    nx, ny, nz = front.shape
    out_pad = np.pad(outside, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=True)

    def is_out(i, j, k):
        return bool(out_pad[i + 1, j + 1, k + 1])

    def pnt_key(gx, gy, gz):
        return (int(gx), int(gy), int(gz))

    def to_xyz(gx, gy, gz):
        ox, oy, oz = origin
        return (ox + gx * pitch, oy + gy * pitch, oz + gz * pitch)

    point_map = {}
    points = []
    tris = []

    def add_point(gx, gy, gz):
        key = pnt_key(gx, gy, gz)
        idx = point_map.get(key)
        if idx is not None:
            return idx
        idx = len(points)
        point_map[key] = idx
        points.append(to_xyz(gx, gy, gz))
        return idx

    def add_face_quad(tr, corners):
        a, b, c, d = corners
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

                # -X face
                if is_out(i - 1, j, k):
                    # outward normal is -X
                    add_face_quad(tris, [(gx0, gy0, gz0), (gx0, gy0, gz1), (gx0, gy1, gz1), (gx0, gy1, gz0)])
                # +X face
                if is_out(i + 1, j, k):
                    # outward normal is +X
                    add_face_quad(tris, [(gx1, gy0, gz0), (gx1, gy1, gz0), (gx1, gy1, gz1), (gx1, gy0, gz1)])
                # -Y face
                if is_out(i, j - 1, k):
                    # outward normal is -Y
                    add_face_quad(tris, [(gx0, gy0, gz0), (gx1, gy0, gz0), (gx1, gy0, gz1), (gx0, gy0, gz1)])
                # +Y face
                if is_out(i, j + 1, k):
                    # outward normal is +Y
                    add_face_quad(tris, [(gx0, gy1, gz0), (gx0, gy1, gz1), (gx1, gy1, gz1), (gx1, gy1, gz0)])
                # -Z face
                if is_out(i, j, k - 1):
                    # outward normal is -Z
                    add_face_quad(tris, [(gx0, gy0, gz0), (gx0, gy1, gz0), (gx1, gy1, gz0), (gx1, gy0, gz0)])
                # +Z face
                if is_out(i, j, k + 1):
                    # outward normal is +Z
                    add_face_quad(tris, [(gx0, gy0, gz1), (gx1, gy0, gz1), (gx1, gy1, gz1), (gx0, gy1, gz1)])

    if not points or not tris:
        return pv.PolyData()

    points = np.asarray(points, dtype=np.float64)
    faces = np.empty((len(tris), 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = np.asarray(tris, dtype=np.int64)
    shell = pv.PolyData(points, faces.reshape(-1))
    shell = shell.clean(tolerance=1e-12).triangulate()
    return shell


def visualize(mesh: pv.PolyData, vox_front: pv.DataSet, shell: pv.PolyData):
    pl = pv.Plotter()
    pl.add_text(
        "Mouse rotate pan zoom\n"
        "O toggle original\n"
        "C toggle cubes\n"
        "H toggle shell\n"
        "W toggle shell wireframe\n"
        "S save screenshot",
        font_size=10
    )

    state = {"orig": True, "cubes": True, "shell": True, "wire": True}

    a_orig = pl.add_mesh(mesh, opacity=0.25, show_edges=False)
    a_cubes = pl.add_mesh(vox_front, opacity=0.35, show_edges=True)
    a_shell = pl.add_mesh(shell, opacity=0.60, show_edges=False)
    a_wire = pl.add_mesh(shell, style="wireframe", line_width=1)

    def toggle(actor, key):
        state[key] = not state[key]
        actor.SetVisibility(1 if state[key] else 0)
        pl.render()

    pl.add_key_event("o", lambda: toggle(a_orig, "orig"))
    pl.add_key_event("c", lambda: toggle(a_cubes, "cubes"))
    pl.add_key_event("h", lambda: toggle(a_shell, "shell"))
    pl.add_key_event("w", lambda: toggle(a_wire, "wire"))

    def save_shot():
        pl.screenshot("cubes_view.png")

    pl.add_key_event("s", save_shot)
    pl.show()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--pitch", type=float, default=2.0)
    ap.add_argument("--padding", type=float, default=-1.0)
    ap.add_argument("--band", type=float, default=-1.0)
    ap.add_argument("--max-dim", type=int, default=260)
    ap.add_argument("--out-shell", type=str, default="")
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.inp):
        raise FileNotFoundError(args.inp)

    mesh = load_mesh(args.inp)

    pitch = float(args.pitch)
    padding = float(args.padding)
    if padding < 0:
        padding = 3.0 * pitch

    b = padded_bounds(mesh.bounds, padding)
    nx, ny, nz, pitch_used, clamped = choose_dims(b, pitch, int(args.max_dim))

    if args.band < 0:
        band = 0.75 * pitch_used
    else:
        band = float(args.band)

    dims_points = (nx + 1, ny + 1, nz + 1)

    img, sdf_name = sample_sdf_on_points(mesh, b, dims_points)
    img_origin = img.origin
    img_spacing = img.spacing

    sdf_cell = point_to_cell_average(img, sdf_name)  # (nx, ny, nz)

    geom = np.abs(sdf_cell) <= band
    outside = (sdf_cell > 0.0) & (~geom)
    front = compute_front(geom, outside)

    # attach cell arrays to the same image grid
    front_flat = front.astype(np.uint8).reshape(-1, order="F")
    img.cell_data["front"] = front_flat

    # extract voxels for front cubes
    vox_front = img.threshold([0.5, 1.5], scalars="front", preference="cell")

    shell = build_shell_from_front(front, outside, origin=img_origin, pitch=img_spacing[0])

    if args.out_shell:
        shell.save(args.out_shell)

    if clamped:
        print(f"Pitch was increased to {pitch_used:.6g} to keep grid dimensions under max-dim.")

    visualize(mesh, vox_front, shell)


if __name__ == "__main__":
    main()