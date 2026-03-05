import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import pyvista as pv
import vtk


@dataclass
class WrapParams:
    pitch: float
    offset: float
    padding: float
    smooth_iters: int
    smooth_relax: float
    fill_holes: float
    decimate: float


def load_stl(path: str) -> pv.PolyData:
    mesh = pv.read(path)
    mesh = mesh.triangulate()
    mesh = mesh.clean(tolerance=1e-12)
    mesh = mesh.compute_normals(
        cell_normals=False,
        point_normals=True,
        auto_orient_normals=True,
        consistent_normals=True,
        split_vertices=False,
    )
    return mesh


def try_fill_holes(mesh: pv.PolyData, hole_size: float) -> pv.PolyData:
    if hole_size <= 0:
        return mesh
    try:
        out = mesh.fill_holes(hole_size)
        out = out.triangulate().clean(tolerance=1e-12)
        return out
    except Exception:
        return mesh


def compute_bounds_with_padding(bounds, padding: float):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (
        xmin - padding,
        xmax + padding,
        ymin - padding,
        ymax + padding,
        zmin - padding,
        zmax + padding,
    )


def choose_dims(model_bounds, pitch: float, max_dim: int = 420):
    xmin, xmax, ymin, ymax, zmin, zmax = model_bounds
    sx = max(xmax - xmin, pitch)
    sy = max(ymax - ymin, pitch)
    sz = max(zmax - zmin, pitch)

    nx = int(math.ceil(sx / pitch)) + 1
    ny = int(math.ceil(sy / pitch)) + 1
    nz = int(math.ceil(sz / pitch)) + 1

    if max(nx, ny, nz) > max_dim:
        scale = max(nx, ny, nz) / max_dim
        pitch2 = pitch * scale
        nx = int(math.ceil(sx / pitch2)) + 1
        ny = int(math.ceil(sy / pitch2)) + 1
        nz = int(math.ceil(sz / pitch2)) + 1
        return nx, ny, nz, pitch2, True

    return nx, ny, nz, pitch, False


def sample_signed_distance(mesh: pv.PolyData, pitch: float, padding: float):
    b = compute_bounds_with_padding(mesh.bounds, padding)
    nx, ny, nz, pitch_used, was_clamped = choose_dims(b, pitch)

    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh)

    sampler = vtk.vtkSampleFunction()
    sampler.SetImplicitFunction(imp)
    sampler.SetModelBounds(b)
    sampler.SetSampleDimensions(nx, ny, nz)
    sampler.ComputeNormalsOff()
    sampler.Update()

    volume = pv.wrap(sampler.GetOutput())

    scalar_name = None
    for name in volume.array_names:
        scalar_name = name
        break
    if scalar_name is None:
        raise RuntimeError("SDF sampling failed. No scalars in sampled volume.")

    return volume, scalar_name, pitch_used, was_clamped


def extract_wrapper(volume: pv.DataSet, sdf_name: str, offset: float) -> pv.PolyData:
    rng = volume.get_data_range(sdf_name)
    if not (rng[0] <= offset <= rng[1]):
        raise RuntimeError(
            f"Offset {offset} is outside SDF range {rng}. "
            f"Reduce offset or increase padding or decrease pitch."
        )

    surf = volume.contour(isosurfaces=[offset], scalars=sdf_name)
    surf = surf.extract_surface().triangulate().clean(tolerance=1e-12)
    return surf


def postprocess(mesh: pv.PolyData, smooth_iters: int, smooth_relax: float, decimate: float) -> pv.PolyData:
    out = mesh
    if smooth_iters > 0:
        out = out.smooth(
            n_iter=int(smooth_iters),
            relaxation_factor=float(smooth_relax),
            feature_smoothing=False,
            boundary_smoothing=True,
        )
        out = out.triangulate().clean(tolerance=1e-12)

    if decimate > 0:
        dec = float(decimate)
        dec = min(max(dec, 0.0), 0.95)
        if dec > 0:
            try:
                out = out.decimate_pro(decimate=dec, preserve_topology=True)
                out = out.triangulate().clean(tolerance=1e-12)
            except Exception:
                pass

    return out


def show(mesh: pv.PolyData, wrapper: pv.PolyData):
    pl = pv.Plotter()
    pl.add_mesh(mesh, opacity=0.25, show_edges=False)
    pl.add_mesh(wrapper, opacity=0.60, show_edges=False)
    pl.add_mesh(wrapper, style="wireframe", line_width=1)

    pl.add_text(
        "Mouse: rotate pan zoom\n"
        "Keys: O toggle original, W toggle wireframe, S save screenshot",
        font_size=10,
    )

    state = {"orig": True, "wire": True}
    actors = {}

    actors["orig"] = pl.add_mesh(mesh, opacity=0.25, show_edges=False)
    actors["wrap_solid"] = pl.add_mesh(wrapper, opacity=0.60, show_edges=False)
    actors["wrap_wire"] = pl.add_mesh(wrapper, style="wireframe", line_width=1)

    def toggle_original():
        state["orig"] = not state["orig"]
        actors["orig"].SetVisibility(1 if state["orig"] else 0)
        pl.render()

    def toggle_wire():
        state["wire"] = not state["wire"]
        actors["wrap_wire"].SetVisibility(1 if state["wire"] else 0)
        pl.render()

    def save_shot():
        pl.screenshot("wrap_view.png")

    pl.add_key_event("o", toggle_original)
    pl.add_key_event("w", toggle_wire)
    pl.add_key_event("s", save_shot)

    pl.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="Input STL path")
    p.add_argument("--out", dest="out", default="", help="Output STL path for wrapper")
    p.add_argument("--pitch", type=float, default=2.0, help="Grid spacing in model units")
    p.add_argument("--offset", type=float, default=2.0, help="Wrapper offset. Positive gives outer shell")
    p.add_argument("--padding", type=float, default=-1.0, help="Extra padding around bounds. Default 3*pitch+abs(offset)")
    p.add_argument("--smooth-iters", type=int, default=10, help="Smoothing iterations on wrapper mesh")
    p.add_argument("--smooth-relax", type=float, default=0.2, help="Smoothing relaxation factor")
    p.add_argument("--fill-holes", type=float, default=0.0, help="Try fill holes up to this size before wrapping")
    p.add_argument("--decimate", type=float, default=0.0, help="Decimate ratio 0..0.95")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.inp):
        raise FileNotFoundError(args.inp)

    pitch = float(args.pitch)
    offset = float(args.offset)

    padding = float(args.padding)
    if padding < 0:
        padding = 3.0 * pitch + abs(offset)

    params = WrapParams(
        pitch=pitch,
        offset=offset,
        padding=padding,
        smooth_iters=int(args.smooth_iters),
        smooth_relax=float(args.smooth_relax),
        fill_holes=float(args.fill_holes),
        decimate=float(args.decimate),
    )

    mesh = load_stl(args.inp)
    mesh = try_fill_holes(mesh, params.fill_holes)

    volume, sdf_name, pitch_used, clamped = sample_signed_distance(mesh, params.pitch, params.padding)
    wrapper = extract_wrapper(volume, sdf_name, params.offset)
    wrapper = postprocess(wrapper, params.smooth_iters, params.smooth_relax, params.decimate)

    if args.out:
        wrapper.save(args.out)

    if clamped:
        print(f"Pitch was increased to {pitch_used:.6g} to keep grid dimensions reasonable.")

    show(mesh, wrapper)


if __name__ == "__main__":
    main()