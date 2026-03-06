import math
import numpy as np
import pyvista as pv
import vtk


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


def sample_sdf_points(mesh: pv.PolyData, bounds, dims_points):
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
        raise RuntimeError("SDF sampling produced no scalars")
    return img, scalar_name


def point_to_cell_average(img: pv.ImageData, point_scalar_name: str) -> np.ndarray:
    nxp, nyp, nzp = img.dimensions
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
    return c


def build_grid_and_sdf(mesh: pv.PolyData, pitch: float, padding_mul: float, max_dim: int):
    padding = padding_mul * pitch
    b = padded_bounds(mesh.bounds, padding)

    nx, ny, nz, pitch_used, clamped = choose_dims(b, pitch, max_dim)
    dims_points = (nx + 1, ny + 1, nz + 1)

    img, sdf_name = sample_sdf_points(mesh, b, dims_points)
    sdf_cell = point_to_cell_average(img, sdf_name)

    return img, sdf_cell, pitch_used, clamped