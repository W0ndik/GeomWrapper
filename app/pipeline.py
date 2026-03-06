from dataclasses import dataclass
import pyvista as pv

from app.config import AppParams
from app.mesh_io import load_stl
from app.sdf_grid import build_grid_and_sdf
from app.front import build_masks_from_sdf, extract_front_cubes
from app.shell import build_shell_from_front
from app.shrink import shrink_shell_to_mesh


@dataclass
class PipelineResult:
    mesh: pv.PolyData
    vox_front: pv.DataSet
    shell0: pv.PolyData
    shell1: pv.PolyData
    pitch_used: float
    clamped: bool
    band: float


def run_pipeline(stl_path: str, params: AppParams) -> PipelineResult:
    mesh = load_stl(stl_path)

    img, sdf_cell, pitch_used, clamped = build_grid_and_sdf(
        mesh=mesh,
        pitch=params.grid.pitch,
        padding_mul=params.grid.padding_mul,
        max_dim=params.grid.max_dim,
    )

    band = params.grid.band_mul * pitch_used
    _geom, outside, front = build_masks_from_sdf(sdf_cell, band)

    vox_front = extract_front_cubes(img, front)

    shell0 = build_shell_from_front(front, outside, origin=img.origin, pitch=img.spacing[0])
    if shell0.n_points == 0 or shell0.n_cells == 0:
        raise RuntimeError("Shell is empty. Increase pitch or band")

    shell1 = shrink_shell_to_mesh(
        shell=shell0,
        target_mesh=mesh,
        iters=params.shrink.iters,
        step=params.shrink.step,
        constraint_mul=params.shrink.constraint_mul,
        lap_iters_per_step=params.shrink.lap_iters_per_step,
        lap_relax=params.shrink.lap_relax,
    )

    return PipelineResult(
        mesh=mesh,
        vox_front=vox_front,
        shell0=shell0,
        shell1=shell1,
        pitch_used=pitch_used,
        clamped=clamped,
        band=band,
    )