from dataclasses import dataclass
import pyvista as pv

from app.config import AppParams
from app.mesh_io import load_stl
from app.sdf_grid import build_grid_and_sdf
from app.front import build_masks_from_sdf, extract_front_cubes
from app.shell import build_shell_from_front
from app.shrink import shrink_shell_to_mesh
from app.metrics import MeshMetrics, compute_mesh_metrics
from app.octree import build_octree_leaves, balance_2to1, compute_front_leaves_and_outside_grid, front_cubes_polydata
from app.shell_octree import build_shell_from_front_leaves


from dataclasses import dataclass
from app.metrics import MeshMetrics

@dataclass
class PipelineResult:
    mesh: pv.PolyData
    vox_front: pv.PolyData
    shell0: pv.PolyData
    shell1: pv.PolyData
    pitch_used: float
    clamped: bool
    band: float
    mode: str
    shell0_metrics: MeshMetrics | None = None
    shell1_metrics: MeshMetrics | None = None


def run_pipeline(stl_path: str, params: AppParams) -> PipelineResult:
    mesh = load_stl(stl_path)

    img, sdf_cell, pitch_used, clamped = build_grid_and_sdf(
        mesh=mesh,
        pitch=params.grid.pitch,
        padding_mul=params.grid.padding_mul,
        max_dim=params.grid.max_dim,
    )

    band = params.grid.band_mul * pitch_used

    if not params.octree.enabled:
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
        return PipelineResult(mesh, vox_front, shell0, shell1, pitch_used, clamped, band, "uniform")

    leaves, dims_pad, max_level = build_octree_leaves(
        sdf_cell=sdf_cell,
        band=band,
        max_level=int(params.octree.max_level),
    )

    leaves = balance_2to1(
        leaves=leaves,
        sdf_cell=sdf_cell,
        band=band,
        dims=dims_pad,
        max_level=max_level,
        max_iters=int(params.octree.balance_max_iters),
    )

    front_leaves, outside_grid = compute_front_leaves_and_outside_grid(leaves, dims_pad)

    vox_front = front_cubes_polydata(front_leaves, origin=img.origin, pitch=img.spacing[0])
    shell0 = build_shell_from_front_leaves(front_leaves, outside_grid, origin=img.origin, pitch=img.spacing[0])

    if shell0.n_points == 0 or shell0.n_cells == 0:
        raise RuntimeError("Octree shell is empty. Increase pitch or band or decrease octree max_level")

    shell1 = shrink_shell_to_mesh(
        shell=shell0,
        target_mesh=mesh,
        iters=params.shrink.iters,
        step=params.shrink.step,
        constraint_mul=params.shrink.constraint_mul,
        lap_iters_per_step=params.shrink.lap_iters_per_step,
        lap_relax=params.shrink.lap_relax,
    )

    shell0_metrics = compute_mesh_metrics(shell0, mesh)
    shell1_metrics = compute_mesh_metrics(shell1, mesh)

    return PipelineResult(
        mesh=mesh,
        vox_front=vox_front,
        shell0=shell0,
        shell1=shell1,
        pitch_used=pitch_used,
        clamped=clamped,
        band=band,
        mode=mode,
        shell0_metrics=shell0_metrics,
        shell1_metrics=shell1_metrics,
    )