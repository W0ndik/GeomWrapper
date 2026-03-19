from __future__ import annotations
from dataclasses import dataclass
from typing import Union

import pyvista as pv

from app.config import AppParams
from app.mesh_io import load_stl
from app.sdf_grid import build_grid_and_sdf
from app.front import build_masks_from_sdf, extract_front_cubes
from app.shell import build_shell_from_front
from app.shrink import shrink_shell_to_mesh

from app.octree import (
    build_octree_leaves,
    balance_2to1,
    compute_front_leaves_and_outside_grid,
    front_cubes_polydata,
)
from app.shell_octree import build_shell_from_front_leaves
from app.front_repair import repair_front_defects


@dataclass
class PipelineResult:
    mesh: pv.PolyData
    vox_front: pv.DataSet
    shell0: pv.PolyData
    shell1: pv.PolyData
    pitch_used: float
    clamped: bool
    band: float
    mode: str


def run_pipeline(stl_path: str, params: AppParams, progress_cb=None) -> PipelineResult:
    def report(msg: str):
        if progress_cb is not None:
            progress_cb(msg)

    report("Шаг 1/5: загрузка STL")
    mesh = load_stl(stl_path)

    report("Шаг 2/5: построение декартовой сетки и SDF")
    img, sdf_cell, pitch_used, clamped = build_grid_and_sdf(
        mesh=mesh,
        pitch=params.grid.pitch,
        padding_mul=params.grid.padding_mul,
        max_dim=params.grid.max_dim,
    )

    band = params.grid.band_mul * pitch_used

    if not params.octree.enabled:
        report("Шаг 3/5: выделение фронта на равномерной сетке")
        _geom, outside, front = build_masks_from_sdf(sdf_cell, band)
        vox_front = extract_front_cubes(img, front)

        report("Шаг 4/5: построение начальной оболочки")
        shell0 = build_shell_from_front(
            front,
            outside,
            origin=img.origin,
            pitch=img.spacing[0],
        )

        if shell0.n_points == 0 or shell0.n_cells == 0:
            raise RuntimeError("Shell is empty. Increase pitch or band")

        report("Шаг 5/5: стягивание оболочки")
        shell1 = shrink_shell_to_mesh(
            shell=shell0,
            target_mesh=mesh,
            iters=params.shrink.iters,
            step=params.shrink.step,
            constraint_mul=params.shrink.constraint_mul,
            lap_iters_per_step=params.shrink.lap_iters_per_step,
            lap_relax=params.shrink.lap_relax,
            progress_cb=lambda it, total: report(f"Шаг 5/5: стягивание оболочки, итерация {it}/{total}"),
        )

        return PipelineResult(
            mesh=mesh,
            vox_front=vox_front,
            shell0=shell0,
            shell1=shell1,
            pitch_used=pitch_used,
            clamped=clamped,
            band=band,
            mode="uniform",
        )

    report("Шаг 3/6: построение октодерева")
    leaves, dims_pad, max_level = build_octree_leaves(
        sdf_cell=sdf_cell,
        band=band,
        max_level=int(params.octree.max_level),
    )

    report("Шаг 4/6: 2:1 балансировка октодерева")
    leaves = balance_2to1(
        leaves=leaves,
        sdf_cell=sdf_cell,
        band=band,
        dims=dims_pad,
        max_level=max_level,
        max_iters=int(params.octree.balance_max_iters),
    )

    report("Шаг 5/6: согласование и исправление фронта")
    leaves, front_leaves, outside_grid = repair_front_defects(
        leaves=leaves,
        sdf_cell=sdf_cell,
        band=band,
        dims=dims_pad,
        max_level=max_level,
        balance_max_iters=int(params.octree.balance_max_iters),
        repair_max_iters=max(1, int(params.octree.balance_max_iters)),
    )

    front_leaves, outside_grid = compute_front_leaves_and_outside_grid(leaves, dims_pad)

    vox_front = front_cubes_polydata(
        front_leaves,
        origin=img.origin,
        pitch=img.spacing[0],
    )

    report("Шаг 6/6: построение начальной оболочки")
    shell0 = build_shell_from_front_leaves(
        front_leaves,
        outside_grid,
        origin=img.origin,
        pitch=img.spacing[0],
    )

    if shell0.n_points == 0 or shell0.n_cells == 0:
        raise RuntimeError("Octree shell is empty. Increase pitch or band or decrease octree max_level")
    
    print("shell0 points =", shell0.n_points)
    print("shell0 cells  =", shell0.n_cells)

    report("Шаг 7/7: стягивание оболочки")
    shell1 = shrink_shell_to_mesh(
        shell=shell0,
        target_mesh=mesh,
        iters=params.shrink.iters,
        step=params.shrink.step,
        constraint_mul=params.shrink.constraint_mul,
        lap_iters_per_step=params.shrink.lap_iters_per_step,
        lap_relax=params.shrink.lap_relax,
        progress_cb=lambda it, total: report(f"Шаг 7/7: стягивание оболочки, итерация {it}/{total}"),
    )

    return PipelineResult(
        mesh=mesh,
        vox_front=vox_front,
        shell0=shell0,
        shell1=shell1,
        pitch_used=pitch_used,
        clamped=clamped,
        band=band,
        mode="octree",
    )