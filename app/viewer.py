import os
import pyvista as pv

from app.config import AppParams
from app.file_dialog import pick_stl_file
from app.mesh_io import load_stl
from app.sdf_grid import build_grid_and_sdf
from app.front import build_masks_from_sdf, extract_front_cubes
from app.shell import build_shell_from_front
from app.shrink import shrink_shell_to_mesh


def build_pipeline(stl_path: str, params: AppParams):
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
        raise RuntimeError("Shell is empty. Try increasing pitch or band.")

    shell1 = shrink_shell_to_mesh(
        shell=shell0,
        target_mesh=mesh,
        iters=params.shrink.iters,
        step=params.shrink.step,
        constraint_mul=params.shrink.constraint_mul,
        lap_iters_per_step=params.shrink.lap_iters_per_step,
        lap_relax=params.shrink.lap_relax,
    )

    return mesh, vox_front, shell0, shell1, pitch_used, clamped


def show_view(stl_path: str, params: AppParams):
    mesh, vox_front, shell0, shell1, pitch_used, clamped = build_pipeline(stl_path, params)

    pl = pv.Plotter()
    pl.add_text(
        f"{os.path.basename(stl_path)}\n"
        f"pitch={params.grid.pitch} used={pitch_used:.4g} band={params.grid.band_mul * pitch_used:.4g}\n"
        "Keys: O STL, C cubes, 1 cube-shell, 2 shrink-shell, W wire, S screenshot",
        font_size=10
    )

    state = {"orig": True, "cubes": True, "shell0": False, "shell1": True, "wire": True}

    a_orig = pl.add_mesh(mesh, opacity=0.25, show_edges=False)
    a_cubes = pl.add_mesh(vox_front, opacity=0.35, show_edges=True)

    a_shell0 = pl.add_mesh(shell0, opacity=0.20, show_edges=False)
    a_shell0.SetVisibility(0)

    a_shell1 = pl.add_mesh(shell1, opacity=0.60, show_edges=False)
    a_wire = pl.add_mesh(shell1, style="wireframe", line_width=1)

    def toggle(actor, key):
        state[key] = not state[key]
        actor.SetVisibility(1 if state[key] else 0)
        pl.render()

    pl.add_key_event("o", lambda: toggle(a_orig, "orig"))
    pl.add_key_event("c", lambda: toggle(a_cubes, "cubes"))
    pl.add_key_event("1", lambda: toggle(a_shell0, "shell0"))
    pl.add_key_event("2", lambda: toggle(a_shell1, "shell1"))
    pl.add_key_event("w", lambda: toggle(a_wire, "wire"))
    pl.add_key_event("s", lambda: pl.screenshot("wrap_view.png"))

    if clamped:
        print(f"Pitch increased to {pitch_used:.6g} due to max-dim limit")

    pl.show()


def run_app(optional_path: str = ""):
    params = AppParams()

    path = (optional_path or "").strip()
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        show_view(path, params)
        return

    while True:
        path = pick_stl_file()
        if not path:
            return
        show_view(path, params)