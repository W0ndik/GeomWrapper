import numpy as np
import pyvista as pv


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


def build_masks_from_sdf(sdf_cell: np.ndarray, band: float):
    geom = np.abs(sdf_cell) <= band
    outside = (sdf_cell > 0.0) & (~geom)
    front = compute_front(geom, outside)
    return geom, outside, front


def extract_front_cubes(img: pv.ImageData, front: np.ndarray) -> pv.DataSet:
    front_flat = front.astype(np.uint8).reshape(-1, order="F")
    img2 = img.copy(deep=True)
    img2.cell_data["front"] = front_flat
    vox_front = img2.threshold([0.5, 1.5], scalars="front", preference="cell")
    return vox_front