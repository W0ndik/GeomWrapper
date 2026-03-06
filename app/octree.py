from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pyvista as pv


@dataclass
class Leaf:
    id: int
    i0: int
    j0: int
    k0: int
    level: int
    size: int
    sdf_center: float
    typ: int  # 0 outside, 1 geometry, 2 inside


def _center_index(i0: int, size: int, n: int) -> int:
    c = i0 + (size // 2)
    if c < 0:
        return 0
    if c >= n:
        return n - 1
    return c


def _sample_block_sdf(sdf_cell: np.ndarray, i0: int, j0: int, k0: int, size: int) -> Tuple[float, float, float]:
    nx, ny, nz = sdf_cell.shape
    i1 = min(i0 + size, nx)
    j1 = min(j0 + size, ny)
    k1 = min(k0 + size, nz)

    ic = _center_index(i0, i1 - i0, nx)
    jc = _center_index(j0, j1 - j0, ny)
    kc = _center_index(k0, k1 - k0, nz)

    c = float(sdf_cell[ic, jc, kc])

    ci0 = i0
    cj0 = j0
    ck0 = k0
    ci1 = max(i1 - 1, i0)
    cj1 = max(j1 - 1, j0)
    ck1 = max(k1 - 1, k0)

    corners = np.array([
        sdf_cell[ci0, cj0, ck0],
        sdf_cell[ci1, cj0, ck0],
        sdf_cell[ci0, cj1, ck0],
        sdf_cell[ci1, cj1, ck0],
        sdf_cell[ci0, cj0, ck1],
        sdf_cell[ci1, cj0, ck1],
        sdf_cell[ci0, cj1, ck1],
        sdf_cell[ci1, cj1, ck1],
        sdf_cell[ic,  jc,  kc],
    ], dtype=np.float64)

    mn = float(np.min(corners))
    mx = float(np.max(corners))
    return c, mn, mx


def build_octree_leaves(
    sdf_cell: np.ndarray,
    band: float,
    max_level: int,
) -> Tuple[List[Leaf], Tuple[int, int, int], int]:
    nx, ny, nz = sdf_cell.shape
    root_size = 1 << max_level

    nxp = int(np.ceil(nx / root_size) * root_size)
    nyp = int(np.ceil(ny / root_size) * root_size)
    nzp = int(np.ceil(nz / root_size) * root_size)

    next_id = 1
    stack: List[Tuple[int, int, int, int]] = []

    for i0 in range(0, nxp, root_size):
        for j0 in range(0, nyp, root_size):
            for k0 in range(0, nzp, root_size):
                stack.append((i0, j0, k0, 0))

    leaves: List[Leaf] = []

    while stack:
        i0, j0, k0, level = stack.pop()
        size = 1 << (max_level - level)

        ic0 = min(i0, nx - 1) if nx > 0 else 0
        jc0 = min(j0, ny - 1) if ny > 0 else 0
        kc0 = min(k0, nz - 1) if nz > 0 else 0

        c, mn, mx = _sample_block_sdf(sdf_cell, ic0, jc0, kc0, min(size, nx - ic0, ny - jc0, nz - kc0))

        need_refine = False
        if level < max_level:
            if mn <= 0.0 <= mx:
                need_refine = True
            else:
                margin = band + 0.6 * float(size)
                if abs(c) <= margin:
                    need_refine = True

        if need_refine and level < max_level:
            h = size // 2
            nl = level + 1
            stack.extend([
                (i0 + 0, j0 + 0, k0 + 0, nl),
                (i0 + h, j0 + 0, k0 + 0, nl),
                (i0 + 0, j0 + h, k0 + 0, nl),
                (i0 + h, j0 + h, k0 + 0, nl),
                (i0 + 0, j0 + 0, k0 + h, nl),
                (i0 + h, j0 + 0, k0 + h, nl),
                (i0 + 0, j0 + h, k0 + h, nl),
                (i0 + h, j0 + h, k0 + h, nl),
            ])
            continue

        typ = 1 if abs(c) <= band else (0 if c > band else 2)

        leaf = Leaf(
            id=next_id,
            i0=i0,
            j0=j0,
            k0=k0,
            level=level,
            size=size,
            sdf_center=c,
            typ=typ,
        )
        next_id += 1
        leaves.append(leaf)

    return leaves, (nxp, nyp, nzp), max_level


def _fill_owner_arrays(
    leaves: List[Leaf],
    dims: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = dims
    owner_id = np.zeros((nx, ny, nz), dtype=np.int32)
    owner_lvl = np.zeros((nx, ny, nz), dtype=np.uint8)
    owner_typ = np.zeros((nx, ny, nz), dtype=np.uint8)

    for lf in leaves:
        i1 = min(lf.i0 + lf.size, nx)
        j1 = min(lf.j0 + lf.size, ny)
        k1 = min(lf.k0 + lf.size, nz)
        owner_id[lf.i0:i1, lf.j0:j1, lf.k0:k1] = lf.id
        owner_lvl[lf.i0:i1, lf.j0:j1, lf.k0:k1] = lf.level
        owner_typ[lf.i0:i1, lf.j0:j1, lf.k0:k1] = lf.typ

    return owner_id, owner_lvl, owner_typ


def _refine_leaf(leaves_by_id: Dict[int, Leaf], leaf_id: int, max_level: int) -> List[Leaf]:
    lf = leaves_by_id.get(leaf_id)
    if lf is None:
        return []
    if lf.level >= max_level:
        return []

    h = lf.size // 2
    nl = lf.level + 1

    children: List[Leaf] = []
    for dx in (0, h):
        for dy in (0, h):
            for dz in (0, h):
                children.append(Leaf(
                    id=0,
                    i0=lf.i0 + dx,
                    j0=lf.j0 + dy,
                    k0=lf.k0 + dz,
                    level=nl,
                    size=h,
                    sdf_center=lf.sdf_center,
                    typ=lf.typ,
                ))
    return children


def balance_2to1(
    leaves: List[Leaf],
    sdf_cell: np.ndarray,
    band: float,
    dims: Tuple[int, int, int],
    max_level: int,
    max_iters: int,
) -> List[Leaf]:
    next_id = max((lf.id for lf in leaves), default=0) + 1

    for _ in range(max_iters):
        leaves_by_id = {lf.id: lf for lf in leaves}
        owner_id, owner_lvl, owner_typ = _fill_owner_arrays(leaves, dims)

        bad_ids = []

        a = owner_lvl[:-1, :, :]
        b = owner_lvl[1:, :, :]
        m = (np.abs(a.astype(np.int16) - b.astype(np.int16)) > 1)
        if np.any(m):
            coarser_is_left = a < b
            cid = np.where(coarser_is_left, owner_id[:-1, :, :], owner_id[1:, :, :])
            bad_ids.append(np.unique(cid[m]))

        a = owner_lvl[:, :-1, :]
        b = owner_lvl[:, 1:, :]
        m = (np.abs(a.astype(np.int16) - b.astype(np.int16)) > 1)
        if np.any(m):
            coarser_is_left = a < b
            cid = np.where(coarser_is_left, owner_id[:, :-1, :], owner_id[:, 1:, :])
            bad_ids.append(np.unique(cid[m]))

        a = owner_lvl[:, :, :-1]
        b = owner_lvl[:, :, 1:]
        m = (np.abs(a.astype(np.int16) - b.astype(np.int16)) > 1)
        if np.any(m):
            coarser_is_left = a < b
            cid = np.where(coarser_is_left, owner_id[:, :, :-1], owner_id[:, :, 1:])
            bad_ids.append(np.unique(cid[m]))

        if not bad_ids:
            break

        refine_ids = np.unique(np.concatenate(bad_ids)).astype(np.int32)
        refine_ids = refine_ids[refine_ids != 0]
        if refine_ids.size == 0:
            break

        new_leaves: List[Leaf] = []
        refined = set(int(x) for x in refine_ids.tolist())

        for lf in leaves:
            if lf.id in refined and lf.level < max_level:
                children = _refine_leaf(leaves_by_id, lf.id, max_level)
                for ch in children:
                    i0 = min(ch.i0, sdf_cell.shape[0] - 1)
                    j0 = min(ch.j0, sdf_cell.shape[1] - 1)
                    k0 = min(ch.k0, sdf_cell.shape[2] - 1)
                    c, _mn, _mx = _sample_block_sdf(sdf_cell, i0, j0, k0, min(ch.size, sdf_cell.shape[0]-i0, sdf_cell.shape[1]-j0, sdf_cell.shape[2]-k0))
                    ch.sdf_center = c
                    ch.typ = 1 if abs(c) <= band else (0 if c > band else 2)
                    ch.id = next_id
                    next_id += 1
                new_leaves.extend(children)
            else:
                new_leaves.append(lf)

        leaves = new_leaves

    return leaves


def compute_front_leaves_and_outside_grid(
    leaves: List[Leaf],
    dims: Tuple[int, int, int],
) -> Tuple[List[Leaf], np.ndarray]:
    nx, ny, nz = dims
    outside = np.zeros((nx, ny, nz), dtype=bool)

    for lf in leaves:
        if lf.typ != 0:
            continue
        i1 = min(lf.i0 + lf.size, nx)
        j1 = min(lf.j0 + lf.size, ny)
        k1 = min(lf.k0 + lf.size, nz)
        outside[lf.i0:i1, lf.j0:j1, lf.k0:k1] = True

    front_leaves: List[Leaf] = []
    for lf in leaves:
        if lf.typ != 1:
            continue

        i0, j0, k0, s = lf.i0, lf.j0, lf.k0, lf.size
        i1 = min(i0 + s, nx)
        j1 = min(j0 + s, ny)
        k1 = min(k0 + s, nz)

        open_face = False

        if i0 > 0 and np.any(outside[i0 - 1, j0:j1, k0:k1]):
            open_face = True
        if i1 < nx and np.any(outside[i1, j0:j1, k0:k1]):
            open_face = True
        if j0 > 0 and np.any(outside[i0:i1, j0 - 1, k0:k1]):
            open_face = True
        if j1 < ny and np.any(outside[i0:i1, j1, k0:k1]):
            open_face = True
        if k0 > 0 and np.any(outside[i0:i1, j0:j1, k0 - 1]):
            open_face = True
        if k1 < nz and np.any(outside[i0:i1, j0:j1, k1]):
            open_face = True

        if open_face:
            front_leaves.append(lf)

    return front_leaves, outside


def front_cubes_polydata(front_leaves: List[Leaf], origin, pitch: float) -> pv.PolyData:
    if not front_leaves:
        return pv.PolyData()

    blocks = pv.MultiBlock()
    for idx, lf in enumerate(front_leaves):
        x0 = origin[0] + lf.i0 * pitch
        y0 = origin[1] + lf.j0 * pitch
        z0 = origin[2] + lf.k0 * pitch
        s = lf.size * pitch
        c = (x0 + s * 0.5, y0 + s * 0.5, z0 + s * 0.5)
        cube = pv.Cube(center=c, x_length=s, y_length=s, z_length=s)
        blocks[idx] = cube

    merged = blocks.combine()
    return merged.clean(tolerance=1e-12)