from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from app.octree import Leaf, balance_2to1, compute_front_leaves_and_outside_grid


SegmentKey = Tuple[Tuple[int, int, int], Tuple[int, int, int]]
FaceRef = Tuple[int, int]  # (leaf_id, face_id)


def _sample_block_sdf(
    sdf_cell: np.ndarray,
    i0: int,
    j0: int,
    k0: int,
    size: int,
) -> Tuple[float, float, float]:
    nx, ny, nz = sdf_cell.shape

    i1 = min(i0 + size, nx)
    j1 = min(j0 + size, ny)
    k1 = min(k0 + size, nz)

    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        ic = min(max(i0, 0), max(nx - 1, 0))
        jc = min(max(j0, 0), max(ny - 1, 0))
        kc = min(max(k0, 0), max(nz - 1, 0))
        c = float(sdf_cell[ic, jc, kc])
        return c, c, c

    ic = min(i0 + (i1 - i0) // 2, nx - 1)
    jc = min(j0 + (j1 - j0) // 2, ny - 1)
    kc = min(k0 + (k1 - k0) // 2, nz - 1)

    c = float(sdf_cell[ic, jc, kc])

    ci0 = i0
    cj0 = j0
    ck0 = k0
    ci1 = max(i1 - 1, i0)
    cj1 = max(j1 - 1, j0)
    ck1 = max(k1 - 1, k0)

    samples = np.array(
        [
            sdf_cell[ci0, cj0, ck0],
            sdf_cell[ci1, cj0, ck0],
            sdf_cell[ci0, cj1, ck0],
            sdf_cell[ci1, cj1, ck0],
            sdf_cell[ci0, cj0, ck1],
            sdf_cell[ci1, cj0, ck1],
            sdf_cell[ci0, cj1, ck1],
            sdf_cell[ci1, cj1, ck1],
            sdf_cell[ic, jc, kc],
        ],
        dtype=np.float64,
    )

    return c, float(np.min(samples)), float(np.max(samples))


def _refine_leaf(
    lf: Leaf,
    sdf_cell: np.ndarray,
    band: float,
) -> List[Leaf]:
    if lf.size <= 1:
        return [lf]

    h = lf.size // 2
    nl = lf.level + 1

    out: List[Leaf] = []

    for dx in (0, h):
        for dy in (0, h):
            for dz in (0, h):
                i0 = lf.i0 + dx
                j0 = lf.j0 + dy
                k0 = lf.k0 + dz

                ii = min(i0, sdf_cell.shape[0] - 1)
                jj = min(j0, sdf_cell.shape[1] - 1)
                kk = min(k0, sdf_cell.shape[2] - 1)

                c, _mn, _mx = _sample_block_sdf(
                    sdf_cell=sdf_cell,
                    i0=ii,
                    j0=jj,
                    k0=kk,
                    size=min(
                        h,
                        max(sdf_cell.shape[0] - ii, 1),
                        max(sdf_cell.shape[1] - jj, 1),
                        max(sdf_cell.shape[2] - kk, 1),
                    ),
                )

                typ = 1 if abs(c) <= band else (0 if c > band else 2)

                out.append(
                    Leaf(
                        id=0,
                        i0=i0,
                        j0=j0,
                        k0=k0,
                        level=nl,
                        size=h,
                        sdf_center=c,
                        typ=typ,
                    )
                )

    return out


def _reassign_ids(leaves: List[Leaf]) -> List[Leaf]:
    out: List[Leaf] = []
    next_id = 1
    for lf in leaves:
        out.append(
            Leaf(
                id=next_id,
                i0=lf.i0,
                j0=lf.j0,
                k0=lf.k0,
                level=lf.level,
                size=lf.size,
                sdf_center=lf.sdf_center,
                typ=lf.typ,
            )
        )
        next_id += 1
    return out


def _leaf_bbox(lf: Leaf) -> Tuple[int, int, int, int, int, int]:
    return (
        lf.i0,
        lf.i0 + lf.size,
        lf.j0,
        lf.j0 + lf.size,
        lf.k0,
        lf.k0 + lf.size,
    )


def _boxes_overlap(
    a: Tuple[int, int, int, int, int, int],
    b: Tuple[int, int, int, int, int, int],
) -> bool:
    return not (
        a[1] <= b[0]
        or b[1] <= a[0]
        or a[3] <= b[2]
        or b[3] <= a[2]
        or a[5] <= b[4]
        or b[5] <= a[4]
    )


def _segment_box(seg: SegmentKey, halo: int = 1) -> Tuple[int, int, int, int, int, int]:
    p0, p1 = seg
    xmin = min(p0[0], p1[0]) - halo
    xmax = max(p0[0], p1[0]) + halo
    ymin = min(p0[1], p1[1]) - halo
    ymax = max(p0[1], p1[1]) + halo
    zmin = min(p0[2], p1[2]) - halo
    zmax = max(p0[2], p1[2]) + halo

    return (xmin, xmax, ymin, ymax, zmin, zmax)


def _canon_seg(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> SegmentKey:
    return (a, b) if a <= b else (b, a)


def _open_faces_for_leaf(
    lf: Leaf,
    outside: np.ndarray,
) -> List[int]:
    nx, ny, nz = outside.shape

    i0, j0, k0 = lf.i0, lf.j0, lf.k0
    i1 = min(i0 + lf.size, nx)
    j1 = min(j0 + lf.size, ny)
    k1 = min(k0 + lf.size, nz)

    faces: List[int] = []

    if i0 > 0 and np.any(outside[i0 - 1, j0:j1, k0:k1]):
        faces.append(0)
    if i1 < nx and np.any(outside[i1, j0:j1, k0:k1]):
        faces.append(1)
    if j0 > 0 and np.any(outside[i0:i1, j0 - 1, k0:k1]):
        faces.append(2)
    if j1 < ny and np.any(outside[i0:i1, j1, k0:k1]):
        faces.append(3)
    if k0 > 0 and np.any(outside[i0:i1, j0:j1, k0 - 1]):
        faces.append(4)
    if k1 < nz and np.any(outside[i0:i1, j0:j1, k1]):
        faces.append(5)

    return faces


def _face_segments(lf: Leaf, face_id: int) -> List[SegmentKey]:
    i0, j0, k0 = lf.i0, lf.j0, lf.k0
    s = lf.size
    i1 = i0 + s
    j1 = j0 + s
    k1 = k0 + s

    segs: List[SegmentKey] = []

    if face_id in (0, 1):
        x = i0 if face_id == 0 else i1

        for y in range(j0, j1):
            segs.append(_canon_seg((x, y, k0), (x, y + 1, k0)))
            segs.append(_canon_seg((x, y, k1), (x, y + 1, k1)))

        for z in range(k0, k1):
            segs.append(_canon_seg((x, j0, z), (x, j0, z + 1)))
            segs.append(_canon_seg((x, j1, z), (x, j1, z + 1)))

        return segs

    if face_id in (2, 3):
        y = j0 if face_id == 2 else j1

        for x in range(i0, i1):
            segs.append(_canon_seg((x, y, k0), (x + 1, y, k0)))
            segs.append(_canon_seg((x, y, k1), (x + 1, y, k1)))

        for z in range(k0, k1):
            segs.append(_canon_seg((i0, y, z), (i0, y, z + 1)))
            segs.append(_canon_seg((i1, y, z), (i1, y, z + 1)))

        return segs

    z = k0 if face_id == 4 else k1

    for x in range(i0, i1):
        segs.append(_canon_seg((x, j0, z), (x + 1, j0, z)))
        segs.append(_canon_seg((x, j1, z), (x + 1, j1, z)))

    for y in range(j0, j1):
        segs.append(_canon_seg((i0, y, z), (i0, y + 1, z)))
        segs.append(_canon_seg((i1, y, z), (i1, y + 1, z)))

    return segs


def detect_front_defects(
    front_leaves: List[Leaf],
    outside: np.ndarray,
) -> Tuple[Dict[SegmentKey, Set[FaceRef]], List[SegmentKey]]:
    seg_refs: Dict[SegmentKey, Set[FaceRef]] = {}

    for lf in front_leaves:
        open_faces = _open_faces_for_leaf(lf, outside)
        for face_id in open_faces:
            face_ref = (lf.id, face_id)
            for seg in _face_segments(lf, face_id):
                seg_refs.setdefault(seg, set()).add(face_ref)

    defects: List[SegmentKey] = []

    for seg, refs in seg_refs.items():
        # Для замкнутой манифолдной поверхности каждый элементарный подотрезок
        # должен принадлежать ровно двум фронтовым patch-границам.
        if len(refs) != 2:
            defects.append(seg)

    return seg_refs, defects


def _collect_refine_ids_around_defects(
    leaves: List[Leaf],
    defects: List[SegmentKey],
    max_level: int,
) -> Set[int]:
    if not defects:
        return set()

    leaf_boxes = {lf.id: _leaf_bbox(lf) for lf in leaves}
    refine_ids: Set[int] = set()

    for seg in defects:
        box = _segment_box(seg, halo=1)
        for lf in leaves:
            if lf.level >= max_level:
                continue
            if _boxes_overlap(leaf_boxes[lf.id], box):
                refine_ids.add(lf.id)

    return refine_ids


def _refine_selected_leaves(
    leaves: List[Leaf],
    refine_ids: Set[int],
    sdf_cell: np.ndarray,
    band: float,
    max_level: int,
) -> List[Leaf]:
    out: List[Leaf] = []

    for lf in leaves:
        if lf.id in refine_ids and lf.level < max_level and lf.size > 1:
            out.extend(_refine_leaf(lf, sdf_cell, band))
        else:
            out.append(lf)

    return _reassign_ids(out)


def repair_front_defects(
    leaves: List[Leaf],
    sdf_cell: np.ndarray,
    band: float,
    dims: Tuple[int, int, int],
    max_level: int,
    balance_max_iters: int,
    repair_max_iters: int = 8,
) -> Tuple[List[Leaf], List[Leaf], np.ndarray]:
    work = _reassign_ids(leaves)

    for _ in range(int(repair_max_iters)):
        front_leaves, outside = compute_front_leaves_and_outside_grid(work, dims)
        _seg_refs, defects = detect_front_defects(front_leaves, outside)

        if not defects:
            return work, front_leaves, outside

        refine_ids = _collect_refine_ids_around_defects(work, defects, max_level)
        if not refine_ids:
            return work, front_leaves, outside

        new_work = _refine_selected_leaves(
            leaves=work,
            refine_ids=refine_ids,
            sdf_cell=sdf_cell,
            band=band,
            max_level=max_level,
        )

        if len(new_work) == len(work):
            return work, front_leaves, outside

        work = balance_2to1(
            leaves=new_work,
            sdf_cell=sdf_cell,
            band=band,
            dims=dims,
            max_level=max_level,
            max_iters=int(balance_max_iters),
        )
        work = _reassign_ids(work)

    front_leaves, outside = compute_front_leaves_and_outside_grid(work, dims)
    return work, front_leaves, outside