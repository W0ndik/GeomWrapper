from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import json
import math


@dataclass
class MeshReport:
    vertices: int
    faces: int

    # topology
    unique_vertices_exact: int
    boundary_edges: int
    nonmanifold_edges: int
    watertight: bool

    # degenerates / duplicates
    degenerate_faces: int
    duplicate_faces: int

    # orientation
    signed_volume: float
    orientation: str  # "outward_or_unknown" | "inverted" | "zero_or_open"

    # quality stats
    min_angle_deg: Optional[float]
    max_angle_deg: Optional[float]
    min_edge_len: Optional[float]
    max_edge_len: Optional[float]
    min_aspect: Optional[float]
    max_aspect: Optional[float]


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a) -> float:
    return math.sqrt(_dot(a, a))


def canonicalize_vertices_exact(
    vertices: List[Tuple[float, float, float]],
    faces: List[Tuple[int, int, int]],
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    """
    Склеивает вершины только по точному совпадению координат (без допусков).
    Полезно, если сетка содержит дубликаты вершин в одинаковых точках.
    """
    map_idx: Dict[Tuple[float, float, float], int] = {}
    new_vertices: List[Tuple[float, float, float]] = []
    remap: List[int] = [0] * len(vertices)

    for i, v in enumerate(vertices):
        key = (float(v[0]), float(v[1]), float(v[2]))
        j = map_idx.get(key)
        if j is None:
            j = len(new_vertices)
            map_idx[key] = j
            new_vertices.append(key)
        remap[i] = j

    new_faces: List[Tuple[int, int, int]] = []
    for a, b, c in faces:
        new_faces.append((remap[a], remap[b], remap[c]))

    return new_vertices, new_faces


def analyze_mesh(
    vertices: List[Tuple[float, float, float]],
    faces: List[Tuple[int, int, int]],
    canonicalize: bool = True,
) -> MeshReport:
    """
    Проверки без EPS и без "почти равно".
    Это строгая диагностика: совпадения вершин только по точному равенству.
    """
    v_in = vertices
    f_in = faces

    if canonicalize:
        v, f = canonicalize_vertices_exact(v_in, f_in)
        unique_vertices_exact = len(v)
    else:
        v, f = v_in, f_in
        unique_vertices_exact = len(v_in)

    # ---- edge counts ----
    edge_count: Dict[Tuple[int, int], int] = {}
    for a, b, c in f:
        e0 = (a, b) if a < b else (b, a)
        e1 = (b, c) if b < c else (c, b)
        e2 = (c, a) if c < a else (a, c)
        edge_count[e0] = edge_count.get(e0, 0) + 1
        edge_count[e1] = edge_count.get(e1, 0) + 1
        edge_count[e2] = edge_count.get(e2, 0) + 1

    boundary_edges = 0
    nonmanifold_edges = 0
    for _, cnt in edge_count.items():
        if cnt == 1:
            boundary_edges += 1
        elif cnt != 2:
            nonmanifold_edges += 1

    watertight = (boundary_edges == 0 and nonmanifold_edges == 0)

    # ---- duplicates / degenerates ----
    face_seen: Dict[Tuple[int, int, int], int] = {}
    duplicate_faces = 0
    degenerate_faces = 0

    for a, b, c in f:
        key = tuple(sorted((a, b, c)))
        prev = face_seen.get(key, 0)
        if prev >= 1:
            duplicate_faces += 1
        face_seen[key] = prev + 1

        if a == b or b == c or c == a:
            degenerate_faces += 1
            continue

        A = v[a]
        B = v[b]
        C = v[c]
        ab = _vec_sub(B, A)
        ac = _vec_sub(C, A)
        cr = _cross(ab, ac)
        if _dot(cr, cr) == 0.0:
            degenerate_faces += 1

    # ---- signed volume (closed mesh only really meaningful, но считаем всегда) ----
    vol6 = 0.0
    for a, b, c in f:
        A = v[a]
        B = v[b]
        C = v[c]
        vol6 += _dot(A, _cross(B, C))
    signed_volume = vol6 / 6.0

    if not watertight:
        orientation = "zero_or_open"
    else:
        orientation = "outward_or_unknown" if signed_volume >= 0.0 else "inverted"

    # ---- quality stats ----
    min_angle = None
    max_angle = None
    min_edge = None
    max_edge = None
    min_aspect = None
    max_aspect = None

    for a, b, c in f:
        if a == b or b == c or c == a:
            continue

        A = v[a]
        B = v[b]
        C = v[c]
        ab = _vec_sub(B, A)
        bc = _vec_sub(C, B)
        ca = _vec_sub(A, C)

        lab = _norm(ab)
        lbc = _norm(bc)
        lca = _norm(ca)

        if lab == 0.0 or lbc == 0.0 or lca == 0.0:
            continue

        # update edge stats
        for L in (lab, lbc, lca):
            min_edge = L if (min_edge is None or L < min_edge) else min_edge
            max_edge = L if (max_edge is None or L > max_edge) else max_edge

        # aspect ratio: max / min
        Lmin = min(lab, lbc, lca)
        Lmax = max(lab, lbc, lca)
        aspect = Lmax / Lmin

        min_aspect = aspect if (min_aspect is None or aspect < min_aspect) else min_aspect
        max_aspect = aspect if (max_aspect is None or aspect > max_aspect) else max_aspect

        # angles via cosine law
        # angle at A is between AB and AC
        ac = _vec_sub(C, A)
        la = lab
        lc = _norm(ac)
        # actually edges around A: AB and AC
        if lc == 0.0:
            continue

        cosA = _dot(ab, ac) / (lab * lc)
        cosA = _clamp(cosA, -1.0, 1.0)
        angA = math.degrees(math.acos(cosA))

        # angle at B: between BA and BC
        ba = _vec_sub(A, B)
        if _norm(ba) == 0.0:
            continue
        cosB = _dot(ba, bc) / (_norm(ba) * lbc)
        cosB = _clamp(cosB, -1.0, 1.0)
        angB = math.degrees(math.acos(cosB))

        # angle at C: between CB and CA
        cb = _vec_sub(B, C)
        if _norm(cb) == 0.0:
            continue
        cosC = _dot(cb, ca) / (_norm(cb) * lca)
        cosC = _clamp(cosC, -1.0, 1.0)
        angC = math.degrees(math.acos(cosC))

        for ang in (angA, angB, angC):
            min_angle = ang if (min_angle is None or ang < min_angle) else min_angle
            max_angle = ang if (max_angle is None or ang > max_angle) else max_angle

    return MeshReport(
        vertices=len(v_in),
        faces=len(f_in),
        unique_vertices_exact=unique_vertices_exact,
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        watertight=watertight,
        degenerate_faces=degenerate_faces,
        duplicate_faces=duplicate_faces,
        signed_volume=signed_volume,
        orientation=orientation,
        min_angle_deg=min_angle,
        max_angle_deg=max_angle,
        min_edge_len=min_edge,
        max_edge_len=max_edge,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
    )


def save_report_json(path: str, report: MeshReport) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)