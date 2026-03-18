from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pyvista as pv
import vtk


@dataclass
class MeshMetrics:
    n_points: int
    n_triangles: int
    min_edge: float
    mean_edge: float
    max_edge: float
    min_area: float
    mean_area: float
    max_area: float
    min_angle_deg: float
    mean_angle_deg: float
    max_angle_deg: float
    mean_distance_to_target: float | None = None
    max_distance_to_target: float | None = None


def _triangle_faces(mesh: pv.PolyData) -> np.ndarray:
    tri = mesh.triangulate().clean()
    faces = tri.faces.reshape((-1, 4))
    if np.any(faces[:, 0] != 3):
        raise ValueError("Mesh contains non-triangle faces after triangulate()")
    return faces[:, 1:4]


def _edge_lengths(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    p0 = points[tris[:, 0]]
    p1 = points[tris[:, 1]]
    p2 = points[tris[:, 2]]

    e01 = np.linalg.norm(p1 - p0, axis=1)
    e12 = np.linalg.norm(p2 - p1, axis=1)
    e20 = np.linalg.norm(p0 - p2, axis=1)

    return np.concatenate([e01, e12, e20])


def _triangle_areas(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    p0 = points[tris[:, 0]]
    p1 = points[tris[:, 1]]
    p2 = points[tris[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _triangle_angles_deg(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    p0 = points[tris[:, 0]]
    p1 = points[tris[:, 1]]
    p2 = points[tris[:, 2]]

    def angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        denom = na * nb
        valid = denom > 0.0

        cosv = np.empty(len(a), dtype=float)
        cosv.fill(1.0)
        cosv[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
        cosv = np.clip(cosv, -1.0, 1.0)
        return np.degrees(np.arccos(cosv))

    a0 = angle(p1 - p0, p2 - p0)
    a1 = angle(p0 - p1, p2 - p1)
    a2 = angle(p0 - p2, p1 - p2)

    return np.concatenate([a0, a1, a2])


def _distance_to_target(mesh: pv.PolyData, target: pv.PolyData) -> tuple[float, float]:
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(target)
    locator.BuildLocator()

    pts = np.asarray(mesh.points, dtype=float)
    cp = [0.0, 0.0, 0.0]
    cell_id = vtk.mutable(0)
    sub_id = vtk.mutable(0)
    dist2 = vtk.mutable(0.0)

    distances = np.empty(len(pts), dtype=float)

    for i, p in enumerate(pts):
        locator.FindClosestPoint(p, cp, cell_id, sub_id, dist2)
        distances[i] = math.sqrt(float(dist2))

    return float(np.mean(distances)), float(np.max(distances))


def compute_mesh_metrics(mesh: pv.PolyData, target: pv.PolyData | None = None) -> MeshMetrics:
    tri = mesh.triangulate().clean()
    points = np.asarray(tri.points, dtype=float)
    tris = _triangle_faces(tri)

    edges = _edge_lengths(points, tris)
    areas = _triangle_areas(points, tris)
    angles = _triangle_angles_deg(points, tris)

    mean_dist = None
    max_dist = None
    if target is not None and target.n_points > 0 and tri.n_points > 0:
        mean_dist, max_dist = _distance_to_target(tri, target)

    return MeshMetrics(
        n_points=int(tri.n_points),
        n_triangles=int(tri.n_cells),
        min_edge=float(np.min(edges)) if edges.size else 0.0,
        mean_edge=float(np.mean(edges)) if edges.size else 0.0,
        max_edge=float(np.max(edges)) if edges.size else 0.0,
        min_area=float(np.min(areas)) if areas.size else 0.0,
        mean_area=float(np.mean(areas)) if areas.size else 0.0,
        max_area=float(np.max(areas)) if areas.size else 0.0,
        min_angle_deg=float(np.min(angles)) if angles.size else 0.0,
        mean_angle_deg=float(np.mean(angles)) if angles.size else 0.0,
        max_angle_deg=float(np.max(angles)) if angles.size else 0.0,
        mean_distance_to_target=mean_dist,
        max_distance_to_target=max_dist,
    )