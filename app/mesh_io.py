import pyvista as pv


def load_stl(path: str) -> pv.PolyData:
    mesh = pv.read(path)
    mesh = mesh.triangulate().clean(tolerance=1e-12)
    mesh = mesh.compute_normals(
        cell_normals=False,
        point_normals=True,
        auto_orient_normals=True,
        consistent_normals=True,
        split_vertices=False,
    )
    return mesh