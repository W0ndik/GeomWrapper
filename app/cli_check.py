from __future__ import annotations

import argparse
from app.io_stl import read_stl_binary
from app.checks import analyze_mesh, save_report_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stl", help="input binary STL")
    p.add_argument("--no-canon", action="store_true", help="do not merge exact duplicate vertices")
    p.add_argument("--json", default="report.json", help="output report file")
    args = p.parse_args()

    vertices, faces = read_stl_binary(args.stl)

    report = analyze_mesh(vertices, faces, canonicalize=not args.no_canon)

    print("vertices:", report.vertices)
    print("faces:", report.faces)
    print("unique_vertices_exact:", report.unique_vertices_exact)
    print("watertight:", report.watertight)
    print("boundary_edges:", report.boundary_edges)
    print("nonmanifold_edges:", report.nonmanifold_edges)
    print("degenerate_faces:", report.degenerate_faces)
    print("duplicate_faces:", report.duplicate_faces)
    print("signed_volume:", report.signed_volume)
    print("orientation:", report.orientation)
    print("min_angle_deg:", report.min_angle_deg)
    print("min_edge_len:", report.min_edge_len)
    print("max_edge_len:", report.max_edge_len)
    print("min_aspect:", report.min_aspect)
    print("max_aspect:", report.max_aspect)

    save_report_json(args.json, report)


if __name__ == "__main__":
    main()