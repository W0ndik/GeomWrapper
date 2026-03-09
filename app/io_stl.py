from __future__ import annotations

from typing import List, Tuple
import struct
import math


def read_stl_binary(path: str) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    with open(path, "rb") as f:
        header = f.read(80)
        if len(header) != 80:
            raise ValueError("Bad STL: header too short")
        n_tri_bytes = f.read(4)
        if len(n_tri_bytes) != 4:
            raise ValueError("Bad STL: missing triangle count")
        n_tri = struct.unpack("<I", n_tri_bytes)[0]

        vertices: List[Tuple[float, float, float]] = []
        faces: List[Tuple[int, int, int]] = []

        for _ in range(n_tri):
            rec = f.read(50)
            if len(rec) != 50:
                raise ValueError("Bad STL: truncated triangle record")

            # normal (ignored)
            nx, ny, nz = struct.unpack_from("<fff", rec, 0)

            v0 = struct.unpack_from("<fff", rec, 12)
            v1 = struct.unpack_from("<fff", rec, 24)
            v2 = struct.unpack_from("<fff", rec, 36)
            # attr = struct.unpack_from("<H", rec, 48)[0]

            i0 = len(vertices); vertices.append((float(v0[0]), float(v0[1]), float(v0[2])))
            i1 = len(vertices); vertices.append((float(v1[0]), float(v1[1]), float(v1[2])))
            i2 = len(vertices); vertices.append((float(v2[0]), float(v2[1]), float(v2[2])))
            faces.append((i0, i1, i2))

    return vertices, faces


def write_stl_binary(path: str, vertices: List[Tuple[float, float, float]], faces: List[Tuple[int, int, int]]) -> None:
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def norm(a):
        return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

    def normalize(a):
        l = norm(a)
        if l == 0.0:
            return (0.0, 0.0, 0.0)
        return (a[0] / l, a[1] / l, a[2] / l)

    with open(path, "wb") as f:
        header = b"GeomWrapper STL\x00" + b"\x00" * (80 - len(b"GeomWrapper STL\x00"))
        f.write(header)
        f.write(struct.pack("<I", len(faces)))

        for a, b, c in faces:
            A = vertices[a]
            B = vertices[b]
            C = vertices[c]
            n = cross(sub(B, A), sub(C, A))
            n = normalize(n)

            f.write(struct.pack("<fff", float(n[0]), float(n[1]), float(n[2])))
            f.write(struct.pack("<fff", float(A[0]), float(A[1]), float(A[2])))
            f.write(struct.pack("<fff", float(B[0]), float(B[1]), float(B[2])))
            f.write(struct.pack("<fff", float(C[0]), float(C[1]), float(C[2])))
            f.write(struct.pack("<H", 0))