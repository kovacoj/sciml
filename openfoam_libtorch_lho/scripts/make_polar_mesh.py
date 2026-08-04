"""Generate a structured polar mesh of the unit disk as OpenFOAM polyMesh.

Rings x sectors. Ring 0 cells are wedge prisms (apex at the disk centre),
all other cells are hex prisms. The mesh is exactly orthogonal: every face
normal passes through the two adjacent cell centroids (angular faces at
fixed theta are bisected by symmetry, radial faces lie on ring chords whose
midpoint is on the cell-bisector ray) -- ideal for a plain central FV
Laplacian without non-orthogonal correction.

Two z-layers (empty front/back BCs) -> 2D.

Usage: python3 make_polar_mesh.py --radial 24 --angular 96 --case <casedir>
Writes constant/polyMesh/{points,faces,owner,neighbour,boundary}.
"""

import argparse
import math
import os

ZH = 0.5  # half thickness in z


def vlayer_pts(k, j, layer, n_ang):
    # ring point index: ring k=1..n_rad, sector j=0..n_ang-1, layer 0/1
    return 2 + ((k - 1) * n_ang + j) * 2 + layer


def face_str(verts):
    return f"{len(verts)}(" + " ".join(str(v) for v in verts) + ")"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radial", type=int, required=True)
    ap.add_argument("--angular", type=int, required=True)
    ap.add_argument("--case", type=str, required=True)
    ap.add_argument("--radius", type=float, default=1.0)
    args = ap.parse_args()

    nr, nt, R = args.radial, args.angular, args.radius
    assert nt % 4 == 0, "use a multiple of 4 angular sectors (mode-friendly)"

    outdir = os.path.join(args.case, "constant", "polyMesh")
    os.makedirs(outdir, exist_ok=True)

    radii = [R * k / nr for k in range(1, nr + 1)]
    thetas = [2.0 * math.pi * j / nt for j in range(nt)]

    # ---- points -----------------------------------------------------------
    pts = [(0.0, 0.0, -ZH), (0.0, 0.0, ZH)]
    for k in range(1, nr + 1):
        r = radii[k - 1]
        for th in thetas:
            pts.append((r * math.cos(th), r * math.sin(th), -ZH))
            pts.append((r * math.cos(th), r * math.sin(th), ZH))

    def P(k, j, layer):
        return vlayer_pts(k, j % nt, layer, nt)

    def cell(i, j):
        return i * nt + (j % nt)

    n_cells = nr * nt

    # ---- internal faces ----------------------------------------------------
    faces, owner, neighbour = [], [], []

    for i in range(nr):
        for j in range(nt):
            c = cell(i, j)

            # angular face at theta_{j+1}: owner (i,j), nb (i,j+1), normal +e_theta
            if j < nt - 1:
                jf = (j + 1) % nt
                if i == 0:
                    v = [0, 1, P(1, jf, 1), P(1, jf, 0)]
                else:
                    v = [P(i, jf, 0), P(i, jf, 1), P(i + 1, jf, 1), P(i + 1, jf, 0)]
                faces.append(face_str(v))
                owner.append(c)
                neighbour.append(cell(i, j + 1))

            # wrap angular face at theta_0: owner (i,0), nb (i,nt-1), normal -e_theta
            if j == 0:
                if i == 0:
                    v = [0, P(1, 0, 0), P(1, 0, 1), 1]
                else:
                    v = [P(i, 0, 0), P(i + 1, 0, 0), P(i + 1, 0, 1), P(i, 0, 1)]
                faces.append(face_str(v))
                owner.append(c)
                neighbour.append(cell(i, nt - 1))

            # radial face at radius r_{i+1}: owner (i,j), nb (i+1,j), normal +e_r
            if i < nr - 1:
                k = i + 1
                v = [P(k, j, 0), P(k, j + 1, 0), P(k, j + 1, 1), P(k, j, 1)]
                faces.append(face_str(v))
                owner.append(c)
                neighbour.append(cell(i + 1, j))

    n_internal = len(faces)

    # ---- boundary faces ----------------------------------------------------
    rim_faces, rim_owner = [], []
    for j in range(nt):
        v = [P(nr, j, 0), P(nr, j + 1, 0), P(nr, j + 1, 1), P(nr, j, 1)]
        rim_faces.append(face_str(v))
        rim_owner.append(cell(nr - 1, j))

    tb_faces, tb_owner = [], []
    # bottom (outward normal -z)
    for i in range(nr):
        for j in range(nt):
            if i == 0:
                v = [0, P(1, j + 1, 0), P(1, j, 0)]
            else:
                v = [P(i, j, 0), P(i, j + 1, 0), P(i + 1, j + 1, 0), P(i + 1, j, 0)]
            tb_faces.append(face_str(v))
            tb_owner.append(cell(i, j))
    # top (outward normal +z)
    for i in range(nr):
        for j in range(nt):
            if i == 0:
                v = [1, P(1, j, 1), P(1, j + 1, 1)]
            else:
                v = [P(i, j, 1), P(i + 1, j, 1), P(i + 1, j + 1, 1), P(i, j + 1, 1)]
            tb_faces.append(face_str(v))
            tb_owner.append(cell(i, j))

    all_faces = faces + rim_faces + tb_faces
    all_owner = owner + rim_owner + tb_owner

    # ---- write files -------------------------------------------------------
    def header(cls, obj):
        return (
            "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
            f"    class       {cls};\n"
            '    location    "constant/polyMesh";\n'
            f"    object      {obj};\n}}\n\n"
        )

    with open(os.path.join(outdir, "points"), "w") as f:
        f.write(header("vectorField", "points"))
        f.write(f"{len(pts)}\n(\n")
        for x, y, z in pts:
            f.write(f"({x:.12g} {y:.12g} {z:.12g})\n")
        f.write(")\n")

    with open(os.path.join(outdir, "faces"), "w") as f:
        f.write(header("faceList", "faces"))
        f.write(f"{len(all_faces)}\n(\n")
        for fs in all_faces:
            f.write(fs + "\n")
        f.write(")\n")

    with open(os.path.join(outdir, "owner"), "w") as f:
        f.write(header("labelList", "owner"))
        f.write(f"{len(all_owner)}\n(\n")
        for o_ in all_owner:
            f.write(f"{o_}\n")
        f.write(")\n")

    with open(os.path.join(outdir, "neighbour"), "w") as f:
        f.write(header("labelList", "neighbour"))
        f.write(f"{len(neighbour)}\n(\n")
        for n_ in neighbour:
            f.write(f"{n_}\n")
        f.write(")\n")

    boundary = (
        header("polyBoundaryMesh", "boundary")
        + "2\n(\n"
        + "    rim\n    {\n        type            patch;\n"
        + f"        nFaces          {len(rim_faces)};\n"
        + f"        startFace       {n_internal};\n    }}\n"
        + "    topAndBottom\n    {\n        type            empty;\n"
        + f"        nFaces          {len(tb_faces)};\n"
        + f"        startFace       {n_internal + len(rim_faces)};\n    }}\n"
        + ")\n"
    )
    with open(os.path.join(outdir, "boundary"), "w") as f:
        f.write(boundary)

    print(f"cells={n_cells} points={len(pts)} faces={len(all_faces)} "
          f"internal={n_internal} rim={len(rim_faces)} empty={len(tb_faces)}")


if __name__ == "__main__":
    main()
