#!/usr/bin/env python3
"""Generate cases/four_port_64 (blockMeshDict + fields + physicsProperties).

Geometry recovered from techMathGroup/tpfm_unet data/ (see references/UPSTREAM.md):
  - 64x64 area of interest = 0.128 m x 0.128 m, cubic cells h = 0.002 m
    (cell volume 8e-9), z thickness 0.002 (1 cell, empty patches).
  - Ports at bitmap rows 1 and 6 (dataset edge statistics):
    y in [0.096, 0.112] (upper) and y in [0.016, 0.032] (lower), symmetric
    about the centreline y = 0.064.
  - Port extension length outside the AOI is NOT in the dataset; it is
    RECONSTRUCTED here as 0.016 m (8 cells). See RECONSTRUCTION_NOTES.

Boundary conditions (values from the paper):
  inlet:  U = (0.1, 0, 0),  zeroGradient p
  outlet: zeroGradient U,   p = 0
  wall:   U = 0,            zeroGradient p
"""
import os
import sys

H = 0.128          # chamber width [m]
h = 0.002          # cell size [m]
PORT = 0.016       # port width and (reconstructed) extension length [m]

X = [-PORT, 0.0, H, H + PORT]          # 4 x positions
Y = [0.0, 0.016, 0.032, 0.096, 0.112, H]  # 6 y positions
Z = [0.0, h]                            # 2 z positions
NX = [8, 64, 8]                        # cells per x interval
NY = [8, 8, 32, 8, 8]                  # cells per y interval

# Occupied blocks: (ix0, iy0) pairs into X/Y intervals; central column full,
# side columns only at the port bands J1 (lower) and J3 (upper).
BLOCKS = [(1, j) for j in range(5)] + [(0, 1), (0, 3), (2, 1), (2, 3)]


def fmt(v):
    return f"{v:g}"


def write_blockmesh(path):
    # vertex id: (ix, iy, iz) -> ix*12 + iy*2 + iz
    def vid(ix, iy, iz):
        return ix * 12 + iy * 2 + iz

    verts = []
    for x in X:
        for y in Y:
            for z in Z:
                verts.append(f"({fmt(x)} {fmt(y)} {fmt(z)})")

    lines = []
    lines.append("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

scale   1;

vertices
(
""")
    for v in verts:
        lines.append(f"    {v}")
    lines.append(");\n\nblocks\n(")
    for (ix, iy) in BLOCKS:
        # hex ordering: (000,100,110,010, 001,101,111,011)
        v = [
            vid(ix, iy, 0), vid(ix + 1, iy, 0),
            vid(ix + 1, iy + 1, 0), vid(ix, iy + 1, 0),
            vid(ix, iy, 1), vid(ix + 1, iy, 1),
            vid(ix + 1, iy + 1, 1), vid(ix, iy + 1, 1),
        ]
        lines.append(
            f"    hex ({' '.join(map(str, v))}) "
            f"({NX[ix]} {NY[iy]} 1) simpleGrading (1 1 1)"
        )
    lines.append(");\n")

    def quad(a, b, c, d):
        return f"({a} {b} {c} {d})"

    inlet, outlet, walls = [], [], []
    for (ix, iy) in BLOCKS:
        # face x = X[ix]  (west): vertices (ix,iy),(ix,iy+1) at both z
        if ix == 0:
            inlet.append(quad(vid(ix, iy, 0), vid(ix, iy + 1, 0),
                              vid(ix, iy + 1, 1), vid(ix, iy, 1)))
        elif (ix - 1, iy) not in BLOCKS:
            walls.append(quad(vid(ix, iy, 0), vid(ix, iy + 1, 0),
                              vid(ix, iy + 1, 1), vid(ix, iy, 1)))
        # face x = X[ix+1] (east)
        if ix == 2:
            outlet.append(quad(vid(ix + 1, iy, 0), vid(ix + 1, iy + 1, 0),
                               vid(ix + 1, iy + 1, 1), vid(ix + 1, iy, 1)))
        elif (ix + 1, iy) not in BLOCKS:
            walls.append(quad(vid(ix + 1, iy, 0), vid(ix + 1, iy + 1, 0),
                              vid(ix + 1, iy + 1, 1), vid(ix + 1, iy, 1)))
        # face y = Y[iy] (south)
        if (ix, iy - 1) not in BLOCKS:
            walls.append(quad(vid(ix, iy, 0), vid(ix + 1, iy, 0),
                              vid(ix + 1, iy, 1), vid(ix, iy, 1)))
        # face y = Y[iy+1] (north)
        if (ix, iy + 1) not in BLOCKS:
            walls.append(quad(vid(ix, iy + 1, 0), vid(ix + 1, iy + 1, 0),
                              vid(ix + 1, iy + 1, 1), vid(ix, iy + 1, 1)))

    lines.append("boundary\n(")
    lines.append("    inlet\n    {\n        type patch;\n        faces\n        ("
                 )
    for q in inlet:
        lines.append(f"            {q}")
    lines.append("        );\n    }")
    lines.append("    outlet\n    {\n        type patch;\n        faces\n        (")
    for q in outlet:
        lines.append(f"            {q}")
    lines.append("        );\n    }")
    lines.append("    walls\n    {\n        type wall;\n        faces\n        (")
    for q in walls:
        lines.append(f"            {q}")
    lines.append("        );\n    }")
    lines.append("""    frontAndBack
    {
        type empty;
        faces
        (
""")
    for (ix, iy) in BLOCKS:
        lines.append("            " + quad(vid(ix, iy, 0), vid(ix + 1, iy, 0),
                                           vid(ix + 1, iy + 1, 0), vid(ix, iy + 1, 0)))
        lines.append("            " + quad(vid(ix, iy, 1), vid(ix + 1, iy, 1),
                                           vid(ix + 1, iy + 1, 1), vid(ix, iy + 1, 1)))
    lines.append("        );\n    }\n);\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write(path, content):
    with open(path, "w") as f:
        f.write(content)


HEADER = """FoamFile
{
    version     2.0;
    format      ascii;
    class       %s;
    object      %s;
}
"""


def main():
    case = os.path.join(os.path.dirname(__file__), "..", "cases", "four_port_64")
    case = os.path.abspath(case)
    os.makedirs(os.path.join(case, "0"))
    os.makedirs(os.path.join(case, "constant"))
    os.makedirs(os.path.join(case, "system"))

    write_blockmesh(os.path.join(case, "system", "blockMeshDict"))

    write(os.path.join(case, "0", "U"), HEADER % ("volVectorField", "U") + """
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (0.1 0 0);
    }
    outlet
    {
        type            zeroGradient;
    }
    walls
    {
        type            fixedValue;
        value           uniform (0 0 0);
    }
    frontAndBack
    {
        type            empty;
    }
}
""")

    # kinematic pressure p~ = p/rho, [m^2/s^2]
    write(os.path.join(case, "0", "p"), HEADER % ("volScalarField", "p") + """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    walls
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}
""")

    write(os.path.join(case, "system", "controlDict"),
          HEADER % ("dictionary", "controlDict") + """
application     hfdibNeuralDemo;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
""")

    write(os.path.join(case, "system", "fvSchemes"),
          HEADER % ("dictionary", "fvSchemes") + """
ddtSchemes      { default         steadyState; }
gradSchemes     { default         Gauss linear; }
divSchemes      { default         Gauss linear; }
laplacianSchemes{ default         Gauss linear uncorrected; }
interpolationSchemes { default    linear; }
snGradSchemes   { default         uncorrected; }
""")

    write(os.path.join(case, "system", "fvSolution"),
          HEADER % ("dictionary", "fvSolution") + """
solvers {}
""")

    # Physics + training configuration
    write(os.path.join(case, "constant", "physicsProperties"),
          HEADER % ("dictionary", "physicsProperties") + """
// Physics (paper values)
nu               0.01;     // kinematic viscosity [m^2/s]
inletVelocity    0.1;      // [m/s]
outletPressure   0.0;      // kinematic pressure [m^2/s^2]

// Reference scales (nondimensionalization)
Uref             0.1;
Lref             0.128;    // central chamber width [m]
// Pref = nu*Uref/Lref = 7.8125e-3 is computed in code.

// Loss weights
gammaContinuity  1.0;
gammaIb          10.0;

// HFDIB interpolation-point spacings (multiples of local cell size)
d1Factor         1.5;
d2Factor         1.0;

// Training
seed             1234;
restarts         1;
adamSteps        3000;
adamLearningRate 1e-3;
logEvery         50;
""")

    # Bitmap from the published dataset sample (row 0 = top, see
    # topology_origin.json): 0 = fluid, 1 = solid.
    bitmap = [
        [1, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1],
    ]
    write(os.path.join(case, "constant", "topology8x8.csv"),
          "\n".join(",".join(map(str, row)) for row in bitmap) + "\n")

    write(os.path.join(case, "constant", "topology_origin.json"), """{
  "origin": "published dataset",
  "sample_id": "tpfm_unet data/mixer_64.npz inputs[1] (8x8 block majority)",
  "repo": "https://github.com/techMathGroup/tpfm_unet",
  "commit": "c0566779b54877ca46053a8ff44096c3351f4805",
  "convention": "0 = fluid, 1 = solid; row 0 = top of chamber",
  "notes": "single connected fluid component incl. all four ports; porosity 0.531"
}
""")

    write(os.path.join(case, "RECONSTRUCTION_NOTES.md"), """# Reconstruction notes

- Chamber (area of interest), cell size, port positions and port widths are
  recovered from the published dataset (`tpfm_unet/data/coordinates_64.csv`,
  statistics over `data/mixer_64.npz`); see ../../references/UPSTREAM.md.
- The port extension length outside the area of interest is NOT published;
  it is reconstructed here as 0.016 m (8 cells). The authors' complete
  OpenFOAM case (blockMeshDict, solver settings) was not available.
- This case is therefore a reconstruction, not the authors' original case.
""")

    print(f"case written to {case}")


if __name__ == "__main__":
    sys.exit(main())
