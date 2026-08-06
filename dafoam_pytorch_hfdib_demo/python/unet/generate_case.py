"""Generate the 64x64 four-port case and topology dataset.

Creates:
  cases/four_port_64x64/  — OpenFOAM case template
  topologies/four_port_64/ — dataset of topologies with converged HFDIB solutions
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))


# Domain parameters
DOMAIN_W = 0.128  # x
DOMAIN_H = 0.128  # y
DOMAIN_D = 0.002  # z extrusion
NX = 64
NY = 64
DX = DOMAIN_W / NX
DY = DOMAIN_H / NY
H = np.sqrt(DX * DY)

# Port locations (bitmap rows 1 and 6, scaled to 64 grid)
PORT_BANDS = [(8, 16), (48, 56)]  # y-cell ranges for inlet/outlet

# Design region for topology bitmap
DESIGN_BOUNDS = (0.0, DOMAIN_W, 0.0, DOMAIN_H)


def generate_blockmesh_dict(case_dir: str):
    """Generate blockMeshDict for 64x64x1 four-port case with separate port patches."""
    path = os.path.join(case_dir, "system", "blockMeshDict")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Port y-bands on the 64-cell grid: rows 8-15 (lower) and 48-55 (upper)
    y_lo_start = 8 * DY    # 0.016
    y_lo_end = 16 * DY      # 0.032
    y_hi_start = 48 * DY    # 0.096
    y_hi_end = 56 * DY       # 0.112

    content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

scale   1;

vertices
(
    // z=0 plane (indices 0-11)
    (0              0              0)
    ({DOMAIN_W}    0              0)
    ({DOMAIN_W}    {y_lo_start}   0)
    (0              {y_lo_start}   0)
    (0              {y_lo_end}     0)
    ({DOMAIN_W}    {y_lo_end}     0)
    ({DOMAIN_W}    {y_hi_start}   0)
    (0              {y_hi_start}   0)
    (0              {y_hi_end}     0)
    ({DOMAIN_W}    {y_hi_end}     0)
    ({DOMAIN_W}    {DOMAIN_H}     0)
    (0              {DOMAIN_H}     0)
    // z={DOMAIN_D} plane (indices 12-23)
    (0              0              {DOMAIN_D})
    ({DOMAIN_W}    0              {DOMAIN_D})
    ({DOMAIN_W}    {y_lo_start}   {DOMAIN_D})
    (0              {y_lo_start}   {DOMAIN_D})
    (0              {y_lo_end}     {DOMAIN_D})
    ({DOMAIN_W}    {y_lo_end}     {DOMAIN_D})
    ({DOMAIN_W}    {y_hi_start}   {DOMAIN_D})
    (0              {y_hi_start}   {DOMAIN_D})
    (0              {y_hi_end}     {DOMAIN_D})
    ({DOMAIN_W}    {y_hi_end}     {DOMAIN_D})
    ({DOMAIN_W}    {DOMAIN_H}     {DOMAIN_D})
    (0              {DOMAIN_H}     {DOMAIN_D})
);

blocks
(
    // y = 0.000 ... 0.016
    hex (0 1 2 3 12 13 14 15) (64 8 1) simpleGrading (1 1 1)
    // y = 0.016 ... 0.032: lower port strip
    hex (3 2 5 4 15 14 17 16) (64 8 1) simpleGrading (1 1 1)
    // y = 0.032 ... 0.096
    hex (4 5 6 7 16 17 18 19) (64 32 1) simpleGrading (1 1 1)
    // y = 0.096 ... 0.112: upper port strip
    hex (7 6 9 8 19 18 21 20) (64 8 1) simpleGrading (1 1 1)
    // y = 0.112 ... 0.128
    hex (8 9 10 11 20 21 22 23) (64 8 1) simpleGrading (1 1 1)
);

boundary
(
    inletLower
    {{
        type patch;
        faces
        (
            (3 15 16 4)
        );
    }}
    inletUpper
    {{
        type patch;
        faces
        (
            (7 19 20 8)
        );
    }}
    outletLower
    {{
        type patch;
        faces
        (
            (2 5 17 14)
        );
    }}
    outletUpper
    {{
        type patch;
        faces
        (
            (6 9 21 18)
        );
    }}
    sideWalls
    {{
        type wall;
        faces
        (
            (0 12 15 3)
            (1 2 14 13)
            (4 16 19 7)
            (5 6 18 17)
            (8 20 23 11)
            (9 10 22 21)
        );
    }}
    topBottomWalls
    {{
        type wall;
        faces
        (
            (0 1 13 12)
            (11 23 22 10)
        );
    }}
    frontAndBack
    {{
        type symmetry;
        faces
        (
            (0 3 2 1)
            (3 4 5 2)
            (4 7 6 5)
            (7 8 9 6)
            (8 11 10 9)
            (12 13 14 15)
            (15 14 17 16)
            (16 17 18 19)
            (19 18 21 20)
            (20 21 22 23)
        );
    }}
);
"""
    with open(path, "w") as f:
        f.write(content)


def generate_field_files(case_dir: str):
    """Generate 0/U, 0/p, 0/nut, 0/nuTilda for the four-port case."""
    zero_dir = os.path.join(case_dir, "0")
    os.makedirs(zero_dir, exist_ok=True)

    patch_names = ["inletLower", "inletUpper", "outletLower", "outletUpper",
                  "sideWalls", "topBottomWalls", "frontAndBack"]

    # U: inlet patches = fixedValue (0.1,0,0), outlet = zeroGradient, walls = fixedValue 0
    inlet_patches = ["inletLower", "inletUpper"]
    outlet_patches = ["outletLower", "outletUpper"]
    wall_patches = ["sideWalls", "topBottomWalls"]

    with open(os.path.join(zero_dir, "U"), "w") as f:
        f.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n")
        f.write("    class       volVectorField;\n    object      U;\n}\n\n")
        f.write("dimensions      [0 1 -1 0 0 0 0];\n")
        f.write("internalField   uniform (0 0 0);\n\nboundaryField\n{\n")
        for p in inlet_patches:
            f.write("    " + p + "\n    {\n        type            fixedValue;\n")
            f.write("        value           uniform (0.1 0 0);\n    }\n")
        for p in outlet_patches:
            f.write("    " + p + "\n    {\n        type            zeroGradient;\n    }\n")
        for p in wall_patches:
            f.write("    " + p + "\n    {\n        type            fixedValue;\n")
            f.write("        value           uniform (0 0 0);\n    }\n")
        f.write("    frontAndBack\n    {\n        type            symmetry;\n    }\n}\n")

    # p: inlet = zeroGradient, outlet = fixedValue 0, walls = zeroGradient
    with open(os.path.join(zero_dir, "p"), "w") as f:
        f.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n")
        f.write("    class       volScalarField;\n    object      p;\n}\n\n")
        f.write("dimensions      [0 2 -2 0 0 0 0];\n")
        f.write("internalField   uniform 0;\n\nboundaryField\n{\n")
        for p in inlet_patches:
            f.write("    " + p + "\n    {\n        type            zeroGradient;\n    }\n")
        for p in outlet_patches:
            f.write("    " + p + "\n    {\n        type            fixedValue;\n")
            f.write("        value           uniform 0;\n    }\n")
        for p in wall_patches:
            f.write("    " + p + "\n    {\n        type            zeroGradient;\n    }\n")
        f.write("    frontAndBack\n    {\n        type            symmetry;\n    }\n}\n")

    # nut, nuTilda: calculated at inlet, zeroGradient outlet, fixedValue 0 walls
    for fname in ["nut", "nuTilda"]:
        with open(os.path.join(zero_dir, fname), "w") as f:
            f.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n")
            f.write("    class       volScalarField;\n    object      " + fname + ";\n}\n\n")
            f.write("dimensions      [0 2 -1 0 0 0 0];\n")
            f.write("internalField   uniform 0;\n\nboundaryField\n{\n")
            for p in inlet_patches:
                f.write("    " + p + "\n    {\n        type            calculated;\n")
                f.write("        value           uniform 0;\n    }\n")
            for p in outlet_patches:
                f.write("    " + p + "\n    {\n        type            zeroGradient;\n    }\n")
            for p in wall_patches:
                f.write("    " + p + "\n    {\n        type            fixedValue;\n")
                f.write("        value           uniform 0;\n    }\n")
            f.write("    frontAndBack\n    {\n        type            symmetry;\n    }\n}\n")


def generate_control_dict(case_dir: str):
    """Generate controlDict, fvSchemes, fvSolution, decomposeParDict."""
    sys_dir = os.path.join(case_dir, "system")
    os.makedirs(sys_dir, exist_ok=True)

    with open(os.path.join(sys_dir, "controlDict"), "w") as f:
        f.write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     simpleDAFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         5000;
deltaT          1;
writeControl    timeStep;
writeInterval   5000;
purgeWrite      0;
writeFormat     ascii;
writePrecision  16;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable true;

DebugSwitches
{
    SolverPerformance 0;
}
""")

    with open(os.path.join(sys_dir, "fvSchemes"), "w") as f:
        f.write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes      { default         steadyState; }
gradSchemes     { default         Gauss linear; }
divSchemes      { default         none; div(phi,U) Gauss linearUpwindV grad(U); div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes{ default         Gauss linear corrected; }
interpolationSchemes { default    linear; }
snGradSchemes   { default         corrected; }
""")

    with open(os.path.join(sys_dir, "fvSolution"), "w") as f:
        f.write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    pRefCell 0;
    pRefValue 0;
}

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-13;
        relTol          0.001;
        smoother        GaussSeidel;
        nSweeps         2;
    }
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-13;
        relTol          0.001;
        nSweeps         1;
    }
}

relaxationFactors
{
    fields  { p 0.3; }
    equations { U 0.7; }
}
""")

    with open(os.path.join(sys_dir, "decomposeParDict"), "w") as f:
        f.write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}

numberOfSubdomains 1;
method          scotch;
""")


def generate_transport_properties(case_dir: str):
    """Generate transportProperties and turbulenceProperties."""
    const_dir = os.path.join(case_dir, "constant")
    os.makedirs(const_dir, exist_ok=True)

    with open(os.path.join(const_dir, "transportProperties"), "w") as f:
        f.write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}

transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] 1.0e-2;
Pr              0.7;
Prt             0.85;
""")

    with open(os.path.join(const_dir, "turbulenceProperties"), "w") as f:
        f.write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}

simulationType  RAS;
RAS
{
    RASModel        dummy;
    printCoeffs     off;
}
""")


def create_four_port_case(case_dir: str):
    """Create the four_port_64x64 case template."""
    os.makedirs(case_dir, exist_ok=True)
    generate_blockmesh_dict(case_dir)
    generate_field_files(case_dir)
    generate_control_dict(case_dir)
    generate_transport_properties(case_dir)


def mask_to_signed_distance_64(mask: np.ndarray) -> np.ndarray:
    """Convert 8x8 bitmap to 64x64 signed distance field."""
    # Upsample mask to 64x64 grid
    solid_grid = np.zeros((NY, NX), dtype=bool)
    for j in range(NY):
        for i in range(NX):
            mj = min(int(j * 8 / NY), 7)
            mi = min(int(i * 8 / NX), 7)
            solid_grid[j, i] = (mask[mj, mi] == 1)

    dist_to_solid = distance_transform_edt(~solid_grid, sampling=(DY, DX))
    dist_to_fluid = distance_transform_edt(solid_grid, sampling=(DY, DX))
    psi = dist_to_solid - dist_to_fluid

    return psi


def write_signed_distance_file(case_dir: str, psi: np.ndarray):
    """Write signed-distance as OpenFOAM scalarList (flattened to cell order)."""
    sd_dir = os.path.join(case_dir, "constant", "hfdibGeometry")
    os.makedirs(sd_dir, exist_ok=True)
    path = os.path.join(sd_dir, "signedDistance")

    psi_flat = np.asarray(psi, dtype=np.float64).reshape(-1, order="C")
    assert psi_flat.size == 4096, f"Expected 4096 values, got {psi_flat.size}"

    with open(path, "w") as f:
        f.write("FoamFile\n{\n")
        f.write("    version     2.0;\n")
        f.write("    format      ascii;\n")
        f.write("    class       scalarList;\n")
        f.write('    location    "constant/hfdibGeometry";\n')
        f.write("    object      signedDistance;\n")
        f.write("}\n\n")
        f.write(f"{psi_flat.size}\n(\n")
        for v in psi_flat:
            f.write(f"{v:.16e}\n")
        f.write(")\n")


def main():
    """Generate the four-port case template and topology dataset."""
    project_root = Path(__file__).resolve().parents[2]
    case_dir = project_root / "cases" / "four_port_64x64"

    print("[gen] Creating four_port_64x64 case...")
    create_four_port_case(str(case_dir))

    print(f"[gen] Case created at {case_dir}")
    print(f"[gen] Run blockMesh + checkMesh to verify")

    # Also generate a few topology masks
    topo_dir = project_root / "topologies" / "four_port_64"
    topo_dir.mkdir(parents=True, exist_ok=True)

    topologies = {
        "topology_000": np.array([
            [1,1,1,0,0,1,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [1,1,1,0,0,1,1,1],
        ]),
        "topology_001": np.array([
            [1,1,1,0,0,1,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,1,1,0,0,0],
            [1,1,0,1,1,0,1,1],
            [1,1,0,1,1,0,1,1],
            [0,0,0,1,1,0,0,0],
            [0,0,0,0,0,0,0,0],
            [1,1,1,0,0,1,1,1],
        ]),
        "topology_002": np.array([
            [1,1,1,0,0,1,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [1,1,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [1,1,1,0,0,1,1,1],
        ]),
        "topology_003": np.array([
            [1,1,1,0,0,1,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,1,1,0,0],
            [1,1,0,0,1,1,0,0],
            [1,1,0,0,0,0,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [1,1,1,0,0,1,1,1],
        ]),
    }

    for tid, mask in topologies.items():
        td = topo_dir / tid
        td.mkdir(exist_ok=True)
        np.save(td / "mask.npy", mask)
        with open(td / "topology.json", "w") as f:
            json.dump({"topology_id": tid}, f)

    with open(topo_dir / "dataset.json", "w") as f:
        json.dump({
            "case_template": "cases/four_port_64x64",
            "design_bounds": [0.0, 0.128, 0.0, 0.128],
            "mask_convention": "1=solid,0=fluid",
            "grid_size": [64, 64],
            "port_bands": [[8, 16], [48, 56]],
        }, f, indent=2)

    print(f"[gen] {len(topologies)} topologies written to {topo_dir}")


if __name__ == "__main__":
    main()
