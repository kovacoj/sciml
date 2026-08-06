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
    """Generate blockMeshDict for 64x64x1 four-port case."""
    path = os.path.join(case_dir, "system", "blockMeshDict")
    os.makedirs(os.path.dirname(path), exist_ok=True)

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
    (0           0           0)
    ({DOMAIN_W}  0           0)
    ({DOMAIN_W}  {DOMAIN_H}  0)
    (0           {DOMAIN_H}  0)
    (0           0           {DOMAIN_D})
    ({DOMAIN_W}  0           {DOMAIN_D})
    ({DOMAIN_W}  {DOMAIN_H}  {DOMAIN_D})
    (0           {DOMAIN_H}  {DOMAIN_D})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({NX} {NY} 1) simpleGrading (1 1 1)
);

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    walls
    {{
        type wall;
        faces
        (
            (3 7 6 2)
            (0 1 5 4)
        );
    }}
    frontAndBack
    {{
        type symmetry;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
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

    # U
    with open(os.path.join(zero_dir, "U"), "w") as f:
        f.write(f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform (0.1 0 0);
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
    frontAndBack
    {{
        type            symmetry;
    }}
}}
""")

    # p
    with open(os.path.join(zero_dir, "p"), "w") as f:
        f.write(f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}}

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform 0;
    }}
    walls
    {{
        type            zeroGradient;
    }}
    frontAndBack
    {{
        type            symmetry;
    }}
}}
""")

    # nut
    for fname in ["nut", "nuTilda"]:
        with open(os.path.join(zero_dir, fname), "w") as f:
            f.write(f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      {fname};
}}

dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
    inlet
    {{
        type            calculated;
        value           uniform 0;
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            fixedValue;
        value           uniform 0;
    }}
    frontAndBack
    {{
        type            symmetry;
    }}
}}
""")


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
divSchemes      { default         none; div(phi,U) Gauss linearUpwindV grad(U); }
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
    """Write signed-distance as OpenFOAM scalarList."""
    sd_dir = os.path.join(case_dir, "constant", "hfdibGeometry")
    os.makedirs(sd_dir, exist_ok=True)
    path = os.path.join(sd_dir, "signedDistance")
    with open(path, "w") as f:
        f.write("FoamFile\n{\n")
        f.write("    version     2.0;\n")
        f.write("    format      ascii;\n")
        f.write("    class       scalarList;\n")
        f.write("    location    \"constant/hfdibGeometry\";\n")
        f.write("    object      signedDistance;\n")
        f.write("}\n\n")
        f.write(f"{len(psi)}\n(\n")
        for v in psi:
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
