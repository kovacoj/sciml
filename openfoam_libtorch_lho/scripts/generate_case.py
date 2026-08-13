#!/usr/bin/env python3
"""
Generate OpenFOAM case with specified mesh resolution
Usage: python3 scripts/generate_case.py --cells 256 --half-width 8 --output cases/lho_256
"""

import argparse
import os
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Generate LHO case')
    parser.add_argument('--cells', type=int, default=256, help='Number of cells in x direction')
    parser.add_argument('--half-width', type=float, default=8.0, help='Half-width L of domain')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    ROOT = Path(__file__).parent.parent
    template = ROOT / 'cases' / 'lho_template'
    output = Path(args.output)

    if output.exists():
        print(f"Removing existing directory: {output}")
        shutil.rmtree(output)

    shutil.copytree(template, output)

    blockMeshDict = output / 'system' / 'blockMeshDict'
    content = blockMeshDict.read_text()

    # Update vertices
    L = args.half_width
    vertices = f"""vertices
(
    ({-L} -0.5 -0.5)   // 0
    ( {L} -0.5 -0.5)   // 1
    ( {L}  0.5 -0.5)   // 2
    ({-L}  0.5 -0.5)   // 3
    ({-L} -0.5  0.5)   // 4
    ( {L} -0.5  0.5)   // 5
    ( {L}  0.5  0.5)   // 6
    ({-L}  0.5  0.5)   // 7
);"""
    content = content.replace(
        """vertices
(
    (-8 -0.5 -0.5)   // 0
    ( 8 -0.5 -0.5)   // 1
    ( 8  0.5 -0.5)   // 2
    (-8  0.5 -0.5)   // 3
    (-8 -0.5  0.5)   // 4
    ( 8 -0.5  0.5)   // 5
    ( 8  0.5  0.5)   // 6
    (-8  0.5  0.5)   // 7
);""",
        vertices
    )

    # Update blocks
    blocks = f"""blocks
(
    hex (0 1 2 3 4 5 6 7) ({args.cells} 1 1) simpleGrading (1 1 1)
);"""
    content = content.replace(
        """blocks
(
    hex (0 1 2 3 4 5 6 7) (256 1 1) simpleGrading (1 1 1)
);""",
        blocks
    )

    blockMeshDict.write_text(content)

    print(f"Generated case at {output}")
    print(f"  Cells: {args.cells}")
    print(f"  Domain: [{-L}, {L}]")

if __name__ == '__main__':
    main()
