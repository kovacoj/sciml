"""Audit the physical and numerical settings of an OpenFOAM case."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return re.sub(r"//.*", "", path.read_text())


def scalar(source: str, pattern: str) -> float | None:
    match = re.search(pattern, source, re.MULTILINE | re.DOTALL)
    return float(match.group(1)) if match else None


def block(source: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\b\s*\{{", source)
    if not match:
        return ""
    start = match.end()
    depth = 1
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index]
    return ""


def patch_values(source: str, field: str) -> dict:
    boundary = block(source, "boundaryField")
    result = {}
    for match in re.finditer(r"(\w+)\s*\{", boundary):
        name = match.group(1)
        body = block(boundary, name)
        kind = re.search(r"\btype\s+(\w+)\s*;", body)
        value = re.search(r"\bvalue\s+uniform\s+([^;]+);", body)
        result[name] = {
            f"{field}_type": kind.group(1) if kind else None,
            f"{field}_value": value.group(1).strip() if value else None,
        }
    return result


def audit(case: Path) -> dict:
    u = text(case / "0/U")
    p = text(case / "0/p")
    transport = text(case / "constant/transportProperties")
    mesh = text(case / "system/blockMeshDict")
    schemes = text(case / "system/fvSchemes")
    solution = text(case / "system/fvSolution")

    vertices_section = re.search(r"\bvertices\s*\((.*?)\)\s*;", mesh, re.DOTALL)
    vertices = [tuple(map(float, values)) for values in re.findall(
        r"\(\s*(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s*\)",
        vertices_section.group(1) if vertices_section else "",
    )]
    cells = [tuple(map(int, values)) for values in re.findall(
        r"hex\s*\([^)]*\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", mesh
    )]
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    nx = max((entry[0] for entry in cells), default=0)
    ny = sum(entry[1] for entry in cells)
    patches = patch_values(u, "U")
    for name, values in patch_values(p, "p").items():
        patches.setdefault(name, {}).update(values)

    uin = scalar(u, r"value\s+uniform\s+\(\s*([\d.eE+-]+)\s+0\s+0\s*\)")
    pout = scalar(p, r"value\s+uniform\s+([\d.eE+-]+)\s*;")
    simple = block(solution, "SIMPLE")
    relaxation = block(solution, "relaxationFactors")
    return {
        "uin": uin,
        "pout": pout,
        "nu": scalar(transport, r"\bnu\s+\[[^]]+\]\s+([\d.eE+-]+)\s*;"),
        "mesh_cells": [nx, ny, sum(a * b * c for a, b, c in cells)],
        "dx": (max(xs) - min(xs)) / nx if xs and nx else None,
        "dy": (max(ys) - min(ys)) / ny if ys and ny else None,
        "xmin": min(xs) if xs else None,
        "xmax": max(xs) if xs else None,
        "ymin": min(ys) if ys else None,
        "ymax": max(ys) if ys else None,
        "patches": patches,
        "div_scheme": re.search(r"div\(phi,U\)\s+([^;]+);", schemes).group(1),
        "grad_scheme": re.search(r"gradSchemes\s*\{\s*default\s+([^;]+);", schemes).group(1),
        "laplacian_scheme": re.search(r"laplacianSchemes\s*\{\s*default\s+([^;]+);", schemes).group(1),
        "simple_settings": {
            key: scalar(simple, rf"\b{key}\s+([\d.eE+-]+)\s*;")
            for key in ("nNonOrthogonalCorrectors", "pRefCell", "pRefValue")
        },
        "relaxation": {
            "p": scalar(relaxation, r"\bp\s+([\d.eE+-]+)\s*;"),
            "U": scalar(relaxation, r"\bU\s+([\d.eE+-]+)\s*;"),
        },
        "article_match": {"uin": uin == 0.1, "pout": pout == 0.0},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.case)
    result["article_match"]["nu"] = result["nu"] == 1.0e-2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if all(result["article_match"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
