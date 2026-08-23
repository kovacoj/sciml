"""Structured Gmsh mesh with segmented two-port boundary markers."""

from __future__ import annotations

from pathlib import Path

from .brinkman_topologies import DOMAIN_SIZE, PORT_INTERVALS


INLET_MARKER = 1
OUTLET_MARKER = 2
WALL_MARKER = 3
DOMAIN_MARKER = 10


def _is_port(y: float) -> bool:
    return any(lower <= y < upper for lower, upper in PORT_INTERVALS)


def write_segmented_square_mesh(path: Path, resolution: int) -> Path:
    """Write a Gmsh v2 triangular mesh with interval-specific port markers."""
    if resolution <= 0 or resolution % 16:
        raise ValueError("resolution must be a positive multiple of 16")
    path.parent.mkdir(parents=True, exist_ok=True)
    spacing = DOMAIN_SIZE / resolution

    def node(i: int, j: int) -> int:
        return j * (resolution + 1) + i + 1

    nodes = [
        (node(i, j), i * spacing, j * spacing)
        for j in range(resolution + 1) for i in range(resolution + 1)
    ]
    elements = []
    element_id = 1

    def line(a: int, b: int, marker: int) -> None:
        nonlocal element_id
        elements.append(f"{element_id} 1 2 {marker} {marker} {a} {b}")
        element_id += 1

    for j in range(resolution):
        midpoint = (j + 0.5) * spacing
        line(node(0, j), node(0, j + 1), INLET_MARKER if _is_port(midpoint) else WALL_MARKER)
        line(node(resolution, j), node(resolution, j + 1), OUTLET_MARKER if _is_port(midpoint) else WALL_MARKER)
    for i in range(resolution):
        line(node(i, 0), node(i + 1, 0), WALL_MARKER)
        line(node(i, resolution), node(i + 1, resolution), WALL_MARKER)
    for j in range(resolution):
        for i in range(resolution):
            n00, n10 = node(i, j), node(i + 1, j)
            n01, n11 = node(i, j + 1), node(i + 1, j + 1)
            elements.append(f"{element_id} 2 2 {DOMAIN_MARKER} {DOMAIN_MARKER} {n00} {n10} {n11}")
            element_id += 1
            elements.append(f"{element_id} 2 2 {DOMAIN_MARKER} {DOMAIN_MARKER} {n00} {n11} {n01}")
            element_id += 1
    text = [
        "$MeshFormat", "2.2 0 8", "$EndMeshFormat",
        "$PhysicalNames", "4",
        '1 1 "inlet"', '1 2 "outlet"', '1 3 "walls"', '2 10 "domain"',
        "$EndPhysicalNames", "$Nodes", str(len(nodes)),
    ]
    text.extend(f"{index} {x:.16g} {y:.16g} 0" for index, x, y in nodes)
    text.extend(("$EndNodes", "$Elements", str(len(elements)), *elements, "$EndElements"))
    path.write_text("\n".join(text) + "\n")
    return path
