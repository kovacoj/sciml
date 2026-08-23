"""Conforming fitted mesh for analytic axis-aligned channel Topology A."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .brinkman_mesh import DOMAIN_MARKER, INLET_MARKER, OUTLET_MARKER, WALL_MARKER


def _axis_coordinates(size: float, nominal_h: float, features: tuple[float, ...]) -> np.ndarray:
    regular = np.arange(0.0, size + nominal_h / 2.0, nominal_h)
    return np.unique(np.concatenate((regular, np.asarray(features), [0.0, size])))


def write_fitted_topology_a(path: Path, nominal_h: float = 0.004) -> Path:
    """Triangulate the exact union of two channels and one vertical bridge."""
    size = 0.128
    lower = (0.020, 0.028)
    upper = (0.100, 0.108)
    bridge = (0.060, 0.068)
    x_values = _axis_coordinates(size, nominal_h, bridge)
    y_values = _axis_coordinates(size, nominal_h, (*lower, *upper))

    def fluid(x: float, y: float) -> bool:
        horizontal = lower[0] < y < lower[1] or upper[0] < y < upper[1]
        vertical = bridge[0] < x < bridge[1] and lower[0] < y < upper[1]
        return horizontal or vertical

    cells = []
    for j in range(len(y_values) - 1):
        for i in range(len(x_values) - 1):
            midpoint = (
                0.5 * (x_values[i] + x_values[i + 1]),
                0.5 * (y_values[j] + y_values[j + 1]),
            )
            if fluid(*midpoint):
                cells.append((i, j))
    used = set()
    for i, j in cells:
        used.update(((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)))
    node_ids = {key: index + 1 for index, key in enumerate(sorted(used, key=lambda p: (p[1], p[0])))}
    triangles = []
    for i, j in cells:
        n00, n10 = node_ids[(i, j)], node_ids[(i + 1, j)]
        n01, n11 = node_ids[(i, j + 1)], node_ids[(i + 1, j + 1)]
        triangles.extend(((n00, n10, n11), (n00, n11, n01)))
    edge_counts = Counter()
    for triangle in triangles:
        for edge in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge_counts[tuple(sorted(edge))] += 1
    reverse = {value: key for key, value in node_ids.items()}
    boundary = []
    for edge, count in edge_counts.items():
        if count != 1:
            continue
        coordinates = [
            (x_values[reverse[node][0]], y_values[reverse[node][1]]) for node in edge
        ]
        if all(np.isclose(point[0], 0.0) for point in coordinates):
            marker = INLET_MARKER
        elif all(np.isclose(point[0], size) for point in coordinates):
            marker = OUTLET_MARKER
        else:
            marker = WALL_MARKER
        boundary.append((edge, marker))

    nodes = [
        (node_id, x_values[key[0]], y_values[key[1]])
        for key, node_id in sorted(node_ids.items(), key=lambda item: item[1])
    ]
    elements = []
    element_id = 1
    for edge, marker in boundary:
        elements.append(f"{element_id} 1 2 {marker} {marker} {edge[0]} {edge[1]}")
        element_id += 1
    for triangle in triangles:
        elements.append(
            f"{element_id} 2 2 {DOMAIN_MARKER} {DOMAIN_MARKER} "
            f"{triangle[0]} {triangle[1]} {triangle[2]}"
        )
        element_id += 1
    text = [
        "$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "4",
        '1 1 "inlet"', '1 2 "outlet"', '1 3 "walls"', '2 10 "fluid"',
        "$EndPhysicalNames", "$Nodes", str(len(nodes)),
    ]
    text.extend(f"{index} {x:.16g} {y:.16g} 0" for index, x, y in nodes)
    text.extend(("$EndNodes", "$Elements", str(len(elements)), *elements, "$EndElements"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n")
    return path
