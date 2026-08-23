"""Conforming fitted mesh for analytic axis-aligned channel Topology A."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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


@dataclass(frozen=True)
class FittedTopology:
    name: str
    polylines: tuple[tuple[tuple[float, float], ...], ...]
    width: float = 0.012
    chambers: tuple[tuple[float, float, float, float], ...] = ()

    def contains(self, x: float, y: float) -> bool:
        for xmin, xmax, ymin, ymax in self.chambers:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
        point = np.array((x, y))
        for line in self.polylines:
            for start, stop in zip(line[:-1], line[1:]):
                a, b = np.array(start), np.array(stop)
                fraction = np.clip(np.dot(point - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
                if np.linalg.norm(point - (a + fraction * (b - a))) <= self.width / 2:
                    return True
        return False


def make_topology(name: str) -> FittedTopology:
    low, high = 0.024, 0.104
    definitions = {
        "B": FittedTopology("B", (((0, low), (.052, low), (.076, high), (.128, high)),
                                     ((0, high), (.052, high), (.076, low), (.128, low))),
                            chambers=((.054, .074, .054, .074),)),
        "C": FittedTopology("C", (((0, low), (.044, low), (.058, .064), (.090, .064), (.108, low), (.128, low)),
                                     ((0, high), (.044, high), (.058, .064), (.090, .064), (.108, high), (.128, high))),
                            width=.014),
        "D": FittedTopology("D", (((0, low), (.036, low), (.052, .048), (.082, .048), (.098, low), (.128, low)),
                                     ((0, high), (.030, high), (.050, .082), (.086, .082), (.106, high), (.128, high))),
                            width=.012),
        "E": FittedTopology("E", (((0, low), (.048, low), (.058, .050)),
                                     ((0, high), (.048, high), (.058, .078)),
                                     ((.070, .050), (.082, low), (.128, low)),
                                     ((.070, .078), (.082, high), (.128, high))),
                            width=.014, chambers=((.052, .076, .044, .084),)),
        "F": FittedTopology("F", (((0, low), (.128, low)), ((0, high), (.128, high)),
                                     ((.048, low), (.070, .054), (.094, .054), (.108, high))),
                            width=.012),
    }
    if name not in definitions:
        raise ValueError(f"unknown topology {name!r}")
    return definitions[name]


def write_fitted_topology(path: Path, name: str, nominal_h: float = 0.002) -> Path:
    if name == "A":
        return write_fitted_topology_a(path, nominal_h)
    topology = make_topology(name)
    size = .128
    values = _axis_coordinates(size, nominal_h, ())
    cells = [(i, j) for j in range(len(values)-1) for i in range(len(values)-1)
             if topology.contains((values[i]+values[i+1])/2, (values[j]+values[j+1])/2)]
    used = set()
    for i, j in cells:
        used.update(((i,j),(i+1,j),(i,j+1),(i+1,j+1)))
    node_ids = {key: index+1 for index,key in enumerate(sorted(used,key=lambda p:(p[1],p[0])))}
    triangles = []
    for i,j in cells:
        n00,n10,n01,n11=node_ids[(i,j)],node_ids[(i+1,j)],node_ids[(i,j+1)],node_ids[(i+1,j+1)]
        triangles.extend(((n00,n10,n11),(n00,n11,n01)))
    edge_counts=Counter()
    for triangle in triangles:
        for edge in ((triangle[0],triangle[1]),(triangle[1],triangle[2]),(triangle[2],triangle[0])):
            edge_counts[tuple(sorted(edge))]+=1
    reverse={value:key for key,value in node_ids.items()}; boundary=[]
    for edge,count in edge_counts.items():
        if count != 1: continue
        xs=[values[reverse[node][0]] for node in edge]
        marker=INLET_MARKER if all(np.isclose(x,0) for x in xs) else OUTLET_MARKER if all(np.isclose(x,size) for x in xs) else WALL_MARKER
        boundary.append((edge,marker))
    nodes=[(node_id,values[key[0]],values[key[1]]) for key,node_id in sorted(node_ids.items(),key=lambda item:item[1])]
    elements=[]; element_id=1
    for edge,marker in boundary:
        elements.append(f"{element_id} 1 2 {marker} {marker} {edge[0]} {edge[1]}"); element_id+=1
    for triangle in triangles:
        elements.append(f"{element_id} 2 2 {DOMAIN_MARKER} {DOMAIN_MARKER} {' '.join(map(str,triangle))}"); element_id+=1
    text=["$MeshFormat","2.2 0 8","$EndMeshFormat","$PhysicalNames","4",'1 1 "inlet"','1 2 "outlet"','1 3 "walls"','2 10 "fluid"',"$EndPhysicalNames","$Nodes",str(len(nodes))]
    text.extend(f"{index} {x:.16g} {y:.16g} 0" for index,x,y in nodes)
    text.extend(("$EndNodes","$Elements",str(len(elements)),*elements,"$EndElements"))
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text("\n".join(text)+"\n")
    return path
