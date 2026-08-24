"""Exact pixel-union fitted meshes for rugged 64x64 four-port topologies."""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from .brinkman_mesh import DOMAIN_MARKER, INLET_MARKER, OUTLET_MARKER, WALL_MARKER


H = 0.002
ROI_N = 64
STUB_CELLS = 8
PORT_ROWS = ((10, 14), (50, 54))


def port_mask() -> np.ndarray:
    mask = np.zeros(ROI_N, dtype=bool)
    for start, stop in PORT_ROWS:
        mask[start:stop] = True
    return mask


def all_ports_connected(fluid: np.ndarray) -> bool:
    labels, _ = label(fluid)
    ids = []
    for start, stop in PORT_ROWS:
        ids.extend(labels[start:stop, 0].tolist())
        ids.extend(labels[start:stop, -1].tolist())
    return bool(ids) and min(ids) > 0 and len(set(ids)) == 1


def shortest_left_right(fluid: np.ndarray) -> int | None:
    starts = [(row, 0) for start, stop in PORT_ROWS for row in range(start, stop)]
    targets = {(row, ROI_N - 1) for start, stop in PORT_ROWS for row in range(start, stop)}
    queue = deque((point, 0) for point in starts if fluid[point])
    visited = {point for point, _ in queue}
    while queue:
        (row, column), distance = queue.popleft()
        if (row, column) in targets:
            return distance
        for candidate in ((row - 1, column), (row + 1, column),
                          (row, column - 1), (row, column + 1)):
            r, c = candidate
            if (0 <= r < ROI_N and 0 <= c < ROI_N and fluid[r, c]
                    and candidate not in visited):
                visited.add(candidate); queue.append((candidate, distance + 1))
    return None


def descriptors(lambda_field: np.ndarray) -> dict:
    fluid = lambda_field < 0.5
    coarse = fluid.reshape(8, 8, 8, 8).mean(axis=(1, 3)) >= 0.5
    fluid_labels, fluid_components = label(fluid)
    solid_labels, solid_components = label(~fluid)
    boundary_solid = set(np.concatenate((
        solid_labels[0], solid_labels[-1], solid_labels[:, 0], solid_labels[:, -1]
    )).tolist()) - {0}
    islands = len((set(range(1, solid_components + 1)) - boundary_solid))
    interface = int(np.count_nonzero(fluid[:, 1:] != fluid[:, :-1])
                    + np.count_nonzero(fluid[1:, :] != fluid[:-1, :]))
    degrees = np.zeros_like(coarse, dtype=int)
    for row, column in np.argwhere(coarse):
        degrees[row, column] = sum(
            0 <= r < 8 and 0 <= c < 8 and coarse[r, c]
            for r, c in ((row - 1, column), (row + 1, column),
                         (row, column - 1), (row, column + 1))
        )
    shortest = shortest_left_right(fluid)
    fluid_distances = distance_transform_edt(fluid)[fluid]
    throat = 2.0 * float(fluid_distances.min()) if fluid_distances.size else 0.0
    center = fluid[20:44, 20:44]
    asymmetry = float(np.mean(fluid != np.fliplr(fluid)))
    parallel_fraction = float(
        0.5 * (fluid[8:16, :].mean() + fluid[48:56, :].mean())
    )
    return {
        "fluid_fraction": float(fluid.mean()),
        "solid_fraction": float((~fluid).mean()),
        "interface_length_pixels": interface,
        "fluid_connected_components": int(fluid_components),
        "solid_connected_components": int(solid_components),
        "solid_island_count": islands,
        "coarse_fluid_cells": int(coarse.sum()),
        "skeleton_length_proxy": int(coarse.sum()),
        "skeleton_branch_points": int(np.count_nonzero(degrees >= 3)),
        "skeleton_end_points": int(np.count_nonzero(degrees == 1)),
        "mean_fluid_distance_to_wall": float(distance_transform_edt(fluid)[fluid].mean()),
        "minimum_throat_width_estimate_pixels": throat,
        "left_right_shortest_path_length": shortest,
        "tortuosity": float(shortest / 63.0) if shortest is not None else None,
        "central_fluid_fraction": float(center.mean()),
        "asymmetry": asymmetry,
        "parallel_band_fluid_fraction": parallel_fraction,
        "number_of_alternative_paths_estimate": int(max(0, np.count_nonzero(degrees >= 3) - 1)),
        "all_ports_connected": all_ports_connected(fluid),
    }


def full_fluid_mask(binary_fluid: np.ndarray) -> np.ndarray:
    full = np.zeros((ROI_N, ROI_N + 2 * STUB_CELLS), dtype=bool)
    full[:, STUB_CELLS:STUB_CELLS + ROI_N] = binary_fluid
    ports = port_mask()
    full[ports, :STUB_CELLS] = True
    full[ports, STUB_CELLS + ROI_N:] = True
    return full


def write_pixel_union_mesh(path: Path, binary_fluid: np.ndarray) -> Path:
    fluid = full_fluid_mask(binary_fluid)
    ny, nx = fluid.shape
    used = set()
    for row, column in np.argwhere(fluid):
        used.update(((column, row), (column + 1, row),
                     (column, row + 1), (column + 1, row + 1)))
    node_ids = {point: index + 1 for index, point in enumerate(
        sorted(used, key=lambda item: (item[1], item[0])))
    }
    triangles = []
    for row, column in np.argwhere(fluid):
        n00, n10 = node_ids[(column, row)], node_ids[(column + 1, row)]
        n01, n11 = node_ids[(column, row + 1)], node_ids[(column + 1, row + 1)]
        triangles.extend(((n00, n10, n11), (n00, n11, n01)))
    edges = Counter()
    for triangle in triangles:
        for edge in ((triangle[0], triangle[1]), (triangle[1], triangle[2]),
                     (triangle[2], triangle[0])):
            edges[tuple(sorted(edge))] += 1
    reverse = {value: key for key, value in node_ids.items()}
    boundary = []
    for edge, count in edges.items():
        if count != 1:
            continue
        columns = [reverse[node][0] for node in edge]
        marker = INLET_MARKER if all(column == 0 for column in columns) else (
            OUTLET_MARKER if all(column == nx for column in columns) else WALL_MARKER
        )
        boundary.append((edge, marker))
    nodes = [
        (node_id, (point[0] - STUB_CELLS) * H, point[1] * H)
        for point, node_id in sorted(node_ids.items(), key=lambda item: item[1])
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
        '1 1 "inlets"', '1 2 "outlets"', '1 3 "walls"', '2 10 "fluid"',
        "$EndPhysicalNames", "$Nodes", str(len(nodes)),
    ]
    text.extend(f"{index} {x:.16g} {y:.16g} 0" for index, x, y in nodes)
    text.extend(("$EndNodes", "$Elements", str(len(elements)), *elements,
                 "$EndElements"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n")
    return path
