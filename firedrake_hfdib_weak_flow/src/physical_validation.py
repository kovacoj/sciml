"""Hard physical-compatibility gates for benchmark geometry."""

from __future__ import annotations

from collections import deque

import numpy as np


def _boundary_cells(geometry, patches) -> np.ndarray:
    selected = np.zeros(geometry.lambda_field.shape, dtype=bool)
    tolerance = max(1.0e-12, geometry.spacing * 1.0e-10)
    for patch in patches:
        if patch.side in {"left", "right"}:
            tangent = geometry.y
            edge = (slice(None), 0 if patch.side == "left" else -1)
        else:
            tangent = geometry.x
            edge = (0 if patch.side == "bottom" else -1, slice(None))
        interval_selected = np.zeros(tangent.shape, dtype=bool)
        for interval in patch.intervals:
            interval_selected |= (
                (tangent >= interval.minimum - tolerance)
                & (tangent < interval.maximum - tolerance)
            )
        selected[edge] |= interval_selected
    return selected


def _fluid_connected(geometry) -> bool:
    fluid = np.asarray(geometry.lambda_field) < 0.5
    inlet = fluid & _boundary_cells(geometry, geometry.spec.inlet)
    outlet = fluid & _boundary_cells(geometry, geometry.spec.outlet)
    if not inlet.any() or not outlet.any():
        return False
    visited = inlet.copy()
    queue = deque(map(tuple, np.argwhere(inlet)))
    ny, nx = fluid.shape
    while queue:
        row, column = queue.popleft()
        for next_row, next_column in (
            (row - 1, column), (row + 1, column),
            (row, column - 1), (row, column + 1),
        ):
            if (
                0 <= next_row < ny and 0 <= next_column < nx
                and fluid[next_row, next_column]
                and not visited[next_row, next_column]
            ):
                visited[next_row, next_column] = True
                queue.append((next_row, next_column))
    return bool(np.any(visited & outlet))


def validate_physical_domain(
    geometry,
    context,
    mapper,
    allow_anisotropic_mesh: bool = False,
) -> dict:
    """Return a JSON-safe gate report, or raise if any physical gate fails."""
    if type(allow_anisotropic_mesh) is not bool:
        raise ValueError("allow_anisotropic_mesh must be a boolean")
    if getattr(geometry, "spec", None) is None:
        raise ValueError("physical domain validation requires boundary metadata")

    epsilon = mapper.interface_tolerance
    outlet_nodes = np.asarray(context.pressure_outlet_nodes, dtype=np.int64)
    outlet_conflicts = int(mapper.outlet_geometry_conflicts)
    inlet_nodes = int(len(context.inlet_velocity_nodes))
    outlet_node_count = int(len(outlet_nodes))
    finite = {
        "lambda_finite": bool(np.all(np.isfinite(geometry.lambda_field))),
        "sigma_finite": bool(np.all(np.isfinite(geometry.signed_distance))),
        "normals_finite": bool(np.all(np.isfinite(geometry.normals))),
    }
    fluid_connected = _fluid_connected(geometry) if all(finite.values()) else False
    # RectangleMesh cell count alone does not retain nx/ny; infer spacings from vertices.
    coordinates = context.mesh.coordinates.dat.data_ro
    x_values = np.unique(coordinates[:, 0])
    y_values = np.unique(coordinates[:, 1])
    hx = float(np.min(np.diff(x_values)))
    hy = float(np.min(np.diff(y_values)))
    aspect_ratio = hx / hy
    aspect_error = abs(aspect_ratio - 1.0)
    aspect_passed = aspect_error <= 0.1 or allow_anisotropic_mesh
    inlet_conflicts = int(mapper.inlet_geometry_conflicts)
    hard_gate_passed = bool(
        inlet_conflicts == 0
        and outlet_conflicts == 0
        and inlet_nodes > 0
        and outlet_node_count > 0
        and all(finite.values())
        and fluid_connected
        and aspect_passed
    )
    report = {
        "domain_classification": geometry.classification,
        "article_reproduction": bool(geometry.spec.article_reproduction),
        "patch_mapping_valid": True,
        "inlet_geometry_conflicts": inlet_conflicts,
        "outlet_geometry_conflicts": outlet_conflicts,
        "inlet_node_count": inlet_nodes,
        "outlet_node_count": outlet_node_count,
        **finite,
        "fluid_connected": fluid_connected,
        "hx": hx,
        "hy": hy,
        "aspect_ratio": aspect_ratio,
        "aspect_error": aspect_error,
        "allow_anisotropic_mesh": allow_anisotropic_mesh,
        "hard_gate_passed": hard_gate_passed,
    }
    if not hard_gate_passed:
        failures = [
            name for name, passed in (
                ("inlet geometry conflicts", inlet_conflicts == 0),
                ("outlet geometry conflicts", outlet_conflicts == 0),
                ("inlet nodes", inlet_nodes > 0),
                ("outlet nodes", outlet_node_count > 0),
                ("finite lambda/sigma/normals", all(finite.values())),
                ("inlet-to-outlet fluid connectivity", fluid_connected),
                ("mesh aspect", aspect_passed),
            ) if not passed
        ]
        raise RuntimeError("physical domain validation failed: " + ", ".join(failures))
    return report
