"""Strict machine-readable contracts for TPFM benchmark domains."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


RECONSTRUCTED_TPFM_DOMAIN = "RECONSTRUCTED_TPFM_DOMAIN"
CONTROLLED_TPFM_DERIVED_DOMAIN = "CONTROLLED_TPFM_DERIVED_DOMAIN"
MANUFACTURED_EMPTY_CHANNEL = "MANUFACTURED_EMPTY_CHANNEL"
MANUFACTURED_CIRCULAR_HFDIB = "MANUFACTURED_CIRCULAR_HFDIB"
_CLASSIFICATIONS = {RECONSTRUCTED_TPFM_DOMAIN, CONTROLLED_TPFM_DERIVED_DOMAIN}
_MANUFACTURED_CLASSIFICATIONS = {
    MANUFACTURED_EMPTY_CHANNEL, MANUFACTURED_CIRCULAR_HFDIB,
}
_BENCHMARK_MODES = {"article", "controlled", "manufactured"}
_SIDES = {"left", "right", "bottom", "top"}


def _object(value: Any, location: str, keys: set[str], optional: set[str] = set()) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    missing = keys - value.keys()
    unknown = value.keys() - keys - optional
    if missing:
        raise ValueError(f"{location} missing required keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{location} has unknown keys: {sorted(unknown)}")
    return value


def _integer(value: Any, location: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be a finite number")
    return result


@dataclass(frozen=True)
class Bounds:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @classmethod
    def parse(cls, value: Any, location: str) -> "Bounds":
        data = _object(value, location, {"xmin", "ymin", "xmax", "ymax"})
        bounds = cls(**{key: _number(data[key], f"{location}.{key}") for key in data})
        if bounds.xmax <= bounds.xmin or bounds.ymax <= bounds.ymin:
            raise ValueError(f"{location} must have xmin < xmax and ymin < ymax")
        return bounds


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    bounds: Bounds

    @classmethod
    def parse(cls, value: Any, location: str, dx: float) -> "Grid":
        data = _object(value, location, {"nx", "ny", "bounds"})
        grid = cls(
            nx=_integer(data["nx"], f"{location}.nx", minimum=1),
            ny=_integer(data["ny"], f"{location}.ny", minimum=1),
            bounds=Bounds.parse(data["bounds"], f"{location}.bounds"),
        )
        if not math.isclose(grid.bounds.xmax - grid.bounds.xmin, grid.nx * dx):
            raise ValueError(f"{location} x bounds are inconsistent with nx * dx")
        if not math.isclose(grid.bounds.ymax - grid.bounds.ymin, grid.ny * dx):
            raise ValueError(f"{location} y bounds are inconsistent with ny * dx")
        return grid


@dataclass(frozen=True)
class Interval:
    minimum: float
    maximum: float

    @classmethod
    def parse(cls, value: Any, location: str) -> "Interval":
        data = _object(value, location, {"min", "max"})
        interval = cls(
            minimum=_number(data["min"], f"{location}.min"),
            maximum=_number(data["max"], f"{location}.max"),
        )
        if interval.maximum <= interval.minimum:
            raise ValueError(f"{location} must have min < max")
        return interval


@dataclass(frozen=True)
class Patch:
    side: str
    intervals: tuple[Interval, ...]
    marker: int | None = None
    name: str | None = None

    @classmethod
    def parse(
        cls, value: Any, location: str, bounds: Bounds, tolerance: float
    ) -> "Patch":
        data = _object(value, location, {"side", "intervals"}, {"marker", "name"})
        side = data["side"]
        if side not in _SIDES:
            raise ValueError(f"{location}.side must be one of {sorted(_SIDES)}")
        raw_intervals = data["intervals"]
        if not isinstance(raw_intervals, list) or not raw_intervals:
            raise ValueError(f"{location}.intervals must be a non-empty array")
        intervals = tuple(
            Interval.parse(item, f"{location}.intervals[{index}]")
            for index, item in enumerate(raw_intervals)
        )
        if any(
            current.minimum < previous.minimum
            for previous, current in zip(intervals, intervals[1:])
        ):
            raise ValueError(f"{location}.intervals must be ordered by min")
        lower, upper = (
            (bounds.ymin, bounds.ymax) if side in {"left", "right"}
            else (bounds.xmin, bounds.xmax)
        )
        for interval in intervals:
            if (
                interval.minimum < lower - tolerance
                or interval.maximum > upper + tolerance
            ):
                raise ValueError(
                    f"{location} interval [{interval.minimum}, {interval.maximum}] "
                    f"is outside the {side} side range [{lower}, {upper}]"
                )
        _reject_overlaps(intervals, location)
        name = data.get("name")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError(f"{location}.name must be a non-empty string when present")
        marker = (
            _integer(data["marker"], f"{location}.marker", minimum=1)
            if data.get("marker") is not None else None
        )
        if marker is None and name is None:
            raise ValueError(f"{location} requires a marker or name")
        return cls(
            side=side,
            intervals=intervals,
            marker=marker,
            name=name,
        )


def _reject_overlaps(intervals: tuple[Interval, ...] | list[Interval], location: str) -> None:
    ordered = sorted(intervals, key=lambda interval: interval.minimum)
    for previous, current in zip(ordered, ordered[1:]):
        if current.minimum < previous.maximum:
            raise ValueError(f"{location} contains overlapping intervals")


@dataclass(frozen=True)
class DomainSpec:
    classification: str
    dx: float
    roi: Grid
    full_domain: Grid
    left_extension_cells: int
    right_extension_cells: int
    inlet: tuple[Patch, ...]
    outlet: tuple[Patch, ...]
    wall: tuple[Patch, ...]
    uin: float
    pout: float
    nu: float
    roi_cell_indices: tuple[tuple[int, ...], ...]
    provenance: dict[str, Any] | None = None

    @property
    def article_reproduction(self) -> bool:
        return self.classification == RECONSTRUCTED_TPFM_DOMAIN

    @classmethod
    def parse(cls, value: Any) -> "DomainSpec":
        keys = {
            "classification", "dx", "roi", "full_domain",
            "left_extension_cells", "right_extension_cells", "patches",
            "uin", "pout", "nu", "roi_cell_indices",
        }
        data = _object(value, "domain spec", keys, {"provenance"})
        classification = data["classification"]
        if classification not in _CLASSIFICATIONS:
            raise ValueError(
                f"classification must be one of {sorted(_CLASSIFICATIONS)}"
            )
        provenance = _parse_provenance(data.get("provenance"), classification)
        dx = _number(data["dx"], "dx")
        if dx <= 0.0:
            raise ValueError("dx must be > 0")
        tolerance = max(16.0 * math.ulp(dx), dx * 1.0e-10)
        roi = Grid.parse(data["roi"], "roi", dx)
        full = Grid.parse(data["full_domain"], "full_domain", dx)
        left = _integer(data["left_extension_cells"], "left_extension_cells", minimum=0)
        right = _integer(data["right_extension_cells"], "right_extension_cells", minimum=0)
        if full.ny != roi.ny or full.bounds.ymin != roi.bounds.ymin or full.bounds.ymax != roi.bounds.ymax:
            raise ValueError("rectangular layout requires matching ROI/full ny and y bounds")
        if full.nx != left + roi.nx + right:
            raise ValueError("full_domain.nx must equal left extension + roi.nx + right extension")
        expected_roi_xmin = full.bounds.xmin + left * dx
        expected_roi_xmax = full.bounds.xmax - right * dx
        if not math.isclose(roi.bounds.xmin, expected_roi_xmin) or not math.isclose(
            roi.bounds.xmax, expected_roi_xmax
        ):
            raise ValueError("ROI x bounds are inconsistent with the declared extensions")

        patch_data = _object(data["patches"], "patches", {"inlet", "outlet", "wall"})
        collections: dict[str, tuple[Patch, ...]] = {}
        for kind in ("inlet", "outlet", "wall"):
            raw = patch_data[kind]
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"patches.{kind} must be a non-empty array")
            collections[kind] = tuple(
                Patch.parse(
                    item,
                    f"patches.{kind}[{index}]",
                    full.bounds,
                    tolerance,
                )
                for index, item in enumerate(raw)
            )
        by_side: dict[str, list[Interval]] = {side: [] for side in _SIDES}
        for patches in collections.values():
            for patch in patches:
                by_side[patch.side].extend(patch.intervals)
        for side, intervals in by_side.items():
            _reject_overlaps(intervals, f"patches on {side} side")
            lower, upper = (
                (full.bounds.ymin, full.bounds.ymax)
                if side in {"left", "right"}
                else (full.bounds.xmin, full.bounds.xmax)
            )
            if not intervals:
                raise ValueError(f"patches must represent the full {side} side")
            ordered = sorted(intervals, key=lambda interval: interval.minimum)
            if not math.isclose(
                ordered[0].minimum, lower, rel_tol=0.0, abs_tol=tolerance
            ):
                raise ValueError(f"patches on {side} side have a leading gap")
            for previous, current in zip(ordered, ordered[1:]):
                if not math.isclose(
                    previous.maximum,
                    current.minimum,
                    rel_tol=0.0,
                    abs_tol=tolerance,
                ):
                    raise ValueError(f"patches on {side} side have an interior gap")
            if not math.isclose(
                ordered[-1].maximum, upper, rel_tol=0.0, abs_tol=tolerance
            ):
                raise ValueError(f"patches on {side} side have a trailing gap")

        mapping = _parse_mapping(data["roi_cell_indices"], roi, full)
        nu = _number(data["nu"], "nu")
        if nu <= 0.0:
            raise ValueError("nu must be > 0")
        return cls(
            classification=classification,
            dx=dx,
            roi=roi,
            full_domain=full,
            left_extension_cells=left,
            right_extension_cells=right,
            inlet=collections["inlet"],
            outlet=collections["outlet"],
            wall=collections["wall"],
            uin=_number(data["uin"], "uin"),
            pout=_number(data["pout"], "pout"),
            nu=nu,
            roi_cell_indices=mapping,
            provenance=provenance,
        )


def _parse_mapping(value: Any, roi: Grid, full: Grid) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or len(value) != roi.ny:
        raise ValueError(f"roi_cell_indices must have shape ({roi.ny}, {roi.nx})")
    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != roi.nx:
            raise ValueError(f"roi_cell_indices must have shape ({roi.ny}, {roi.nx})")
        rows.append(tuple(
            _integer(item, f"roi_cell_indices[{row_index}][{column}]", minimum=0)
            for column, item in enumerate(row)
        ))
    flat = [item for row in rows for item in row]
    if len(set(flat)) != len(flat):
        raise ValueError("roi_cell_indices must be unique")
    if any(item >= full.nx * full.ny for item in flat):
        raise ValueError("roi_cell_indices contains an out-of-range full-domain index")
    return tuple(rows)


def _parse_provenance(value: Any, classification: str) -> dict[str, Any] | None:
    if classification == RECONSTRUCTED_TPFM_DOMAIN:
        if value is None:
            return None
        return dict(_object(value, "provenance", set(value) if isinstance(value, dict) else set()))

    keys = {
        "source", "source_classification", "source_branch", "source_gate_passed",
        "extension_selection", "article_reproduction",
    }
    data = _object(value, "provenance", keys)
    for key in ("source", "extension_selection"):
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError(f"provenance.{key} must be a non-empty string")
    expected = {
        "source_classification": "TPFM_TOPOLOGIES_ONLY",
        "source_branch": "feat/dafoam-tpfm-reproduction",
        "source_gate_passed": False,
        "article_reproduction": False,
    }
    for key, expected_value in expected.items():
        if type(data[key]) is not type(expected_value) or data[key] != expected_value:
            raise ValueError(f"provenance.{key} must be {expected_value!r}")
    return dict(data)


def validate_benchmark_mode(
    benchmark_mode: Any, domain_spec: DomainSpec | None
) -> str:
    """Validate an explicit benchmark claim without changing its classification."""
    if domain_spec is None:
        if benchmark_mode is not None:
            raise ValueError("benchmark_mode requires domain_spec; cropped geometry is diagnostic")
        return "diagnostic"
    if benchmark_mode is None:
        raise ValueError("benchmark_mode is required whenever domain_spec is present")
    if not isinstance(benchmark_mode, str) or benchmark_mode not in _BENCHMARK_MODES:
        raise ValueError(f"benchmark_mode must be one of {sorted(_BENCHMARK_MODES)}")
    if domain_spec.classification in _MANUFACTURED_CLASSIFICATIONS:
        expected = "manufactured"
    else:
        expected = (
            "article"
            if domain_spec.classification == RECONSTRUCTED_TPFM_DOMAIN
            else "controlled"
        )
    if benchmark_mode != expected:
        raise ValueError(
            f"benchmark_mode {benchmark_mode!r} does not match domain classification "
            f"{domain_spec.classification!r}; expected {expected!r}"
        )
    return benchmark_mode


def load_domain_spec(path: str | Path) -> DomainSpec:
    """Load and strictly validate a DAFoam-produced domain JSON file."""
    source = Path(path)
    try:
        data = json.loads(
            source.read_text(),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value!r} is not allowed")
            ),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load domain spec {source}: {error}") from error
    try:
        return DomainSpec.parse(data)
    except ValueError as error:
        raise ValueError(f"invalid domain spec {source}: {error}") from error
