"""Analytic channel-network topologies for variational Brinkman benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domain import Bounds, Interval, Patch


MANUFACTURED_BRINKMAN = "MANUFACTURED_BRINKMAN"
DOMAIN_SIZE = 0.128
SPACING = 0.002
PORT_INTERVALS = ((0.020, 0.028), (0.100, 0.108))


@dataclass(frozen=True)
class ChannelSegment:
    a: tuple[float, float]
    b: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class CircularChamber:
    center: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class BrinkmanTopology:
    name: str
    segments: tuple[ChannelSegment, ...]
    chambers: tuple[CircularChamber, ...] = ()


class BrinkmanMetadata:
    classification = MANUFACTURED_BRINKMAN
    article_reproduction = False
    uin = 0.1
    pout = 0.0
    nu = 0.01

    def __init__(self) -> None:
        ports = tuple(Interval(*interval) for interval in PORT_INTERVALS)
        side_walls = (
            Interval(0.0, PORT_INTERVALS[0][0]),
            Interval(PORT_INTERVALS[0][1], PORT_INTERVALS[1][0]),
            Interval(PORT_INTERVALS[1][1], DOMAIN_SIZE),
        )
        full = (Interval(0.0, DOMAIN_SIZE),)
        self.inlet = (Patch("left", ports, marker=101, name="two_inlet_ports"),)
        self.outlet = (Patch("right", ports, marker=102, name="two_outlet_ports"),)
        self.wall = (
            Patch("left", side_walls, marker=103, name="left_walls"),
            Patch("right", side_walls, marker=104, name="right_walls"),
            Patch("bottom", full, marker=105, name="bottom_wall"),
            Patch("top", full, marker=106, name="top_wall"),
        )


def topology(name: str) -> BrinkmanTopology:
    lower, upper = 0.024, 0.104
    radius = 0.005
    if name == "A":
        return BrinkmanTopology("A_parallel_bridge", (
            ChannelSegment((0.0, lower), (DOMAIN_SIZE, lower), radius),
            ChannelSegment((0.0, upper), (DOMAIN_SIZE, upper), radius),
            ChannelSegment((0.064, lower), (0.064, upper), radius),
        ))
    if name == "B":
        return BrinkmanTopology("B_shifted_junction", (
            ChannelSegment((0.0, lower), (0.054, 0.050), radius),
            ChannelSegment((0.054, 0.050), (0.078, 0.078), radius),
            ChannelSegment((0.078, 0.078), (DOMAIN_SIZE, upper), radius),
            ChannelSegment((0.0, upper), (0.054, 0.078), radius),
            ChannelSegment((0.054, 0.078), (0.078, 0.050), radius),
            ChannelSegment((0.078, 0.050), (DOMAIN_SIZE, lower), radius),
            ChannelSegment((0.054, 0.050), (0.054, 0.078), radius),
        ))
    if name == "C":
        return BrinkmanTopology("C_merge_split", (
            ChannelSegment((0.0, lower), (0.052, 0.064), radius),
            ChannelSegment((0.0, upper), (0.052, 0.064), radius),
            ChannelSegment((0.076, 0.064), (DOMAIN_SIZE, lower), radius),
            ChannelSegment((0.076, 0.064), (DOMAIN_SIZE, upper), radius),
        ), (CircularChamber((0.064, 0.064), 0.020),))
    raise ValueError(f"unknown Brinkman topology: {name}")


def _segment_distance(points: np.ndarray, segment: ChannelSegment) -> np.ndarray:
    a = np.asarray(segment.a)
    b = np.asarray(segment.b)
    direction = b - a
    parameter = np.sum((points - a) * direction, axis=-1) / np.dot(direction, direction)
    projection = a + np.clip(parameter, 0.0, 1.0)[..., None] * direction
    return np.linalg.norm(points - projection, axis=-1) - segment.radius


def signed_distance(points: np.ndarray, value: BrinkmanTopology) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    distances = [_segment_distance(points, segment) for segment in value.segments]
    distances.extend(
        np.linalg.norm(points - np.asarray(chamber.center), axis=-1) - chamber.radius
        for chamber in value.chambers
    )
    return np.minimum.reduce(distances)


def lambda_values(points: np.ndarray, value: BrinkmanTopology, epsilon: float = SPACING) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(signed_distance(points, value) / epsilon))


class BrinkmanTopologyGeometry:
    """Analytic topology geometry on the fixed 0.128 m square domain."""

    def __init__(self, name: str, spacing: float = SPACING) -> None:
        self.topology = topology(name)
        self.spacing = float(spacing)
        self.xmin = self.ymin = 0.0
        self.xmax = self.ymax = DOMAIN_SIZE
        self.nx = self.ny = int(round(DOMAIN_SIZE / self.spacing))
        self.x = (np.arange(self.nx) + 0.5) * self.spacing
        self.y = (np.arange(self.ny) + 0.5) * self.spacing
        self.classification = MANUFACTURED_BRINKMAN
        self.spec = BrinkmanMetadata()
        self.roi_bounds = Bounds(0.0, 0.0, DOMAIN_SIZE, DOMAIN_SIZE)
        xx, yy = np.meshgrid(self.x, self.y)
        points = np.stack((xx, yy), axis=-1)
        self.signed_distance = signed_distance(points, self.topology)
        self.lambda_field = lambda_values(points, self.topology, self.spacing)
        self.interface = (self.lambda_field > 1.0e-10) & (self.lambda_field < 1.0 - 1.0e-10)
        grad_y, grad_x = np.gradient(self.signed_distance, self.spacing, self.spacing)
        magnitude = np.hypot(grad_x, grad_y)
        self.normals = np.stack((
            np.divide(grad_x, magnitude, out=np.zeros_like(grad_x), where=magnitude > 0),
            np.divide(grad_y, magnitude, out=np.zeros_like(grad_y), where=magnitude > 0),
        ), axis=-1)

    def interpolate(self, coordinates: np.ndarray, field: str = "signed_distance") -> np.ndarray:
        points = np.asarray(coordinates, dtype=np.float64)
        distance = signed_distance(points, self.topology)
        if field == "signed_distance":
            return distance
        if field == "lambda":
            return 0.5 * (1.0 + np.tanh(distance / self.spacing))
        if field == "normals":
            delta = self.spacing * 1.0e-4
            dx = signed_distance(points + (delta, 0.0), self.topology) - signed_distance(points - (delta, 0.0), self.topology)
            dy = signed_distance(points + (0.0, delta), self.topology) - signed_distance(points - (0.0, delta), self.topology)
            magnitude = np.hypot(dx, dy)
            return np.stack((
                np.divide(dx, magnitude, out=np.zeros_like(dx), where=magnitude > 0),
                np.divide(dy, magnitude, out=np.zeros_like(dy), where=magnitude > 0),
            ), axis=-1)
        raise ValueError(f"unknown geometry field: {field}")
