"""Exact DAFoam state layout for the ConvergentChannel/DASimpleFoam baseline.

Provenance (verified in-container, 2026-08-04, recorded to
outputs/state_layout/state_layout_summary.json):

  1. The OpenFOAM init log prints the registration:
        Adjoint States:
        4
        (
            U   volVectorState
            phi surfaceScalarState
            p   volScalarState
            T   volScalarState
        )
        Global Cells: 343
        Global Faces: 1176
        Global Adjoint States: 2891

  2. bridge.solver.getOption("adjStateOrdering") == "state".

  3. Component fingerprints on the actual initial state vector gave, in
     array order: a U block whose first 9 values are
     [10, 0, 0, 10, 0, 0, 10, 0, 0]  (cell-major (Ux,Uy,Uz) per cell),
     then a 343-entry all-zero block (p), then a 343-entry all-300 block
     (T), then 1176 block with values ~[-1.3, 1.1] (phi).

Resulting CERTIFIED layout (registered order [U, phi, p, T] does NOT
describe the memory layout; in memory the scalar states come before phi):

    [ U : 1029, cell-major (Ux,Uy,Uz) x 343 ]
    [ p : 343 ]
    [ T : 343 ]
    [ phi : 1176 ]            total = 2891

No nuTilda model state exists (dummy turbulence registers none); the
0/nuTilda field file is unrelated to the adjoint state.

NOTE on entry indices: the pybind11 surface does NOT expose DAIndex
(adjStateName4LocalAdjIdx etc.); boundaries were resolved by fingerprint
statistics and are enforced by tests/test_state_layout.py.  For a volVector
entry i: cell = i // 3, component = i % 3 (0,1,2 <-> x,y,z).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REGISTERED = [
    ("U", "volVectorState", 1029),
    ("p", "volScalarState", 343),
    ("T", "volScalarState", 343),
    ("phi", "surfaceScalarState", 1176),
]

# Thin-3-D isothermal channel (40x16x1 duct): no thermal state.
# Faces are MESH faces: 1224 internal + 1392 boundary (front/back walls are
# 640+640 full slab faces), Global Faces 2616 per the registration print.
REGISTERED_ISO = [
    ("U", "volVectorState", 1920),   # 3 * 640 cells, cell-major
    ("p", "volScalarState", 640),
    ("phi", "surfaceScalarState", 2616),
]


def layout_from_registration(registered) -> StateLayout:
    entries: list[StateEntry] = []
    by_name: dict[str, list[int]] = {}
    offset = 0
    for name, typ, size in registered:
        ids = []
        for j in range(size):
            idx = offset + j
            if name == "U":
                entries.append(StateEntry(idx, name, j // 3, j % 3, "cell"))
            elif typ == "surfaceScalarState":
                entries.append(StateEntry(idx, name, j, None, "face"))
            else:
                entries.append(StateEntry(idx, name, j, None, "cell"))
            ids.append(idx)
        by_name[name] = ids[:]
        offset += size
    return StateLayout(
        ordering="state",
        entries=entries,
        indices_by_name={k: np.asarray(v, dtype=np.int64) for k, v in by_name.items()},
    )


@dataclass(frozen=True)
class StateEntry:
    local_index: int
    state_name: str
    entity_index: int
    component: int | None   # 0,1,2 for vector entries; None otherwise
    entity_type: str        # "cell" or "face"


@dataclass
class StateLayout:
    ordering: str                       # "state"
    entries: list[StateEntry]
    indices_by_name: dict[str, np.ndarray]

    @property
    def size(self) -> int:
        return len(self.entries)

    def name_of(self, index: int) -> str:
        return self.entries[index].state_name

    def indices(self, name: str) -> np.ndarray:
        return self.indices_by_name[name]


def build_state_layout(kind: str = "thermal") -> StateLayout:
    registered = REGISTERED if kind == "thermal" else REGISTERED_ISO
    return layout_from_registration(registered)
