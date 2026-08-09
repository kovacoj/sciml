"""Low-level DAFoam residual bridge (torch-free).

One PYDAFOAM object is created per bridge and kept alive for the entire
training run. State exchange is NumPy-only: no files, no subprocess, and the
primal solver is never invoked for residual evaluation.
"""
from __future__ import annotations

import os

import numpy as np
from mpi4py import MPI

from dafoam import PYDAFOAM


class DAFoamResidualBridge:
    """Opaque W in R^N_state -> residual R(W) and (∂R/∂W)ᵀ·seed."""

    def __init__(self, case_dir: str, da_options: dict, comm=None):
        case_dir = os.path.abspath(case_dir)
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        # PYDAFOAM resolves the case from CWD (verified: no runCaseDir key).
        self._prev_cwd = os.getcwd()
        os.chdir(case_dir)
        self.case_dir = case_dir

        self.solver = PYDAFOAM(options=da_options, comm=comm)
        self.solver.solverAD.initializedRdWTMatrixFree()

        self.discipline = self.solver.getOption("discipline")
        self.state_name = f"{self.discipline}_states"
        self.residual_name = f"{self.discipline}_residuals"

        self._n = int(self.solver.getNLocalAdjointStates())

    @property
    def state_size(self) -> int:
        return self._n

    def initial_state(self) -> np.ndarray:
        return np.ascontiguousarray(
            self.solver.getStates().copy(), dtype=np.float64
        )

    def set_state(self, state: np.ndarray) -> None:
        state = np.ascontiguousarray(state, dtype=np.float64)
        if state.shape != (self._n,):
            raise ValueError(f"state shape {state.shape} != ({self._n},)")
        self.solver.setStates(state)

    def residual(self, state: np.ndarray) -> np.ndarray:
        state = np.ascontiguousarray(state, dtype=np.float64)
        if state.shape != (self._n,):
            raise ValueError(f"state shape {state.shape} != ({self._n},)")

        self.solver.setStates(state)
        # refresh AD residual-side intermediate quantities
        self.solver.solverAD.calcPrimalResidualStatistics("calc")
        out = self.solver.getResiduals()
        return np.ascontiguousarray(out.copy(), dtype=np.float64)

    def simple_step(self, state: np.ndarray) -> np.ndarray:
        """Execute one SIMPLE iteration from the given state.

        Sets the state, runs one momentum+pressure+flux correction
        iteration, and returns the updated state.
        """
        state = np.ascontiguousarray(state, dtype=np.float64)
        if state.shape != (self._n,):
            raise ValueError(f"state shape {state.shape} != ({self._n},)")

        self.solver.setStates(state)
        # Run one SIMPLE iteration on the normal (non-AD) solver
        # to avoid AD tape consistency issues
        self.solver.solver.solvePrimalOneStep()
        # Read back from the normal solver's fields
        w_next = np.zeros(self._n, dtype=np.float64)
        self.solver.solver.getOFFields(w_next)
        return np.ascontiguousarray(w_next.copy(), dtype=np.float64)

    def residual_jacobian_transpose_vector(
        self, state: np.ndarray, seed: np.ndarray
    ) -> np.ndarray:
        state = np.ascontiguousarray(state, dtype=np.float64)
        seed = np.ascontiguousarray(seed, dtype=np.float64)
        if state.shape != (self._n,):
            raise ValueError(f"state shape {state.shape} != ({self._n},)")
        if seed.shape != (self._n,):
            raise ValueError(f"seed shape {seed.shape} != ({self._n},)")

        self.solver.setStates(state)
        # same state-side refresh as the residual path (measured: skipping
        # this leaves the AD tape inconsistent with getResiduals at ~5e-5 rel)
        self.solver.solverAD.calcPrimalResidualStatistics("calc")
        product = np.zeros_like(state)
        self.solver.solverAD.calcJacTVecProduct(
            self.state_name,
            "stateVar",
            state,
            self.residual_name,
            "residual",
            seed,
            product,
        )
        return product
