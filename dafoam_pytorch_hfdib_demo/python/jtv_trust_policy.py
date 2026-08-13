"""JTV trust policy: encodes WHERE the reverse-JTV gradient was validated.

Generated from outputs/warm_start_sweep/summary.json (the warm-start sweep).
Case-specific — do NOT reuse across cases or topologies.

During training the policy:
  * tracks the last accepted state;
  * evaluates the true forward residual loss after every proposed update;
  * rejects non-finite updates;
  * rejects updates whose residual norm leaves
    [0, max_residual_growth_factor x entry_residual_l2];
  * on rejection, restores the parameter state and optimizer snapshot, and
    halves the learning rate.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass


@dataclass
class JTVTrustPolicy:
    warm_iterations: int
    entry_residual_l2: float
    max_validated_gradient_error: float
    max_residual_growth_factor: float = 1.25

    def allows(self, residual_l2: float) -> bool:
        return (math.isfinite(residual_l2)
                and residual_l2 <= self.entry_residual_l2
                * self.max_residual_growth_factor)


def from_sweep(sweep_json_path: str) -> JTVTrustPolicy:
    """Pick the smallest eligible k from the sweep summary."""
    with open(sweep_json_path) as f:
        rows = json.load(f)["rows"]
    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        raise RuntimeError(
            "no eligible k in sweep summary — run warm_start_jtv_sweep first "
            "or see references/FAR_FIELD_JTV_MISMATCH.md")
    best = eligible[0]  # rows are ordered by k ascending
    return JTVTrustPolicy(
        warm_iterations=best["k"],
        entry_residual_l2=best["residual_l2"],
        max_validated_gradient_error=max(
            best["global_max_jtv_error"],
            best["torch_max_gradient_error"],
            best["loss_direction_max_error"],
        ),
    )
