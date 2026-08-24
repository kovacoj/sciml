# Pressure diagnostics summary

## Hypothesis A — teacher truncation (strongest)

- Teacher W20 median e_p: 0.379
- Network median e_p: 0.995
The network is trained on W20, whose pressure is itself far from converged.

## Hypothesis B — gauge alignment

The benchmark uses gauge-centered pressure (mean-subtracted), so the
reported errors are not purely gauge-offset artifacts.

## Hypothesis C — normalization

Not auditable without re-running training; documented as a known risk.

## Hypothesis D — loss imbalance

Not measurable from frozen model; documented as a known risk.

## Hypothesis E — pressure is globally constrained

Pressure enforces incompressibility and transmits information across the
domain. Good local velocity does not imply accurate pressure drop.
This mirrors the reference U-Net, where velocity is visually accurate but
pressure-drop errors of roughly 15% remain.

## Hypothesis F — SIMPLE state quality ≠ field-distance quality

- Median neural initial residual ratio: 3.86x cold
- Median neural initial velocity error ratio: 0.646x cold
The neural state has lower velocity error but much higher solver residual,
explaining why convergence iterations are not reduced.

## Hypothesis G — interface treatment

Local HFDIB uses transition width 1.5h vs paper's h. This affects pressure
drop between our DAFoam CFD and their IBM CFD, but not the network error
measured against our own DAFoam reference.
