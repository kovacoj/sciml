# Article setup reconstruction

Source: Ledl, Kubíčková, Isoz, "Using U-Net to Estimate Fluid Flow in
Mechanical Metamaterials", TPFM 2026 (DOI:10.14311/TPFM.2026.019), plus the
published `tpfm_unet` dataset artifacts (see UPSTREAM_COMMITS.md).

Recovered facts:

* Area of interest: 64×64 cells, 0.128 m × 0.128 m, cubic cells h = 0.002 m
  (cell volume 8e-9; z thickness 0.002 m, one cell deep, empty patches).
* Ports: two inlets (left) and two outlets (right) at bitmap rows 1 and 6
  (dataset edge statistics) → y ∈ [0.096, 0.112] and [0.016, 0.032],
  symmetric about y = 0.064. Port width 0.016 m.
* Boundary conditions: inlet U = (0.1, 0, 0) m/s; outlet p̃ = 0;
  walls no-slip; ν = 1e-2 m²/s. Steady, low-Re regime.
* Topology: 8×8 bitmap (each bitmap cell = 8×8 flow cells). First demo
  topology: dataset sample index 1 (single connected component, porosity
  0.531) — see `../../openfoam_libtorch_hfdib_demo/cases/four_port_64/constant/`
  for the identical bitmap used by both projects.
* λ convention (paper/openHFDIB, NOT the tpfm_unet README): λ = 0 fluid,
  λ = 1 solid, 0 < λ < 1 interface; interface cells smear with
  λ = 0.5 (1 − tanh(σ / V_P^(1/3))), σ > 0 in fluid.

NOT recovered (do not claim otherwise): the authors' full OpenFOAM case
files, port-channel extension geometry (reconstructed), solver settings, RNG
seed, exact λ smearing constant (their dataset shows two intermediate pairs
{0.27016124/0.72983876, 0.33120811/0.66879189}; the tanh/σ formulation used
here is the brief's verbatim formula from the paper summary, not bit-matched
to the authors' implementation).
