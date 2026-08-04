#!/usr/bin/env python3
"""
Eigenfunction diagnostics for the OpenFOAM-LibTorch LHO run.

From postProcessing/fvNeuralLHO/profiles.csv compute:
  - Gram matrix G_ij = <psi_i, psi_j>_M (volume-weighted inner product)
  - M-weighted L2 errors (sign-aligned) NN vs direct-FV, vs analytical
  - parity check <psi_n(x), psi_n(-x)>_M
  - eigenfunction figures: analytical vs direct FV vs neural FV

Usage:
  python3 scripts/analyze_eigenfunctions.py --case cases/lho_128
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
NSTATES = 6  # overridden by --states
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def load_profiles(case_dir, nstates):
    fpath = Path(case_dir) / "postProcessing" / "fvNeuralLHO" / "profiles.csv"
    data = {"x": [], "v": []}
    for n in range(nstates):
        data[f"nn{n}"] = []
        data[f"fv{n}"] = []
    with open(fpath) as f:
        for row in csv.DictReader(f):
            data["x"].append(float(row["x"]))
            data["v"].append(float(row["volume"]))
            for n in range(nstates):
                data[f"nn{n}"].append(float(row[f"psiNN_{n}"]))
                data[f"fv{n}"].append(float(row[f"psiDirectFV_{n}"]))
    return {k: np.asarray(v) for k, v in data.items()}


def m_norm2(psi, V):
    """||psi||_M^2 = sum_c V_c psi_c^2 (cell-volume-weighted)."""
    return float(np.dot(V, psi * psi))


def m_dot(a, b, V):
    return float(np.dot(V, a * b))


def l2diff(a, b, V):
    """M-weighted L2 error after aligning the arbitrary eigenfunction sign."""
    if m_dot(a, b, V) < 0:
        a = -a
    return float(np.sqrt(m_norm2(a - b, V)))


def exact_eigenfunction(n, x):
    """Physicists' Hermite-polynomial oscillator eigenfunctions.
    Normalized on the real line, then renormalized in the discrete
    M-norm for a fair comparison with the FV eigenfunctions."""
    H0 = np.ones_like(x)
    if n == 0:
        H = H0
    else:
        H1 = 2.0 * x
        if n == 1:
            H = H1
        else:
            H0_, H1_ = H0, H1
            for k in range(1, n):
                H0_, H1_ = H1_, 2.0 * x * H1_ - 2.0 * k * H0_
            H = H1_
    prefactor = (math.pi ** 0.25) * math.sqrt(2.0 ** n * math.factorial(n))
    return H * np.exp(-0.5 * x * x) / prefactor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global NSTATES
    NSTATES = args.states
    d = load_profiles(args.case, NSTATES)
    x, V = d["x"], d["v"]

    psiNN = [d[f"nn{n}"] for n in range(NSTATES)]
    psiFV = [d[f"fv{n}"] for n in range(NSTATES)]
    psiEx = []
    for n in range(NSTATES):
        e = exact_eigenfunction(n, x)
        e = e / math.sqrt(m_norm2(e, V))
        psiEx.append(e)

    print("\n================ NN Gram matrix  G_ij = <psi_i, psi_j>_M ================")
    G_nn = np.array([[m_dot(psiNN[i], psiNN[j], V) for j in range(NSTATES)]
                     for i in range(NSTATES)])
    print(np.array2string(G_nn, precision=4, suppress_small=True, max_line_width=140))
    print(f"  max |G_ii - 1|      = {np.abs(np.diag(G_nn) - 1.0).max():.3e}")
    off = G_nn - np.diag(np.diag(G_nn))
    print(f"  max |G_ij|, i != j   = {np.abs(off).max():.3e}")

    print("\n================ Direct-FV Gram matrix ================")
    G_fv = np.array([[m_dot(psiFV[i], psiFV[j], V) for j in range(NSTATES)]
                     for i in range(NSTATES)])
    print(f"  max |G_ii - 1|      = {np.abs(np.diag(G_fv) - 1.0).max():.3e}")
    off_fv = G_fv - np.diag(np.diag(G_fv))
    print(f"  max |G_ij|, i != j   = {np.abs(off_fv).max():.3e}")

    print("\n================ Parity  <psi_n(x), psi_n(-x)>_M ================")
    print(f"{'state':>5} {'parity':>7} {'value':>14}   expected")
    for n in range(NSTATES):
        par = m_dot(psiNN[n], psiNN[n][::-1], V)
        sgn, expect = ("even", 1.0) if n % 2 == 0 else ("odd", -1.0)
        print(f"{n:>5} {sgn:>7} {par:>14.6f}   {expect}")

    print("\n======== M-weighted L2 errors (sign-aligned) ========")
    print(f"{'state':>5} {'||NN-FV||_M':>14} {'||NN-Ex||_M':>14} {'||FV-Ex||_M':>14}")
    for n in range(NSTATES):
        e_fv = l2diff(psiNN[n], psiFV[n], V)
        e_ex = l2diff(psiNN[n], psiEx[n], V)
        e_fvex = l2diff(psiFV[n], psiEx[n], V)
        print(f"{n:>5} {e_fv:>14.3e} {e_ex:>14.3e} {e_fvex:>14.3e}")

    # ---------------- figures ----------------
    import math as _m
    ncol, nrow = 3, _m.ceil(NSTATES / 3)
    fig, axes = plt.subplots(nrow, 3, figsize=(15, 2.8 * nrow), sharex=True)
    axes_flat = axes.reshape(-1)
    for n in range(NSTATES):
        ax = axes_flat[n]
        ref = psiNN[n]
        fv = psiFV[n] if m_dot(psiFV[n], ref, V) >= 0 else -psiFV[n]
        ex = psiEx[n] if m_dot(psiEx[n], ref, V) >= 0 else -psiEx[n]
        ax.plot(x, ex, "k:", lw=1.6, label="analytical", alpha=0.9)
        ax.plot(x, fv, "g--", lw=1.4, label="direct FV", alpha=0.9)
        ax.plot(x, ref, "b-", lw=1.1, label="neural FV", alpha=0.75)
        ax.set_title(rf"$\psi_{{{n}}}(x)$,  $E_{n}$ exact = {n + 0.5}")
        ax.legend(fontsize=8) if n == 0 else None
        ax.grid(alpha=0.3)
    for ax in axes_flat[max(NSTATES-3, 0):NSTATES]:
        ax.set_xlabel("x")
    for i in range(0, NSTATES, 3):
        axes_flat[i].set_ylabel(r"$\psi_n(x)$")
    fig.suptitle(f"Neural finite-volume eigenfunctions ({Path(args.case).name})")
    fig.tight_layout()
    png = out_dir / "eigenfunctions_n0_n5.png"
    fig.savefig(png, dpi=150)
    print(f"\nSaved figure: {png}")

    # Gram-matrix heatmap
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
    im = ax2.imshow(np.abs(G_nn - np.eye(NSTATES)), cmap="viridis", norm="log")
    ax2.set_title(r"NN orthogonality defect $|G_{ij}-\delta_{ij}|$")
    ax2.set_xlabel("j"); ax2.set_ylabel("i")
    fig2.colorbar(im, ax=ax2)
    for i in range(NSTATES):
        for j in range(NSTATES):
            ax2.text(j, i, f"{abs(G_nn[i, j] - (1 if i == j else 0)):.0e}",
                     ha="center", va="center", fontsize=7, color="w")
    png2 = out_dir / "orthogonality_matrix.png"
    fig2.tight_layout(); fig2.savefig(png2, dpi=150)
    print(f"Saved figure: {png2}")


if __name__ == "__main__":
    main()
