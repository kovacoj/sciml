#!/usr/bin/env python3
"""2D eigenfunction diagnostics for the unit-disk Laplacian case.

Reads postProcessing/fvNeuralLHO/{profiles.csv,eigenvalues.csv,training_state_*.csv}
and produces:

  disk_eigenfunctions_s{A}-{B}.png  per-state fields: NN | FV (aligned) | difference
  disk_radial_m0.png                radial profiles of m=0 states vs exact Bessel J0
  disk_energies.png                 energy levels + error decomposition
  disk_orthogonality.png            Gram-matrix defect heatmap
  disk_convergence.png              |E(step) - E_direct| training curves

Degenerate doublets {1,2}, {3,4}, {6,7} are aligned by an orthogonal
(Procrustes) rotation inside the 2D FV subspace before differencing,
since any orthonormal basis of a degenerate eigenspace is equally valid.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.special import jv, jn_zeros
from scipy.linalg import svd

# zeroth-order unit-disk Dirichlet spectrum: state -> (m, k)
DISK_MODES = {0: (0, 1), 1: (1, 1), 2: (1, 1), 3: (2, 1), 4: (2, 1),
              5: (0, 2), 6: (3, 1), 7: (3, 1)}
DEGENERATE_GROUPS = [[0], [1, 2], [3, 4], [5], [6, 7]]


def load_case(case_dir):
    pp = f"{case_dir}/postProcessing/fvNeuralLHO"
    prof = pd.read_csv(f"{pp}/profiles.csv")
    eig = pd.read_csv(f"{pp}/eigenvalues.csv")
    return prof, eig, pp


def mgram(X, M):
    return X @ np.diag(M) @ X.T


def align_nn_to_fv(NN, FV, M):
    """Rotate NN states inside each degenerate group onto the FV basis and
    sign-align non-degenerate states."""
    aligned = NN.copy()
    for group in DEGENERATE_GROUPS:
        if len(group) == 1:
            n = group[0]
            if (M * NN[n] * FV[n]).sum() < 0:
                aligned[n] = -aligned[n]
        else:
            U, V = NN[group], FV[group]
            C = U @ np.diag(M) @ V.T          # M-inner-product cross matrix
            Wl, _, Wt = svd(C)
            W = Wl @ Wt                        # orthogonal Procrustes
            aligned[group] = W @ U
    return aligned


def plot_fields(prof, eig, NN, FV, outdir):
    x, y = prof["x"].values, prof["y"].values
    tri = Triangulation(x, y)
    states = sorted(DISK_MODES)
    for chunk_i in range(0, len(states), 4):
        chunk = states[chunk_i:chunk_i + 4]
        fig, axes = plt.subplots(len(chunk), 3, figsize=(13, 3.4 * len(chunk)),
                                 constrained_layout=True)
        for row, n in enumerate(chunk):
            m, k = DISK_MODES[n]
            fields = [NN[n], FV[n], NN[n] - FV[n]]
            titles = [
                rf"$\psi_{{NN}}$   $E={eig.E_NN[n]:.5f}$",
                rf"$\psi_{{FV}}$   $E_{{direct}}={eig.E_directFV[n]:.5f}$",
                rf"$\psi_{{NN}}-\psi_{{FV}}$   "
                rf"$\Vert\cdot\Vert_M={np.sqrt((prof.volume.values * fields[2]**2).sum()):.2e}$",
            ]
            vmax = np.abs(fields[:2]).max()
            for col, (ax, fld, ttl) in enumerate(zip(axes[row], fields, titles)):
                lim = vmax if col < 2 else np.abs(fld).max()
                im = ax.tricontourf(tri, fld, levels=41, cmap="RdBu_r",
                                    vmin=-lim, vmax=lim)
                ax.add_patch(plt.Circle((0, 0), 1, fill=False, lw=1, color="k"))
                ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                if col == 0:
                    ax.set_ylabel(f"state {n}:  "
                                  rf"$J_{m}(j_{{{m},{k}}} r)\,{'cos' if m > 0 else ''}"
                                  rf"({m}\theta)$" if m > 0 else
                                  f"state {n}:  $J_{m}(j_{{{m},{k}}} r)$",
                                  fontsize=10)
                ax.set_title(ttl, fontsize=10)
        fname = f"{outdir}/disk_eigenfunctions_s{chunk[0]}-{chunk[-1]}.png"
        fig.suptitle(
            rf"unit disk $-\Delta$ eigenstates   "
            rf"(exact: $j^2_{{{DISK_MODES[chunk[0]][0]},{''}}}\ldots$; "
            f"$E_{{exact}}$ in energy figure)",
            fontsize=11)
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print("wrote", fname)


def plot_radial(prof, NN, FV, outdir):
    x, y = prof["x"].values, prof["y"].values
    r = np.hypot(x, y)
    rr = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, n in zip(axes, [0, 5]):
        m, k = DISK_MODES[n]
        j = jn_zeros(m, k)[-1]
        exact = jv(0, j * rr)
        exact /= np.sqrt(np.pi) * abs(jv(1, j))     # M-normalised
        sign = np.sign(NN[n][np.argmin(r)] * exact[0]) or 1.0
        ax.plot(rr, sign * exact, "r-", lw=1.6, label=rf"exact $J_0(j_{{0,{k}}}r)$")
        ax.plot(r, FV[n], ".", ms=2, color="0.55", label="FV direct")
        ax.plot(r, NN[n], ".", ms=2, color="navy", alpha=0.5, label="neural")
        ax.set_xlabel("r"); ax.set_ylabel(rf"$\psi_{n}(r)$")
        ax.set_title(f"state {n}  (m=0, k={k})", fontsize=10)
        ax.legend(fontsize=8, markerscale=3)
    fname = f"{outdir}/disk_radial_m0.png"
    fig.suptitle("radially symmetric states: all angular samples collapse onto the Bessel profile")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print("wrote", fname)


def plot_energies(eig, outdir):
    n_ = len(eig)
    idx = np.arange(n_)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    a1.plot(idx, eig.E_exact, "D", color="k", ms=7, mfc="none", label="exact (Bessel)")
    a1.plot(idx - 0.12, eig.E_directFV, "s", ms=5, color="tab:blue", label="direct FV")
    a1.plot(idx + 0.12, eig.E_NN, "o", ms=5, mfc="none", color="tab:red", label="neural")
    a1.set_xticks(idx)
    a1.set_xticklabels([f"{i}\n$m$={DISK_MODES[i][0]}" for i in idx], fontsize=8)
    a1.set_ylabel("E"); a1.set_title("energy levels"); a1.legend()

    w = 0.38
    a2.bar(idx - w / 2, eig.dE_NN_direct, w, label=r"|E_NN $-$ E_direct|  (optimisation)")
    a2.bar(idx + w / 2, eig.dE_direct_exact, w, label=r"|E_direct $-$ E_exact|  (discretisation)")
    a2.set_yscale("log"); a2.set_xticks(idx); a2.set_xlabel("state")
    a2.set_title("error decomposition"); a2.legend()
    fname = f"{outdir}/disk_energies.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print("wrote", fname)


def plot_gram(prof, NN, FV, outdir):
    G = mgram(NN, prof["volume"].values)
    D = np.abs(G - np.eye(len(G)))
    fig, ax = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    im = ax.imshow(np.log10(D + 1e-16), cmap="viridis", vmin=-16, vmax=-8)
    ax.set_xticks(range(len(G))); ax.set_yticks(range(len(G)))
    ax.set_title(f"Gram defect $|\\langle\\psi_i,\\psi_j\\rangle_M - \\delta_{{ij}}|$, "
                 f"max = {D.max():.1e}")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$")
    fname = f"{outdir}/disk_orthogonality.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print("wrote", fname)


def plot_convergence(eig, pp, outdir):
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for n in range(len(eig)):
        try:
            t = pd.read_csv(f"{pp}/training_state_{n}.csv")
        except FileNotFoundError:
            continue
        ax.semilogy(t.step, np.abs(t.energy - eig.E_directFV[n]), lw=1, label=f"state {n}")
    ax.set_xlabel("optimiser step"); ax.set_ylabel(r"$|E(step) - E_{direct}|$")
    ax.set_title("training convergence (150 pretrain + 500 Adam + 20 LBFGS)")
    ax.legend(ncol=2, fontsize=8)
    fname = f"{outdir}/disk_convergence.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print("wrote", fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    prof, eig, pp = load_case(args.case)
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    n_states = len(eig)
    M = prof["volume"].values
    NN = np.stack([prof[f"psiNN_{n}"].values for n in range(n_states)])
    FV = np.stack([prof[f"psiDirectFV_{n}"].values for n in range(n_states)])

    NN_al = align_nn_to_fv(NN, FV, M)

    plot_fields(prof, eig, NN_al, FV, args.output_dir)
    plot_radial(prof, NN_al, FV, args.output_dir)
    plot_energies(eig, args.output_dir)
    plot_gram(prof, NN_al, FV, args.output_dir)
    plot_convergence(eig, pp, args.output_dir)


if __name__ == "__main__":
    main()
