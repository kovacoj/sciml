#!/usr/bin/env python3
"""Gate E1: warm-started free-state residual minimization.

    W0 --(k SIMPLE iterations)--> W_k --(PyTorch/JTV optimization)--> W_final

Parameterization (brief section 6):
    W = W_k + S_W . z,   z the trainable dimensionless parameter
    S_W per-certified-block scales (U 10, p 50, T 300, phi 1);
    phi entries are FROZEN (their JTV columns are the measured bad subspace)
    and phi residual rows are masked out of the loss (they are ~5e-5 of the
    residual energy and uncertified); U/p/T residual rows carry it.

Step rejection (JTV trust policy): after each optimizer step, the true
forward loss is recomputed; a step is rejected (state+optimizer restored,
lr halved) if the loss is non-finite or residual_l2 grows beyond
entry_residual_l2 x growth_factor.

Usage: train_free_state.py [--k 8] [--steps 1500] [--lr 1e-3]
       [--clip 0.0] [--no-mask-phi] [--seed 1234]
"""
from __future__ import annotations

import argparse
import json

import os
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import torch  # noqa: E402

from common import channel_baseline_options, write_json, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from residual_autograd import dafoam_residual_loss  # noqa: E402
from state_layout import build_state_layout  # noqa: E402

BLOCK_SCALES = {"U": 10.0, "p": 50.0, "T": 300.0, "phi": 1.0}
GROWTH_FACTOR = 1.25


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lbfgs-iters", type=int, default=0)
    ap.add_argument("--gn-iters", type=int, default=0,
                    help="Gauss-Newton outer iterations (CG on normal eqns)")
    ap.add_argument("--cg-iters", type=int, default=300)
    ap.add_argument("--pc-probes", type=int, default=8,
                    help="Hutchinson probes for the Jacobi preconditioner")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--no-mask-phi", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_baseline")
    bridge = DAFoamResidualBridge(case_dir, channel_baseline_options(case_dir))
    layout = build_state_layout()
    n = bridge.state_size

    w_k = np.load(os.path.join(PROJECT_ROOT, "outputs", "work",
                               f"partial-k{args.k}", "W_k.npy"))
    meta = json.load(open(os.path.join(PROJECT_ROOT, "outputs", "work",
                                       f"partial-k{args.k}",
                                       "partial_primal.json")))

    # dimensionless parameterization W = W_k + S_W . z ; phi rows frozen
    scale = np.ones(n)
    for name, ids in layout.indices_by_name.items():
        scale[ids] = BLOCK_SCALES[name]
    if not args.no_mask_phi:
        scale[layout.indices("phi")] = 0.0
    S = torch.from_numpy(scale)

    # residual row mask: drop phi rows from the loss unless requested full
    weights = torch.ones(n)
    if not args.no_mask_phi:
        weights[layout.indices("phi")] = 0.0

    entry_res = float(np.linalg.norm(bridge.residual(w_k)))
    ceiling = entry_res * GROWTH_FACTOR

    w_k_t = torch.from_numpy(w_k)
    z = torch.nn.Parameter(torch.zeros(n, dtype=torch.float64))
    opt = torch.optim.Adam([z], lr=args.lr)

    def forward_loss():
        return dafoam_residual_loss(w_k_t + S * z, bridge, weights)

    def raw_res_l2():
        r = bridge.residual((w_k_t + S * z).detach().numpy())
        return float(np.linalg.norm(r)), 0.5 * float(r @ r)

    t0 = time.perf_counter()
    hist = []
    rejected = 0
    accepted = 0
    last_loss = None
    for step in range(args.steps):
        opt.zero_grad()
        loss = forward_loss()
        if not torch.isfinite(loss):
            print(f"[e1] step {step}: non-finite loss: FAIL")
            return 1
        loss.backward()
        g_raw = z.grad.clone()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_([z], max_norm=args.clip)

        pre_loss = float(loss.item())
        param_snapshot = z.detach().clone()
        opt_snapshot = {k2: (v.clone() if isinstance(v, torch.Tensor) else v)
                        for k2, v in opt.state[z].items()}

        opt.step()
        # step rejection: evaluate TRUE forward loss at the proposal
        with torch.no_grad():
            post_r2, post_loss = raw_res_l2()
        ok = (post_r2 is not None and post_r2 == post_r2
              and post_r2 <= ceiling and post_loss <= pre_loss)
        if ok:
            accepted += 1
        else:
            rejected += 1
            with torch.no_grad():
                z.data.copy_(param_snapshot)
            opt.state[z].clear()
            opt.state[z].update(opt_snapshot)
            for group in opt.param_groups:
                # halve on rejection but keep a floor so Adam stays alive
                group["lr"] = max(group["lr"] * 0.5, 1e-5)
        last_loss = post_loss if ok else pre_loss

        if step % 25 == 0 or step == args.steps - 1 or not ok:
            with torch.no_grad():
                g = z.grad
                hist.append({
                    "step": step,
                    "pre_step_loss": pre_loss,
                    "post_step_loss": post_loss,
                    "residual_l2": post_r2,
                    "raw_gradient_l2": float(g_raw.norm().item()),
                    "raw_gradient_linf": float(g_raw.abs().max().item()),
                    "accepted": bool(ok),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "state_l2": float((w_k_t + S * z).norm().item()),
                    "z_l2": float(z.norm().item()),
                    "elapsed_seconds": round(time.perf_counter() - t0, 2),
                })
                print(f"[e1] step {step:4d} pre={pre_loss:.6e} "
                      f"post={post_loss:.6e} R={post_r2:.3e} "
                      f"acc={int(ok)} lr={opt.param_groups[0]['lr']:.2g} "
                      f"rej={rejected}", flush=True)

    # --- Phase 2: Gauss-Newton via CG on the normal equations --------------
    # Solves JᵀJ Δ = -JᵀR by CG using only matrix-free products:
    #   J v  ~ ( R(W + h v) - R(W - h v) ) / (2 h)   (central FD, scaled)
    #   Jᵀw  = reverse JTV
    # Each outer GN step does a true-loss halving line search (trust policy
    # ceiling applies as everywhere else).
    if args.gn_iters > 0:
        w_np = lambda: (w_k_t + S * z).detach().numpy().copy()
        for gn in range(args.gn_iters):
            W = w_np()
            R = bridge.residual(W)
            L_now = 0.5 * float(R @ R)
            rnorm = float(np.linalg.norm(R))
            g = bridge.residual_jacobian_transpose_vector(W, R)
            gnorm = float(np.linalg.norm(g))
            if gnorm == 0.0:
                print(f"[e1] gn {gn}: zero gradient, stop"); break
            # CG
            cg_calls = 0

            def Jv(v):
                # J v via central FD: normalized direction, 1e-5 raw step
                vn = float(np.linalg.norm(v)) or 1.0
                vh = v / vn
                hh = 1e-5
                hp = bridge.residual(W + hh * vh)
                hm = bridge.residual(W - hh * vh)
                return vn * (hp - hm) / (2 * hh)

            def Hx(x):
                nonlocal cg_calls
                cg_calls += 1
                return bridge.residual_jacobian_transpose_vector(W, Jv(x))

            # Jacobi (diagonal) preconditioner from Hutchinson probes of H
            rng_pc = np.random.default_rng(2024 + gn)
            diag_acc = np.zeros(w_np().size)
            for _probe in range(args.pc_probes):
                xi = rng_pc.choice([-1.0, 1.0], size=diag_acc.size)
                diag_acc += Hx(xi) * xi
            diag = np.abs(diag_acc) / args.pc_probes
            floor = max(gnorm ** 2 * 1e-14, 1e-30)
            diag = np.maximum(diag, floor)

            # preconditioned CG on H x = -g :  z = M^-1 r
            x = np.zeros_like(g)
            r_cg = -g.copy()
            z_cg = r_cg / diag
            p_cg = z_cg.copy()
            rz = float(r_cg @ z_cg)
            for cg in range(args.cg_iters):
                Hp = Hx(p_cg)
                pHp = float(p_cg @ Hp)
                if abs(pHp) < 1e-30:
                    break
                alpha = rz / pHp
                x = x + alpha * p_cg
                r_cg = r_cg - alpha * Hp
                z_cg = r_cg / diag
                rz_new = float(r_cg @ z_cg)
                if rz_new < 1e-16 * rz + 1e-30:
                    rz = rz_new
                    break
                beta = rz_new / rz
                p_cg = z_cg + beta * p_cg
                rz = rz_new

            # true-loss line search (halving), trust ceiling enforced
            alpha = 1.0
            accepted_gn = False
            for ls in range(8):
                cand = W + alpha * x
                r2c = float(np.linalg.norm(bridge.residual(cand)))
                Lc = 0.5 * r2c ** 2
                if r2c <= ceiling and Lc < L_now:
                    accepted_gn = True
                    break
                alpha *= 0.5
            if accepted_gn:
                with torch.no_grad():
                    z.data.copy_(torch.from_numpy((cand - w_k) / scale))
                print(f"[e1] gn {gn:2d} L={L_now:.6e}->{Lc:.6e} "
                      f"R={rnorm:.3e}->{r2c:.3e} alpha={alpha:.3g} "
                      f"cg_calls={cg_calls}", flush=True)
            else:
                print(f"[e1] gn {gn:2d} line search failed at L={L_now:.6e} "
                      f"(|g|={gnorm:.2e}); stop GN", flush=True)
                break

    # --- Phase 3: LBFGS polish (optional) ---------------------------------
    if args.lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(
            [z], max_iter=20, max_eval=25,
            tolerance_grad=1e-12, tolerance_change=1e-14,
            history_size=20, line_search_fn="strong_wolfe",
            lr=1.0)
        closure_calls = 0

        def closure():
            nonlocal closure_calls
            closure_calls += 1
            lbfgs.zero_grad()
            l_ = forward_loss()
            l_.backward()
            return l_

        iters_done = 0
        while iters_done < args.lbfgs_iters:
            lbfgs.step(closure)
            iters_done += 20
            with torch.no_grad():
                r2_now, l_now = raw_res_l2()
            print(f"[e1] lbfgs ~iter {iters_done:4d} loss={l_now:.6e} "
                  f"R={r2_now:.3e} closures={closure_calls}", flush=True)

    with torch.no_grad():
        post_r2, post_loss = raw_res_l2()

    final_state = (w_k_t + S * z).detach().numpy().copy()
    bridge.set_state(final_state)
    rt = float(np.max(np.abs(bridge.initial_state() - final_state)))
    r_wk = bridge.residual(w_k)
    l_wk = 0.5 * float(r_wk @ r_wk)
    # true cold-start reference: the k=0 partial-primal capture (initial
    # OpenFOAM fields), NOT the current solver state (which is W_final)
    w0_path = os.path.join(PROJECT_ROOT, "outputs", "work",
                           "partial-k0", "W_k.npy")
    if os.path.exists(w0_path):
        r_w0 = bridge.residual(np.load(w0_path))
    else:
        r_w0 = bridge.residual(np.asarray(bridge.initial_state()))
    l_w0 = 0.5 * float(r_w0 @ r_w0)
    loss_final = 0.5 * float(np.linalg.norm(bridge.residual(final_state))) ** 2
    ratio_warm = l_wk / max(loss_final, 1e-300)
    ratio_total = l_w0 / max(loss_final, 1e-300)
    ratio_warmup = l_w0 / max(l_wk, 1e-300)

    out = os.path.join(PROJECT_ROOT, "outputs", "free_state_warm.json")
    write_json(out, {
        "k": args.k, "steps": args.steps, "lr": args.lr, "clip": args.clip,
        "mask_phi": not args.no_mask_phi,
        "history": hist, "rejected": rejected, "accepted": accepted,
        "entry_residual_l2": entry_res,
        "loss_at_warm": l_wk, "loss_at_w0": l_w0, "loss_final": loss_final,
        "ratio_from_warm": ratio_warm, "ratio_total": ratio_total,
        "ratio_warmup_only": ratio_warmup,
        "roundtrip_inf_err": rt,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    })
    print(f"[e1] k={args.k} L(W0)={l_w0:.4e} L(Wk)={l_wk:.4e} "
          f"L(final)={loss_final:.4e}")
    print(f"[e1] ratios: warm {ratio_warm:.2f}x total {ratio_total:.2f}x "
          f"warmup-only {ratio_warmup:.2f}x; rejected={rejected} rt={rt:.1e}")
    ok = (ratio_warm >= 100.0 and rt == 0.0 and accepted > 0
          and np.all(np.isfinite(final_state)))
    print(f"[e1] Gate E1: {'PASS' if ok else 'FAIL'} (>=100x from warm state)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
