#!/usr/bin/env bash
# Overnight K=20 scaling experiment campaign.
# Generates 144-topology dataset, K=20 targets, trains N=16/32/64/128 × 3 seeds,
# evaluates on 16 held-out topologies, runs warm-start experiment.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

IMG="${SCIML_HFDIB_IMAGE:-sciml-dafoam-torch-hfdib:latest}"
LOGDIR="/tmp/opencode"

run_in_container() {
    docker run --rm --ipc=host \
        --user "$(id -u):$(id -g)" \
        -e HOME=/home/dafoamuser \
        -e OMP_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 \
        -e MKL_NUM_THREADS=1 \
        -e MPLCONFIGDIR=/tmp/mplcfg \
        --mount type=bind,src="$PROJECT/..",target=/home/dafoamuser/sciml \
        -w /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo \
        "$IMG" bash -lc "
            source \$HOME/activate_dafoam_torch.sh
            export PYTHONPATH=\$PWD/python:\${PYTHONPATH:-}
            cd /home/dafoamuser/sciml/dafoam_pytorch_hfdib_demo
            $1
        "
}

DONE_MARKER="$PROJECT/outputs/overnight_k20_scaling"
mkdir -p "$DONE_MARKER"

echo "[$(date)] Starting overnight K=20 scaling campaign"

# ================================================================
# Step 1: Generate 144 topologies (host-side)
# ================================================================
if [ ! -f "$DONE_MARKER/.done_topologies" ]; then
    echo "[$(date)] Step 1: Generating 144 topologies..."
    cd "$PROJECT"
    PYTHONPATH=python python3 -m unet.generate_case 2>&1 | tail -5
    touch "$DONE_MARKER/.done_topologies"
    echo "[$(date)] Done: 144 topologies generated"
fi

# ================================================================
# Step 2: Generate full dataset (144 converged HFDIB solves)
# ================================================================
if [ ! -f "$DONE_MARKER/.done_dataset" ]; then
    echo "[$(date)] Step 2: Generating 144-topology HFDIB dataset..."
    cd "$PROJECT"
    rm -rf datasets/four_port_64
    ./scripts/generate_four_port_dataset.sh 2>&1 | grep -E "\[dataset\]|\[A\]|\[B\]|\[C\]|done|Error" | tail -20
    touch "$DONE_MARKER/.done_dataset"
    echo "[$(date)] Done: 144-topology dataset generated"
fi

# ================================================================
# Step 3: Generate K=20 SIMPLE targets for 128 training topologies
# ================================================================
if [ ! -f "$DONE_MARKER/.done_targets_128" ]; then
    echo "[$(date)] Step 3: Generating K=20 SIMPLE targets for 128 topologies..."
    run_in_container "python -m unet.generate_simple_targets --dataset datasets/four_port_64 --k-max 20 --k-primary 10" 2>&1 | grep -E "\[targets\]|E_|rel_|Done" | tail -20
    touch "$DONE_MARKER/.done_targets_128"
    echo "[$(date)] Done: K=20 targets generated"
fi

# ================================================================
# Step 4: Train N=16,32,64,128 × 3 seeds × 2000 steps
# ================================================================
for N in 16 32 64 128; do
    for SEED in 0 1 2; do
        MARKER="$DONE_MARKER/.done_n${N}_seed${SEED}"
        if [ -f "$MARKER" ]; then
            echo "[$(date)] Skipping N=$N seed=$SEED (already done)"
            continue
        fi

        echo "[$(date)] Training N=$N seed=$SEED..."

        # Build training subset topology list
        TOPO_LIST=""
        for i in $(seq 0 $((N-1))); do
            TOPO_LIST="${TOPO_LIST}topology_$(printf '%03d' $i) "
        done

        # Create a temporary split file for this N
        python3 -c "
import json
splits = {'train': [f'topology_{i:03d}' for i in range($N)],
          'test': [f'topology_{i:03d}' for i in range(128,144)]}
with open('$PROJECT/datasets/four_port_64/splits.json','w') as f:
    json.dump(splits, f, indent=2)
"

        OUTPUT="outputs/overnight_k20_scaling/n${N}_seed${SEED}"

        run_in_container "
            python -m unet.train \
                --mode solver-distilled \
                --architecture simple \
                --dataset datasets/four_port_64 \
                --split train \
                --steps 2000 \
                --topology-batch-size 16 \
                --lr 1e-3 \
                --save-every 200 \
                --eval-every 100 \
                --target-k 20 \
                --output $OUTPUT \
                2>&1 | grep -E '\[distill\]|\[field\]|done|Error'
        " 2>&1 | tail -20

        touch "$MARKER"
        echo "[$(date)] Done: N=$N seed=$SEED"
    done
done

# Restore full split
run_in_container "
    import json
    splits = {'train': [f'topology_{i:03d}' for i in range(128)],
              'test': [f'topology_{i:03d}' for i in range(128,144)]}
    with open('datasets/four_port_64/splits.json','w') as f:
        json.dump(splits, f, indent=2)
" 2>/dev/null || true

# ================================================================
# Step 5: Evaluate all models on 16 held-out topologies
# ================================================================
echo "[$(date)] Step 5: Evaluating all models on 16 held-out topologies..."

for N in 16 32 64 128; do
    for SEED in 0 1 2; do
        CKPT="outputs/overnight_k20_scaling/n${N}_seed${SEED}/checkpoint.pt"
        if [ ! -f "$PROJECT/$CKPT" ]; then
            echo "  Skipping N=$N seed=$SEED (no checkpoint)"
            continue
        fi

        echo "  Evaluating N=$N seed=$SEED..."
        run_in_container "python3 -m unet.plot_comparison \
            --dataset datasets/four_port_64 --split test \
            --supervised-ckpt outputs/unet_supervised_unet/checkpoint.pt \
            --physics-ckpt $CKPT \
            --output outputs/overnight_k20_scaling/eval_n${N}_seed${SEED} \
            2>&1 | tail -10"
    done
done

# ================================================================
# Step 6: NN warm-start experiment (best N=128 model)
# ================================================================
echo "[$(date)] Step 6: NN warm-start experiment..."

BEST_CKPT="outputs/overnight_k20_scaling/n128_seed0/checkpoint.pt"
if [ -f "$PROJECT/$BEST_CKPT" ]; then
    run_in_container "python3 -m diagnostics.evaluate_warm_start \
        --dataset datasets/four_port_64 \
        --checkpoint $BEST_CKPT \
        --simple-steps 0,1,2,3,5,10,20 \
        2>&1 | grep -E '\[metrics\]|\[warm\]|SUMMARY|rel_U|rel_p|dp_ratio|cont'" | tail -80
else
    echo "  No N=128 checkpoint found for warm-start"
fi

# ================================================================
# Step 7: Generate summary
# ================================================================
echo "[$(date)] Step 7: Generating summary..."

run_in_container "
import json, os, glob
import numpy as np

results = {}
for N in [16, 32, 64, 128]:
    for seed in [0, 1, 2]:
        eval_dir = f'outputs/overnight_k20_scaling/eval_n{N}_seed{seed}'
        metrics_path = os.path.join(eval_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            rel_us = [m['physics']['rel_velocity_l2'] for m in metrics]
            rel_ps = [m['physics']['rel_pressure_l2'] for m in metrics]
            key = f'n{N}_seed{seed}'
            results[key] = {
                'n_train': N,
                'seed': seed,
                'mean_rel_u': float(np.mean(rel_us)),
                'mean_rel_p': float(np.mean(rel_ps)),
                'std_rel_u': float(np.std(rel_us)),
                'std_rel_p': float(np.std(rel_ps)),
            }

with open('outputs/overnight_k20_scaling/summary.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary table
print()
print('N_train | seed | mean rel_U | mean rel_p')
print('-' * 50)
for key, r in sorted(results.items()):
    print(f'{r[\"n_train\"]:>7} | {r[\"seed\"]:>4} | {r[\"mean_rel_u\"]:.4e} | {r[\"mean_rel_p\"]:.4e}')

# Compute mean ± std over seeds
print()
print('N_train | mean rel_U (±std) | mean rel_p (±std)')
print('-' * 60)
for N in [16, 32, 64, 128]:
    us = [r['mean_rel_u'] for k,r in results.items() if r['n_train']==N]
    ps = [r['mean_rel_p'] for k,r in results.items() if r['n_train']==N]
    if us:
        print(f'{N:>7} | {np.mean(us):.4e} ± {np.std(us):.4e} | {np.mean(ps):.4e} ± {np.std(ps):.4e}')
" 2>&1 | tail -30

echo "[$(date)] Campaign complete!"
