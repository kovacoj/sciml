#!/usr/bin/env bash
# Fixed-budget study: N×K = 2560 SIMPLE iterations total.
# Compares sample diversity (large N, small K) vs teacher fidelity (small N, large K).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

IMG="${SCIML_HFDIB_IMAGE:-sciml-dafoam-torch-hfdib:latest}"
LOGDIR="/tmp/opencode"
RESULTS_DIR="$PROJECT/outputs/fixed_budget_study"
mkdir -p "$RESULTS_DIR"

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

# Configurations: N_train × K = 2560
# (N=128, K=20) already exists — reuse
CONFIGS=(
    "512 5"
    "256 10"
    "128 20"   # existing
    "64 40"
    "32 80"
)

echo "[$(date)] Fixed-budget study: N×K=2560"

# ================================================================
# Step 1: Generate 512 topologies + HFDIB dataset
# ================================================================
if [ ! -f "$RESULTS_DIR/.done_topologies" ]; then
    echo "[$(date)] Generating 512 topologies..."
    cd "$PROJECT"
    PYTHONPATH=python python3 -m unet.generate_case 2>&1 | tail -5
    touch "$RESULTS_DIR/.done_topologies"
fi

if [ ! -f "$RESULTS_DIR/.done_dataset" ]; then
    echo "[$(date)] Generating HFDIB dataset (512 topologies)..."
    cd "$PROJECT"
    rm -rf datasets/four_port_64
    ./scripts/generate_four_port_dataset.sh 2>&1 | grep -E "\[dataset\]|\[A\]|\[B\]|\[C\]|done|Error" | tail -20
    touch "$RESULTS_DIR/.done_dataset"
fi

# ================================================================
# Step 2: Generate K=80 SIMPLE targets (covers K=5,10,20,40,80)
# ================================================================
if [ ! -f "$RESULTS_DIR/.done_targets" ]; then
    echo "[$(date)] Generating K=80 SIMPLE targets (saves K=5,10,20,40,80)..."
    run_in_container "python -m unet.generate_simple_targets \
        --dataset datasets/four_port_64 \
        --k-max 80 --k-primary 20 \
        --save-ks 5,10,20,40,80" 2>&1 | grep -E "\[targets\]|E_|rel_|Done" | tail -20
    touch "$RESULTS_DIR/.done_targets"
fi

# ================================================================
# Step 3: Train each configuration
# ================================================================
for CONFIG in "${CONFIGS[@]}"; do
    N=$(echo $CONFIG | cut -d' ' -f1)
    K=$(echo $CONFIG | cut -d' ' -f2)
    TAG="n${N}_k${K}"
    MARKER="$RESULTS_DIR/.done_train_${TAG}"

    if [ -f "$MARKER" ]; then
        echo "[$(date)] Skipping $TAG (already done)"
        continue
    fi

    echo "[$(date)] Training $TAG (N=$N, K=$K)..."

    # Update split file for this N
    python3 -c "
import json
train = [f'topology_{i:03d}' for i in range($N) if i not in range(128,144)]
test = [f'topology_{i:03d}' for i in range(128,144)]
splits = {'train': train, 'test': test}
with open('$PROJECT/datasets/four_port_64/splits.json','w') as f:
    json.dump(splits, f, indent=2)
"

    OUTPUT="outputs/fixed_budget_study/${TAG}"
    run_in_container "python -m unet.train \
        --mode solver-distilled \
        --architecture simple \
        --dataset datasets/four_port_64 \
        --split train \
        --steps 3000 \
        --topology-batch-size 16 \
        --lr 1e-3 \
        --save-every 500 \
        --eval-every 200 \
        --target-k $K \
        --seed 42 \
        --output $OUTPUT \
        2>&1 | grep -E '\[distill\]|\[field\]|done|Error'" 2>&1 | tail -20

    touch "$MARKER"
    echo "[$(date)] Done: $TAG"
done

# Restore full split
python3 -c "
import json
train = [f'topology_{i:03d}' for i in range(496) if i not in range(128,144)]
test = [f'topology_{i:03d}' for i in range(128,144)]
splits = {'train': train, 'test': test}
with open('$PROJECT/datasets/four_port_64/splits.json','w') as f:
    json.dump(splits, f, indent=2)
"

# ================================================================
# Step 4: Evaluate all configurations on 16 held-out topologies
# ================================================================
echo "[$(date)] Evaluating all configurations..."

for CONFIG in "${CONFIGS[@]}"; do
    N=$(echo $CONFIG | cut -d' ' -f1)
    K=$(echo $CONFIG | cut -d' ' -f2)
    TAG="n${N}_k${K}"
    CKPT="outputs/fixed_budget_study/${TAG}/checkpoint.pt"

    if [ ! -f "$PROJECT/$CKPT" ]; then
        echo "  Skipping $TAG (no checkpoint)"
        continue
    fi

    echo "  Evaluating $TAG..."
    run_in_container "python3 -m unet.plot_comparison \
        --dataset datasets/four_port_64 --split test \
        --supervised-ckpt outputs/unet_supervised_unet/checkpoint.pt \
        --physics-ckpt $CKPT \
        --output outputs/fixed_budget_study/eval_${TAG} \
        2>&1 | tail -10"
done

# ================================================================
# Step 5: Generate summary CSV
# ================================================================
echo "[$(date)] Generating summary..."

run_in_container "
import json, os, csv
import numpy as np

configs = [(512,5),(256,10),(128,20),(64,40),(32,80)]
results = []

for N, K in configs:
    tag = f'n{N}_k{K}'
    eval_path = f'outputs/fixed_budget_study/eval_{tag}/metrics.json'
    if not os.path.exists(eval_path):
        print(f'Missing: {eval_path}')
        continue
    with open(eval_path) as f:
        metrics = json.load(f)
    us = [m['physics']['rel_velocity_l2'] for m in metrics]
    ps = [m['physics']['rel_pressure_l2'] for m in metrics]
    results.append({
        'N': N, 'K': K, 'NK': N*K,
        'mean_test_rel_U': float(np.mean(us)),
        'mean_test_rel_p': float(np.mean(ps)),
    })

with open('outputs/fixed_budget_study/summary.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=['N','K','NK','mean_test_rel_U','mean_test_rel_p'])
    w.writeheader()
    for r in results:
        w.writerow(r)

print()
print('Fixed-budget study results (N×K=2560):')
print(f'{\"N\":>5} {\"K\":>5} {\"N*K\":>6} {\"test rel_U\":>12} {\"test rel_p\":>12}')
for r in results:
    print(f'{r[\"N\"]:>5} {r[\"K\"]:>5} {r[\"NK\"]:>6} {r[\"mean_test_rel_U\"]:>12.4e} {r[\"mean_test_rel_p\"]:>12.4e}')
" 2>&1 | tail -15

echo "[$(date)] Fixed-budget study complete!"
