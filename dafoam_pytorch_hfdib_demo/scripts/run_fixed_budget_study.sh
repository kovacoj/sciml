#!/usr/bin/env bash
# Fixed-budget study: N×K = 2560 SIMPLE iterations total.
# NO converged HFDIB needed for training — only for 16 test topologies.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

IMG="${SCIML_HFDIB_IMAGE:-sciml-dafoam-torch-hfdib:latest}"
RESULTS_DIR="$PROJECT/outputs/fixed_budget_study"
mkdir -p "$RESULTS_DIR"

run_in_container() {
    docker run --rm --ipc=host \
        --user "$(id -u):$(id -g)" \
        -e HOME=/home/dafoamuser \
        -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
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

CONFIGS=("512 5" "256 10" "128 20" "64 40" "32 80")
echo "[$(date)] Fixed-budget study: N×K=2560"

# ================================================================
# Step 1: Generate topologies (host-side, already done if 513 dirs exist)
# ================================================================
N_TOPOS=$(ls "$PROJECT/topologies/four_port_64/" | grep -c topology)
if [ "$N_TOPOS" -lt 512 ]; then
    echo "[$(date)] Generating 512 topologies..."
    cd "$PROJECT"
    PYTHONPATH=python python3 -m unet.generate_case 2>&1 | tail -5
fi
echo "[$(date)] Topologies ready"

# ================================================================
# Step 2: Prepare dataset directory — blockMesh + signed distance for ALL
#          + converged HFDIB only for test (128-143)
# ================================================================
if [ ! -f "$RESULTS_DIR/.done_dataset" ]; then
    echo "[$(date)] Preparing dataset (blockMesh + signed distance for all, HFDIB only for test)..."

    # Clean and create dataset dir
    rm -rf "$PROJECT/datasets/four_port_64"
    mkdir -p "$PROJECT/datasets/four_port_64/shared"

    # Copy shared case template
    cp -r "$PROJECT/cases/four_port_64x64" "$PROJECT/datasets/four_port_64/case_template"

    # Write splits
    python3 -c "
import json
train = [f'topology_{i:03d}' for i in range(496)]
test = [f'topology_{i:03d}' for i in range(128,144)]
splits = {'train': train, 'test': test}
with open('$PROJECT/datasets/four_port_64/splits.json','w') as f:
    json.dump(splits, f, indent=2)
"

    # Prepare dataset: blockMesh + signed distance for all, HFDIB only for test
    run_in_container "python -m unet.prepare_fixed_budget_dataset" 2>&1 | grep -E "Found|Done|Running|Error|Traceback" | tail -20

    # Clean up template
    rm -rf "$PROJECT/datasets/four_port_64/case_template"
    touch "$RESULTS_DIR/.done_dataset"
    echo "[$(date)] Dataset prepared"
fi

# ================================================================
# Step 3: Generate K=80 SIMPLE targets for training topologies
# ================================================================
if [ ! -f "$RESULTS_DIR/.done_targets" ]; then
    echo "[$(date)] Generating K=80 SIMPLE targets..."
    run_in_container "python -m unet.generate_simple_targets \
        --dataset datasets/four_port_64 \
        --k-max 80 --k-primary 20 \
        --save-ks 5,10,20,40,80" 2>&1 | grep -E "\[targets\]|E_|rel_|Done" | tail -20
    touch "$RESULTS_DIR/.done_targets"
    echo "[$(date)] SIMPLE targets generated"
fi

# ================================================================
# Step 4: Train each configuration
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

    # Update split for this N (train = first N non-test topologies)
    python3 -c "
import json
train = [f'topology_{i:03d}' for i in range($N) if i not in range(128,144)]
test = [f'topology_{i:03d}' for i in range(128,144)]
with open('$PROJECT/datasets/four_port_64/splits.json','w') as f:
    json.dump({'train': train, 'test': test}, f, indent=2)
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
train = [f'topology_{i:03d}' for i in range(496)]
test = [f'topology_{i:03d}' for i in range(128,144)]
with open('$PROJECT/datasets/four_port_64/splits.json','w') as f:
    json.dump({'train': train, 'test': test}, f, indent=2)
"

# ================================================================
# Step 5: Evaluate all configurations on 16 held-out topologies
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
# Step 6: Generate summary
# ================================================================
echo "[$(date)] Generating summary..."

run_in_container "
import json, os, csv, numpy as np

configs = [(512,5),(256,10),(128,20),(64,40),(32,80)]
results = []

for N, K in configs:
    tag = f'n{N}_k{K}'
    eval_path = f'outputs/fixed_budget_study/eval_{tag}/metrics.json'
    if not os.path.exists(eval_path):
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
print('Fixed-budget study (N×K=2560):')
print(f'{\"N\":>5} {\"K\":>5} {\"N*K\":>6} {\"test rel_U\":>12} {\"test rel_p\":>12}')
for r in results:
    print(f'{r[\"N\"]:>5} {r[\"K\"]:>5} {r[\"NK\"]:>6} {r[\"mean_test_rel_U\"]:>12.4e} {r[\"mean_test_rel_p\"]:>12.4e}')
" 2>&1 | tail -15

echo "[$(date)] Fixed-budget study complete!"
