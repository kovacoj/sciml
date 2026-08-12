#!/usr/bin/env bash
# MetaCentrum sprint: fixed-budget correction + seeds + warm-start
set -uo pipefail

echo "[$(date)] === METACENTRUM SPRINT ==="

# Find project
PROJECT_DIR="$HOME/sciml/dafoam_pytorch_hfdib_demo"
cd "$PROJECT_DIR" || { echo "Project not found"; exit 1; }

# Check if inside container or host
if [ -f /home/dafoamuser/dafoam/loadDAFoam.sh ]; then
    source /home/dafoamuser/dafoam/loadDAFoam.sh
    export PYTHONPATH=$PWD/python:${PYTHONPATH:-}
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export MPLCONFIGDIR=/tmp/mplcfg
    PYTHON="python"
else
    export PYTHONPATH=$PWD/python:${PYTHONPATH:-}
    PYTHON="python3"
fi

mkdir -p outputs/meta_sprint

# ================================================================
# PHASE 1: Generate missing K=5 targets for topology_512-527
# (needed for exact N=512 K=5)
# ================================================================
echo "[$(date)] Phase 1: Generate missing K=5 targets for 16 new topologies"

# Update split to only include missing topologies
$PYTHON -c "
import json
train = [f'topology_{i:03d}' for i in range(512, 528)]
with open('datasets/four_port_64/splits.json','w') as f:
    json.dump({'train': train, 'test': []}, f, indent=2)
"

if $PYTHON -m unet.generate_simple_targets --dataset datasets/four_port_64 --k-max 80 --k-primary 20 --save-ks 5,10,20,40,80 2>&1 | grep -E "Done|Error"; then
    echo "[$(date)] K=5 targets generated"
else
    echo "[$(date)] WARNING: K=5 target generation failed"
fi

# Restore full split with 512 training topologies
$PYTHON -c "
import json
train = [f'topology_{i:03d}' for i in range(528) if i not in range(128,144)]
test = [f'topology_{i:03d}' for i in range(128,144)]
with open('datasets/four_port_64/splits.json','w') as f:
    json.dump({'train': train, 'test': test}, f, indent=2)
print(f'Train: {len(train)}, Test: {len(test)}')
"

# ================================================================
# PHASE 2: Train N=512 K=5
# ================================================================
echo "[$(date)] Phase 2: Train N=512 K=5"

$PYTHON -m unet.train \
    --mode solver-distilled --architecture simple \
    --dataset datasets/four_port_64 --split train \
    --steps 3000 --topology-batch-size 16 \
    --lr 1e-3 --save-every 500 --eval-every 200 \
    --target-k 5 --seed 42 \
    --output outputs/meta_sprint/n512_k5 2>&1 | grep -E "distill|field|done|Error"

echo "[$(date)] N=512 K=5 done"

# ================================================================
# PHASE 3: Three genuine seeds for N=128 K=20
# ================================================================
echo "[$(date)] Phase 3: Three seeds for N=128 K=20"

# Update split to N=128
$PYTHON -c "
import json
train = [f'topology_{i:03d}' for i in range(128)]
test = [f'topology_{i:03d}' for i in range(128,144)]
with open('datasets/four_port_64/splits.json','w') as f:
    json.dump({'train': train, 'test': test}, f, indent=2)
"

for SEED in 0 1 2; do
    echo "[$(date)] Training seed=$SEED..."
    $PYTHON -m unet.train \
        --mode solver-distilled --architecture simple \
        --dataset datasets/four_port_64 --split train \
        --steps 3000 --topology-batch-size 16 \
        --lr 1e-3 --save-every 500 --eval-every 200 \
        --target-k 20 --seed $SEED \
        --output outputs/meta_sprint/seed_k20_s${SEED} 2>&1 | grep -E "distill|field|done|Error"
    echo "[$(date)] Seed $SEED done"
done

# ================================================================
# PHASE 4: Evaluate all seeds + N=512 K=5 on 16 test topologies
# ================================================================
echo "[$(date)] Phase 4: Evaluation"

# Restore full split
$PYTHON -c "
import json
train = [f'topology_{i:03d}' for i in range(528) if i not in range(128,144)]
test = [f'topology_{i:03d}' for i in range(128,144)]
with open('datasets/four_port_64/splits.json','w') as f:
    json.dump({'train': train, 'test': test}, f, indent=2)
"

for CONFIG in n512_k5 seed_k20_s0 seed_k20_s1 seed_k20_s2; do
    CKPT="outputs/meta_sprint/$CONFIG/checkpoint.pt"
    if [ -f "$CKPT" ]; then
        echo "  Evaluating $CONFIG..."
        $PYTHON -m unet.plot_comparison \
            --dataset datasets/four_port_64 --split test \
            --supervised-ckpt outputs/unet_supervised_unet/checkpoint.pt \
            --physics-ckpt $CKPT \
            --output outputs/meta_sprint/eval_$CONFIG 2>&1 | tail -5
    fi
done

# ================================================================
# PHASE 5: Summary
# ================================================================
echo "[$(date)] Phase 5: Summary"

$PYTHON -c "
import json, os, csv, numpy as np

print()
print('=== FIXED BUDGET STUDY (N×K=2560) ===')
print(f'{\"N\":>5} {\"K\":>5} {\"N*K\":>6} {\"test rel_U\":>12} {\"test rel_p\":>12}')

configs = [(512,5),(256,10),(128,20),(64,40),(32,80)]
for N, K in configs:
    tag = f'n{N}_k{K}'
    # Try fixed_budget_study first, then meta_sprint
    for path in [f'outputs/fixed_budget_study/eval_{tag}/metrics.json',
                f'outputs/meta_sprint/eval_{tag}/metrics.json']:
        if os.path.exists(path):
            m = json.load(open(path))
            us = [x['physics']['rel_velocity_l2'] for x in m]
            ps = [x['physics']['rel_pressure_l2'] for x in m]
            print(f'{N:>5} {K:>5} {N*K:>6} {np.mean(us):>12.4e} {np.mean(ps):>12.4e}')
            break
    else:
        print(f'{N:>5} {K:>5} {N*K:>6} {\"---\":>12} {\"---\":>12}')

print()
print('=== SEED VARIANCE (N=128, K=20) ===')
print(f'{\"seed\":>5} {\"test rel_U\":>12} {\"test rel_p\":>12}')
seed_results = []
for seed in [0,1,2]:
    path = f'outputs/meta_sprint/eval_seed_k20_s{seed}/metrics.json'
    if os.path.exists(path):
        m = json.load(open(path))
        us = [x['physics']['rel_velocity_l2'] for x in m]
        ps = [x['physics']['rel_pressure_l2'] for x in m]
        print(f'{seed:>5} {np.mean(us):>12.4e} {np.mean(ps):>12.4e}')
        seed_results.append({'seed': seed, 'rel_u': float(np.mean(us)), 'rel_p': float(np.mean(ps))})

if len(seed_results) >= 2:
    us = [r['rel_u'] for r in seed_results]
    ps = [r['rel_p'] for r in seed_results]
    print(f'{\"mean\":>5} {np.mean(us):>12.4e} {np.mean(ps):>12.4e}')
    print(f'{\"std\":>5} {np.std(us):>12.4e} {np.std(ps):>12.4e}')

with open('outputs/meta_sprint/summary.json', 'w') as f:
    json.dump({'fixed_budget': configs, 'seeds': seed_results}, f, indent=2)
"

echo "[$(date)] === SPRINT COMPLETE ==="
