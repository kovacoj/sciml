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

    # Generate blockMesh + signed distance for all 512 topologies
    # + converged HFDIB only for test topologies (128-143)
    run_in_container "
import json, os, shutil, subprocess, sys
import numpy as np
from pathlib import Path

PYTHON_ROOT = Path('.')
sys.path.insert(0, str(PYTHON_ROOT / 'python'))
from common import hfdib_signed_distance_options, PROJECT_ROOT
from dafoam_bridge import DAFoamResidualBridge
from state_layout import build_isothermal_layout
from unet.generate_case import mask_to_signed_distance_64, write_signed_distance_file

ds_dir = Path('datasets/four_port_64')
case_template = str(ds_dir / 'case_template')

# Load topology dirs
topo_src = Path('topologies/four_port_64')
topo_dirs = sorted([d for d in topo_src.iterdir() if d.is_dir() and d.name.startswith('topology_')])
print(f'Found {len(topo_dirs)} topologies')

# Process each topology
for idx, topo_dir in enumerate(topo_dirs):
    tid = topo_dir.name
    mask = np.load(topo_dir / 'mask.npy')
    out_dir = ds_dir / tid
    case_dir = out_dir / 'case'

    # Skip if already fully set up
    if (out_dir / 'signed_distance.npy').exists() and (case_dir / 'constant' / 'polyMesh').exists():
        # Check if HFDIB converged for test topologies
        is_test = 128 <= int(tid.split('_')[1]) < 144
        if is_test and (out_dir / 'ux_hfdib.npy').exists():
            continue
        elif not is_test:
            continue

    print(f'[{idx+1}/{len(topo_dirs)}] {tid}...', flush=True)

    # Create case dir
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.copytree(case_template, case_dir)
    for d in os.listdir(case_dir):
        if d[0].isdigit() and d != '0':
            shutil.rmtree(os.path.join(case_dir, d), ignore_errors=True)
    shutil.rmtree(os.path.join(case_dir, 'constant', 'polyMesh'), ignore_errors=True)
    shutil.rmtree(os.path.join(case_dir, 'postProcessing'), ignore_errors=True)

    # Run blockMesh
    subprocess.run(['blockMesh', '-case', case_dir], check=True, capture_output=True)

    # Generate signed distance + lambda
    psi = mask_to_signed_distance_64(mask)
    write_signed_distance_file(case_dir, psi)
    np.save(out_dir / 'signed_distance.npy', psi)
    np.save(out_dir / 'mask.npy', mask)

    h = np.sqrt((0.128/64) * (0.128/64))
    lam = 0.5 * (1.0 - np.tanh(psi / (1.5 * h)))
    np.save(out_dir / 'lambda.npy', lam)

    # For test topologies (128-143): run converged HFDIB
    tid_num = int(tid.split('_')[1])
    if 128 <= tid_num < 144:
        print(f'  Running converged HFDIB for {tid}...', flush=True)
        os.chdir(case_dir)
        from mpi4py import MPI
        bridge = DAFoamResidualBridge(
            str(case_dir),
            hfdib_signed_distance_options(str(case_dir),
                inlet_patches=['inletLower','inletUpper'],
                outlet_patches=['outletLower','outletUpper']),
            comm=MPI.COMM_SELF)
        bridge.solver()
        w = np.ascontiguousarray(bridge.solver.getStates().copy(), dtype=np.float64)

        n_cells = 4096
        n_u = 3 * n_cells
        u_cells = w[:n_u].reshape(n_cells, 3)
        p_cells = w[n_u:n_u+n_cells]

        np.save(out_dir / 'ux_hfdib.npy', u_cells[:, 0].reshape(64, 64))
        np.save(out_dir / 'uy_hfdib.npy', u_cells[:, 1].reshape(64, 64))
        np.save(out_dir / 'pressure_hfdib.npy', p_cells.reshape(64, 64))
        np.save(out_dir / 'metadata.json', {'topology_id': tid, 'converged': True})
        del bridge
        os.chdir(str(PROJECT_ROOT / 'dafoam_pytorch_hfdib_demo'))
    else:
        # Training topology: save metadata, no HFDIB solve
        with open(out_dir / 'metadata.json', 'w') as f:
            json.dump({'topology_id': tid, 'converged': False}, f)

# Build shared mesh metadata from first topology
first_case = str(ds_dir / 'topology_000' / 'case')
from unet.generate_dataset import build_64x64_mesh_metadata
mesh_meta = build_64x64_mesh_metadata(first_case)
mesh_meta.save(str(ds_dir / 'shared' / 'mesh_metadata.npz'),
               str(ds_dir / 'shared' / 'mesh_metadata.json'))

# Build base state k=0
os.chdir(first_case)
from mpi4py import MPI
bridge_k0 = DAFoamResidualBridge(
    first_case,
    hfdib_signed_distance_options(first_case,
        inlet_patches=['inletLower','inletUpper'],
        outlet_patches=['outletLower','outletUpper']),
    comm=MPI.COMM_SELF)
w0 = np.ascontiguousarray(bridge_k0.solver.getStates().copy(), dtype=np.float64)
np.save(ds_dir / 'shared' / 'base_state_k0.npy', w0)

# Loss config
r0 = bridge_k0.residual(w0)
layout = build_isothermal_layout(mesh_meta.n_cells, mesh_meta.n_faces)
u_ids = layout.indices('U')
p_ids = layout.indices('p')
phi_ids = layout.indices('phi')
lu = 0.5 * float(np.dot(r0[u_ids], r0[u_ids]))
lp = 0.5 * float(np.dot(r0[p_ids], r0[p_ids]))
lphi = 0.5 * float(np.dot(r0[phi_ids], r0[phi_ids]))
loss_config = {'gamma_u': 1.0/(lu+1e-30), 'gamma_p': 1.0/(lp+1e-30), 'gamma_phi': 1.0/(lphi+1e-30)}
with open(ds_dir / 'shared' / 'physics_loss_config.json', 'w') as f:
    json.dump(loss_config, f, indent=2)

print(f'Done: {len(topo_dirs)} topologies prepared')
print(f'  Test (128-143): converged HFDIB')
print(f'  Train: blockMesh + signed distance only (no converged CFD)')
" 2>&1 | grep -E "Found|Done|Running|Error|Traceback" | tail -20

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
