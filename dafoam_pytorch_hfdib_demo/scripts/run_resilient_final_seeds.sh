#!/usr/bin/env bash
# Run the final local-distillation seeds sequentially and resume after
# interruption. Intended for the campaign container after environment setup.
set -euo pipefail

latest_checkpoint() {
    local output=$1
    python -c '
import sys
from pathlib import Path
import torch

output = Path(sys.argv[1])
valid = []
for path in output.glob("checkpoint_s*.pt"):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        step = int(checkpoint["step"])
        if checkpoint.get("mode") != "solver-distilled":
            continue
        valid.append((step, path))
    except Exception as error:
        print(f"[campaign] ignoring invalid {path}: {error}", file=sys.stderr)
if valid:
    print(max(valid)[1])
' "$output"
}

for seed in 22 33 44 55; do
    output="outputs/final_campaign/local_distill_n128_k20_seed${seed}"
    log="outputs/final_campaign/local_distill_n128_k20_seed${seed}.log"

    if [[ -f "$output/checkpoint.pt" ]]; then
        echo "[campaign] seed $seed already complete"
        continue
    fi

    resume=$(latest_checkpoint "$output")
    args=(
        --mode solver-distilled
        --architecture simple
        --dataset datasets/four_port_64
        --split train
        --steps 10000
        --topology-batch-size 4
        --lr 1e-3
        --save-every 250
        --eval-every 100
        --target-k 20
        --seed "$seed"
        --output "$output"
    )
    if [[ -n "$resume" ]]; then
        echo "[campaign] resuming seed $seed from $resume"
        args+=(--resume "$resume")
    else
        echo "[campaign] starting seed $seed"
    fi

    nice -n 10 python -m unet.train "${args[@]}" 2>&1 | tee -a "$log"
    echo "[campaign] completed seed $seed"
done
