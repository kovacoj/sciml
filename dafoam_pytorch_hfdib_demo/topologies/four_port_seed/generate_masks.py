"""Generate four representative four-port topology masks."""
import numpy as np
import json
import os

# 8x8 masks for the four-port metamaterial problem
# Ports at rows 1 and 6 (from top), columns 0 and 7
# 1 = solid, 0 = fluid

topologies = {
    "topology_000": {
        "name": "straight_connection",
        "mask": np.array([
            [1,1,1,0,0,1,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [1,1,1,0,0,1,1,1],
        ]),
    },
    "topology_001": {
        "name": "split_connection",
        "mask": np.array([
            [1,1,1,0,0,1,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,1,1,0,0,0],
            [1,1,0,1,1,0,1,1],
            [1,1,0,1,1,0,1,1],
            [0,0,0,1,1,0,0,0],
            [0,0,0,0,0,0,0,0],
            [1,1,1,0,0,1,1,1],
        ]),
    },
    "topology_002": {
        "name": "central_junction",
        "mask": np.array([
            [1,1,1,0,0,1,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [1,1,0,0,0,0,1,1],
            [1,1,0,0,0,0,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [1,1,1,0,0,1,1,1],
        ]),
    },
    "topology_003": {
        "name": "asymmetric_bent",
        "mask": np.array([
            [1,1,1,0,0,1,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,1,1,0,0],
            [1,1,0,0,1,1,0,0],
            [1,1,0,0,0,0,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [1,1,1,0,0,1,1,1],
        ]),
    },
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))

# Write dataset.json
with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
    json.dump({
        "case_template": "cases/single_obstacle",
        "design_bounds": [0.2, 0.8, 0.01, 0.09],
        "mask_convention": "1=solid,0=fluid",
        "target_problem": "four-port immersed-boundary topology family",
    }, f, indent=2)

# Write topology directories
for tid, info in topologies.items():
    topo_dir = os.path.join(dataset_dir, tid)
    os.makedirs(topo_dir, exist_ok=True)
    np.save(os.path.join(topo_dir, "mask.npy"), info["mask"])
    with open(os.path.join(topo_dir, "topology.json"), "w") as f:
        json.dump({"topology_id": tid, "name": info["name"]}, f, indent=2)

print(f"Generated {len(topologies)} topologies in {dataset_dir}")
