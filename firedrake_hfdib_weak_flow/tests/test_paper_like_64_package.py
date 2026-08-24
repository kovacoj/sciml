import json
from pathlib import Path

import numpy as np


def test_paper_like_64_geometry_package_is_complete():
    root = Path(__file__).parents[1] / "outputs/supervisor_2026_08_24/topic2_firedrake/paper_like_64"
    if not root.exists():
        return
    selected = json.loads((root / "selected_cases.json").read_text())
    assert len(selected) == 3
    for case, selection in selected.items():
        case_dir = root / "cases" / case
        assert (case_dir / "lambda_64x64.npy").exists()
        assert (case_dir / "binary_mask.npy").exists()
        assert (case_dir / "bitmap_8x8.npy").exists()
        assert (case_dir / "mesh.msh").exists()
        assert (case_dir / "connectivity.json").exists()
        connectivity = json.loads((case_dir / "connectivity.json").read_text())
        assert connectivity["status"] == "PASS"
        lam = np.load(case_dir / "lambda_64x64.npy")
        assert lam.shape == (64, 64)
        binary = np.load(case_dir / "binary_mask.npy")
        assert binary.shape == (64, 64)
        coarse = np.load(case_dir / "bitmap_8x8.npy")
        assert coarse.shape == (8, 8)
    assert (root / "figures" / "paper_like_64_selected.png").exists()
    assert (root / "figures" / "paper_like_64_candidate_contact_sheet.png").exists()
    status = json.loads((root / "status.json").read_text())
    assert status["cfd_status"]["P64_A"] == "NOT_RUN_DOCKER_UNAVAILABLE"
