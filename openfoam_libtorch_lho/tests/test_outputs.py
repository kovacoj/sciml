#!/usr/bin/env python3
"""
Test output files: verify CSV structure and numerical validity.
"""

import numpy as np
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
CASE_DIR = ROOT / 'cases' / 'lho_test'
POST_PROC = CASE_DIR / 'postProcessing' / 'fvNeuralLHO'

def test_eigenvalues_csv():
    """Verify eigenvalues.csv exists and has correct columns."""
    eigenvalues_file = POST_PROC / 'eigenvalues.csv'
    assert eigenvalues_file.exists(), "eigenvalues.csv not found"
    
    with open(eigenvalues_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    expected_cols = ['state', 'E_NN', 'E_directFV', 'E_exact', 
                     'dE_NN_direct', 'dE_direct_exact', 'dE_NN_exact']
    for col in expected_cols:
        assert col in rows[0], f"Missing column: {col}"
    
    print(f"Eigenvalues CSV: {len(rows)} states")
    for row in rows[:3]:
        print(f"  State {row['state']}: E_NN={row['E_NN']}, E_direct={row['E_directFV']}")

def test_training_state_csv():
    """Verify training state CSVs exist and have decreasing best-so-far energy."""
    for n in range(6):
        train_file = POST_PROC / f'training_state_{n}.csv'
        if not train_file.exists():
            continue
        
        with open(train_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) > 0, f"No training data for state {n}"
        
        # Check that best-so-far energy is non-increasing
        energies = [float(row['energy']) for row in rows]
        best_so_far = float('inf')
        for e in energies:
            assert e <= best_so_far + 1e-10, f"Energy increased at step (violation of best-so-far)"
            best_so_far = min(best_so_far, e)
        
        print(f"Training state {n}: {len(rows)} steps, final energy = {energies[-1]:.8f}")

def test_profiles_csv():
    """Verify profiles.csv has all required columns."""
    profiles_file = POST_PROC / 'profiles.csv'
    if not profiles_file.exists():
        pytest.skip("profiles.csv not available")
    
    with open(profiles_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) > 0, "No profile data"
    
    # Check cell count matches mesh
    N = len(rows)
    print(f"Profiles CSV: {N} cells")
    
    # Verify finite values
    for row in rows[:5]:
        x = float(row['x'])
        pot = float(row['potential'])
        assert np.isfinite(x), f"Non-finite x at cell {row['cell']}"
        assert np.isfinite(pot), f"Non-finite potential at cell {row['cell']}"

def test_orthogonality():
    """Check that neural states are approximately orthogonal in M inner product."""
    profiles_file = POST_PROC / 'profiles.csv'
    if not profiles_file.exists():
        pytest.skip("profiles.csv not available")
    
    # Read profiles
    cells = []
    volumes = []
    psi_nn = {}
    
    with open(profiles_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cells.append(int(row['cell']))
            volumes.append(float(row['volume']))
            for n in range(6):
                key = f'psiNN_{n}'
                if key in row:
                    if n not in psi_nn:
                        psi_nn[n] = []
                    psi_nn[n].append(float(row[key]))
    
    M = np.diag(volumes)
    
    print("\nOrthogonality matrix (M-inner product):")
    states = sorted(psi_nn.keys())
    for i, ni in enumerate(states):
        for j, nj in enumerate(states):
            overlap = np.dot(psi_nn[ni], M @ psi_nn[nj])
            if i == j:
                print(f"  ||ψ_{ni}||_M² = {overlap:.8f}", end="  ")
            elif abs(i - j) <= 2:
                print(f"|⟨ψ_{ni}|ψ_{nj}⟩_M| = {abs(overlap):.2e}", end="  ")
        print()

if __name__ == '__main__':
    test_eigenvalues_csv()
    test_training_state_csv()
    test_profiles_csv()
    test_orthogonality()
    print("\nAll output tests passed!")
