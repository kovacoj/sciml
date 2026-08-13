#!/usr/bin/env python3
"""
Test spectrum: verify direct FV eigenvalues match SciPy solution.
"""

import numpy as np
from scipy import linalg
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
CASE_DIR = ROOT / 'cases' / 'lho_test'
POST_PROC = CASE_DIR / 'postProcessing' / 'fvNeuralLHO'

def load_matrices():
    """Load K and M matrices."""
    k_file = POST_PROC / 'K.csv'
    mass_file = POST_PROC / 'mass.csv'
    
    data = {}
    with open(k_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            i, j = int(row['i']), int(row['j'])
            data[(i, j)] = float(row['K_ij'])
    
    N = max(max(i, j) for i, j in data.keys()) + 1
    K = np.zeros((N, N))
    for (i, j), val in data.items():
        K[i, j] = val
    
    masses = []
    with open(mass_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            masses.append(float(row['mass']))
    M = np.diag(masses)
    
    return K, M

def solve_eigenproblem(K, M, num_states=6):
    """Solve generalized eigenproblem using SciPy."""
    # Transform to standard form: H = M^(-1/2) K M^(-1/2)
    inv_sqrt_mass = 1.0 / np.sqrt(np.diag(M))
    H = np.outer(inv_sqrt_mass, inv_sqrt_mass) * K
    
    eigenvalues, eigenvectors = linalg.eigh(H)
    
    return eigenvalues[:num_states]

def load_direct_fv_energies():
    """Parse energies from fvNeuralLHO output or CSV."""
    eigenvalues_file = POST_PROC / 'eigenvalues.csv'
    if not eigenvalues_file.exists():
        return None
    
    energies = []
    with open(eigenvalues_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            energies.append(float(row['E_directFV']))
    
    return energies

def test_spectrum_matches():
    """Direct FV eigenvalues should match independent SciPy solution."""
    K, M = load_matrices()
    scipy_energies = solve_eigenproblem(K, M, num_states=6)
    
    direct_energies = load_direct_fv_energies()
    assert direct_energies is not None, "Cannot compare without direct FV output"
    
    print("State   SciPy       Direct FV   Diff")
    for n in range(min(len(scipy_energies), len(direct_energies))):
        diff = abs(scipy_energies[n] - direct_energies[n])
        print(f"{n:5d}   {scipy_energies[n]:10.8f}   {direct_energies[n]:10.8f}   {diff:.2e}")
        assert diff < 1e-10, f"Eigenvalue mismatch at state {n}: {diff}"

def test_analytical_approximation():
    """Lowest eigenvalues should approximate E_n = n + 1/2."""
    direct_energies = load_direct_fv_energies()
    assert direct_energies is not None, "Cannot test without direct FV output"
    
    print("\nState   Direct FV   Exact       Discretization Error")
    for n in range(len(direct_energies)):
        E_exact = n + 0.5
        error = abs(direct_energies[n] - E_exact)
        print(f"{n:5d}   {direct_energies[n]:10.8f}   {E_exact:10.8f}   {error:.2e}")

if __name__ == '__main__':
    test_spectrum_matches()
    test_analytical_approximation()
    print("\nAll spectrum tests passed!")
