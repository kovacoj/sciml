#!/usr/bin/env python3
"""
Test matrix assembly: verify K is symmetric and matches independent tridiagonal assembly.
"""

import numpy as np
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
CASE_DIR = ROOT / 'cases' / 'lho_test'
POST_PROC = CASE_DIR / 'postProcessing' / 'fvNeuralLHO'

def load_matrix():
    """Load K matrix from CSV if available, otherwise return None."""
    k_file = POST_PROC / 'K.csv'
    if not k_file.exists():
        return None
    
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
    
    return K

def test_matrix_exists():
    """Matrix file should exist for small meshes."""
    k_file = POST_PROC / 'K.csv'
    assert k_file.exists(), "K.csv not found - was writeMatrices true?"

def test_matrix_symmetric():
    """K must be symmetric to machine precision."""
    K = load_matrix()
    assert K is not None, "Cannot test symmetry without matrix"
    
    diff = np.abs(K - K.T).max()
    print(f"Max asymmetry: {diff}")
    assert diff < 1e-12, f"K is not symmetric: max diff = {diff}"

def test_positive_mass():
    """All mass entries must be positive."""
    mass_file = POST_PROC / 'mass.csv'
    if not mass_file.exists():
        pytest.skip("mass.csv not available")
    
    masses = []
    with open(mass_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            masses.append(float(row['mass']))
    
    masses = np.array(masses)
    assert (masses > 0).all(), f"Non-positive mass detected: min = {masses.min()}"
    print(f"Mass range: [{masses.min()}, {masses.max()}]")

def test_finite_values():
    """K must contain only finite values."""
    K = load_matrix()
    assert K is not None, "Cannot test finiteness without matrix"
    
    assert np.all(np.isfinite(K)), "K contains NaN or Inf"
    print(f"K range: [{K.min()}, {K.max()}]")

def test_tridiagonal_structure():
    """For uniform 1D mesh, K must be exactly tridiagonal."""
    K = load_matrix()
    assert K is not None, "Cannot test structure without matrix"

    n = K.shape[0]
    mask = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) > 1
    total_off = np.abs(K * mask).sum()
    print(f"|i - j| > 1 sum: {total_off}")
    assert total_off < 1e-12, f"Significant non-tridiagonal entries: {total_off}"


def test_matches_uniform_grid_assembly():
    """Independently assemble the uniform-grid operator:
    interior  : K_ii = h^2(2*kin/h + V(x_c)); K_{i,i±1} = -kin/h
    boundaries: K_11 += kin/(h/2) for each Dirichlet face
    with kin = kineticScale = 0.5, cell volume h^3 on yz-uniform slice."""
    K = load_matrix()
    assert K is not None, "Cannot test structure without matrix"

    xs, ms = [], []
    with open(POST_PROC / 'x.csv') as f:
        for r in csv.DictReader(f):
            xs.append(float(r['x']))
    with open(POST_PROC / 'mass.csv') as f:
        for r in csv.DictReader(f):
            ms.append(float(r['mass']))
    x, volume = np.asarray(xs), np.asarray(ms)
    n = len(x)

    # Transverse extent is exactly 1 in y and z, so volume = dx*1*1 = dx
    dx = float(x[1] - x[0])
    assert np.allclose(np.diff(x), dx, rtol=1e-12), "non-uniform mesh?"
    assert np.allclose(volume, dx, rtol=1e-12), "volume != dx on unit transverse"
    area = 1.0
    kin = 0.5

    Kindep = np.zeros((n, n))
    for i in range(n - 1):
        g = kin * area / dx
        Kindep[i, i] += g
        Kindep[i + 1, i + 1] += g
        Kindep[i, i + 1] -= g
        Kindep[i + 1, i] -= g
    # Dirichlet boundary faces on left/right: cell-centre distance dx/2
    gb = kin * area / (dx / 2.0)
    Kindep[0, 0] += gb
    Kindep[-1, -1] += gb
    Kindep += np.diag(0.5 * x**2 * volume)    # potentialScale x_c^2 V_c

    diff = np.abs(K - Kindep).max()
    print(f"max |K_cpp - K_independent| = {diff:.3e}")
    assert diff < 1e-10, f"C++ matrix differs from independent assembly: {diff}"

if __name__ == '__main__':
    test_matrix_exists()
    test_matrix_symmetric()
    test_positive_mass()
    test_finite_values()
    test_tridiagonal_structure()
    print("All matrix tests passed!")
