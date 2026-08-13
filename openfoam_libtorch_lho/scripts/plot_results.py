#!/usr/bin/env python3
"""
Plot results: generate eigenfunction plots and error analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
CASE_DIR = ROOT / 'cases' / 'lho_256'
POST_PROC = CASE_DIR / 'postProcessing' / 'fvNeuralLHO'
OUTPUT_DIR = ROOT / 'outputs'

def load_profiles():
    """Load profile data from CSV."""
    profiles_file = POST_PROC / 'profiles.csv'
    
    data = {'x': [], 'volumes': [], 'potential': []}
    for n in range(6):
        data[f'psiNN_{n}'] = []
        data[f'psiDirectFV_{n}'] = []
        data[f'psiExact_{n}'] = []
    
    with open(profiles_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['x'].append(float(row['x']))
            data['volumes'].append(float(row['volume']))
            data['potential'].append(float(row['potential']))
            
            for n in range(6):
                if f'psiNN_{n}' in row:
                    data[f'psiNN_{n}'].append(float(row[f'psiNN_{n}']))
                if f'psiDirectFV_{n}' in row:
                    data[f'psiDirectFV_{n}'].append(float(row[f'psiDirectFV_{n}']))
    
    return {k: np.array(v) for k, v in data.items()}

def hermite_eigenfunction(n, x):
    """Compute analytical harmonic oscillator eigenfunction."""
    def hermite_poly(n, x):
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return 2 * x
        else:
            H_prev2 = np.ones_like(x)
            H_prev1 = 2 * x
            for k in range(1, n):
                H_curr = 2 * x * H_prev1 - 2 * k * H_prev2
                H_prev2 = H_prev1
                H_prev1 = H_curr
            return H_prev1
    
    H_n = hermite_poly(n, x)
    norm_factor = (np.pi ** 0.25) * np.sqrt(2**n * np.math.factorial(n))
    return H_n * np.exp(-x**2 / 2) / norm_factor

def plot_eigenfunctions(data, output_path):
    """Plot eigenfunctions for states 0-5."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    M = np.diag(data['volumes'])
    
    for n in range(6):
        ax = axes[n]
        
        psi_nn = np.array(data[f'psiNN_{n}'])
        psi_direct = np.array(data[f'psiDirectFV_{n}'])
        psi_exact = hermite_eigenfunction(n, data['x'])
        
        # Normalize exact to match discrete M-norm
        exact_norm = np.sqrt(np.dot(psi_exact, M @ psi_exact))
        psi_exact = psi_exact / exact_norm
        
        # Correct sign for overlap
        overlap_nn = np.dot(psi_nn, M @ psi_direct)
        if overlap_nn < 0:
            psi_nn = -psi_nn
        
        ax.plot(data['x'], psi_nn, 'r-', label='Neural FV', alpha=0.7)
        ax.plot(data['x'], psi_direct, 'b--', label='Direct FV', alpha=0.7)
        ax.plot(data['x'], psi_exact, 'g:', label='Analytical', alpha=0.7)
        
        ax.set_title(f'State n={n}')
        ax.set_xlabel('x')
        ax.set_ylabel(r'$\psi_n(x)$')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_errors(data, energies_csv, output_path):
    """Plot error decomposition."""
    with open(energies_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    states = [int(r['state']) for r in rows]
    dNN_Direct = [float(r['dE_NN_direct']) for r in rows]
    dDirect_Exact = [float(r['dE_direct_exact']) for r in rows]
    dNN_Exact = [float(r['dE_NN_exact']) for r in rows]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(states, dNN_Direct, 'ro-', label='Neural optimization error |E_NN - E_direct|')
    ax.loglog(states, dDirect_Exact, 'bs-', label='Discretization error |E_direct - E_exact|')
    ax.loglog(states, dNN_Exact, 'g^-', label='Total error |E_NN - E_exact|')
    
    ax.set_xlabel('State n')
    ax.set_ylabel('Error')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    data = load_profiles()
    
    plot_eigenfunctions(data, OUTPUT_DIR / 'eigenfunctions_n0_n5.png')
    plot_errors(data, POST_PROC / 'eigenvalues.csv', OUTPUT_DIR / 'neural_vs_discretization_error.png')
    
    print("Plots generated successfully")

if __name__ == '__main__':
    main()
