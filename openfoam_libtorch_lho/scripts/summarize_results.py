#!/usr/bin/env python3
"""
Summarize results: generate CSV summary of all runs.
"""

import argparse
import csv
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Summarize LHO results')
    parser.add_argument('--case', type=str, required=True, help='Case directory')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    args = parser.parse_args()

    case_dir = Path(args.case)
    post_proc = case_dir / 'postProcessing' / 'fvNeuralLHO'
    
    eigenvalues_file = post_proc / 'eigenvalues.csv'
    
    if not eigenvalues_file.exists():
        print(f"ERROR: {eigenvalues_file} not found")
        return
    
    rows = []
    with open(eigenvalues_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['state', 'E_NN', 'E_directFV', 'E_exact', 
                        'dE_NN_direct', 'dE_direct_exact', 'dE_NN_exact',
                        'rel_neural_error', 'rel_discretization_error'])
        
        for row in rows:
            E_NN = float(row['E_NN'])
            E_direct = float(row['E_directFV'])
            E_exact = float(row['E_exact'])
            
            rel_neural = abs(E_NN - E_direct) / E_exact
            rel_disc = abs(E_direct - E_exact) / E_exact
            
            writer.writerow([
                row['state'],
                f"{E_NN:.10f}",
                f"{E_direct:.10f}",
                f"{E_exact:.10f}",
                f"{float(row['dE_NN_direct']):.2e}",
                f"{float(row['dE_direct_exact']):.2e}",
                f"{float(row['dE_NN_exact']):.2e}",
                f"{rel_neural:.2e}",
                f"{rel_disc:.2e}"
            ])
    
    print(f"Summary written to {output_path}")
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"{'State':<8} {'E_NN':<15} {'E_direct':<15} {'Error (NN-direct)':<18}")
    for row in rows[:6]:
        print(f"{row['state']:<8} {float(row['E_NN']):<15.10f} {float(row['E_directFV']):<15.10f} {float(row['dE_NN_direct']):<18.2e}")

if __name__ == '__main__':
    main()
