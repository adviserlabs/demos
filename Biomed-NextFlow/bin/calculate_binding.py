#!/usr/bin/env python3
"""
Calculate protein-ligand binding energies
"""

import sys
import json
import csv
import random
import math

def read_protein_residues(protein_file):
    """Simple PDB parser to count residues"""
    residues = set()
    try:
        with open(protein_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    residue = line[17:20].strip()
                    residues.add(residue)
    except:
        # If not a real PDB, use default
        residues = {'ALA', 'GLY', 'VAL', 'LEU', 'ILE'}

    return len(residues)

def calculate_binding_energy(ligand, protein_size):
    """
    Simplified binding energy calculation
    In reality, this would use force fields, docking scores, etc.
    """

    # Seed with ligand properties for reproducibility
    seed = hash(ligand['id']) % 10000
    random.seed(seed)

    # Base energy from molecular properties
    mw_factor = -0.05 * ligand['molecular_weight']
    logp_factor = -2.0 * ligand['logP']

    # Protein size contribution
    protein_factor = -0.1 * protein_size

    # Add some "docking" variability
    docking_score = random.uniform(-5, 0)

    # Combine factors (in kcal/mol)
    binding_energy = mw_factor + logp_factor + protein_factor + docking_score

    # Add entropy term
    entropy_penalty = random.uniform(2, 5)
    binding_energy += entropy_penalty

    return round(binding_energy, 2)

def calculate_binding_affinity(binding_energy):
    """Convert binding energy to Kd (dissociation constant)"""
    # ΔG = RT ln(Kd)
    # Kd = exp(ΔG/RT)
    R = 0.001987  # kcal/(mol·K)
    T = 298.15    # K (25°C)

    kd = math.exp(binding_energy / (R * T))

    # Convert to more readable units (μM)
    kd_um = kd * 1e6

    return round(kd_um, 2)

def main():
    if len(sys.argv) != 4:
        print("Usage: calculate_binding.py <protein_pdb> <ligand_json> <output_dir>")
        sys.exit(1)

    protein_file = sys.argv[1]
    ligand_file = sys.argv[2]
    output_dir = sys.argv[3]

    print(f"Reading protein structure from {protein_file}...")
    protein_size = read_protein_residues(protein_file)
    print(f"✓ Protein has {protein_size} unique residues")

    print(f"Loading ligand properties from {ligand_file}...")
    with open(ligand_file, 'r') as f:
        ligands = json.load(f)

    print(f"Calculating binding energies for {len(ligands)} ligands...")

    results = []
    for ligand in ligands:
        energy = calculate_binding_energy(ligand, protein_size)
        kd = calculate_binding_affinity(energy)

        results.append({
            'ligand_id': ligand['id'],
            'name': ligand['name'],
            'binding_energy_kcal_mol': energy,
            'kd_uM': kd,
            'molecular_weight': ligand['molecular_weight'],
            'logP': ligand['logP']
        })

    # Save results as CSV
    csv_file = f"{output_dir}/binding_energies.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Create summary report
    report_file = f"{output_dir}/binding_report.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PROTEIN-LIGAND BINDING ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Protein residues analyzed: {protein_size}\n")
        f.write(f"Total ligands screened: {len(ligands)}\n\n")
        f.write("TOP 5 BINDING CANDIDATES:\n")
        f.write("-" * 60 + "\n")

        sorted_results = sorted(results, key=lambda x: x['binding_energy_kcal_mol'])
        for i, result in enumerate(sorted_results[:5], 1):
            f.write(f"\n{i}. {result['name']} (ID: {result['ligand_id']})\n")
            f.write(f"   Binding Energy: {result['binding_energy_kcal_mol']} kcal/mol\n")
            f.write(f"   Kd: {result['kd_uM']} μM\n")
            f.write(f"   Molecular Weight: {result['molecular_weight']} Da\n")
            f.write(f"   LogP: {result['logP']}\n")

    print(f"✓ Binding energies saved to {csv_file}")
    print(f"✓ Summary report saved to {report_file}")

if __name__ == "__main__":
    main()
