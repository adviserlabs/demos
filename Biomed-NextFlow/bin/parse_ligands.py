#!/usr/bin/env python3
"""
Parse ligand molecules and extract basic properties
"""

import sys
import json
import csv
import math

def calculate_molecular_weight(formula):
    """Simple molecular weight calculation from formula"""
    # Simplified atomic weights
    atoms = {
        'C': 12.01, 'H': 1.008, 'O': 16.00, 'N': 14.01,
        'S': 32.07, 'P': 30.97, 'Cl': 35.45, 'F': 19.00
    }

    weight = 0
    i = 0
    while i < len(formula):
        if formula[i].isalpha():
            atom = formula[i]
            if i + 1 < len(formula) and formula[i + 1].islower():
                atom += formula[i + 1]
                i += 1

            count = ''
            while i + 1 < len(formula) and formula[i + 1].isdigit():
                count += formula[i + 1]
                i += 1

            count = int(count) if count else 1
            weight += atoms.get(atom, 0) * count
        i += 1

    return round(weight, 2)

def parse_ligands(input_file):
    """Parse ligand CSV file and extract properties"""
    ligands = []

    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ligand_id = row['id']
            name = row['name']
            formula = row['formula']

            mol_weight = calculate_molecular_weight(formula)

            # Calculate LogP (simplified - normally would use cheminformatics tools)
            # For demo: estimate based on C and O counts
            c_count = formula.count('C')
            o_count = formula.count('O')
            logp = round(0.5 * c_count - 0.3 * o_count, 2)

            ligands.append({
                'id': ligand_id,
                'name': name,
                'formula': formula,
                'molecular_weight': mol_weight,
                'logP': logp,
                'num_carbons': c_count,
                'num_oxygens': o_count
            })

    return ligands

def main():
    if len(sys.argv) != 3:
        print("Usage: parse_ligands.py <input_csv> <output_json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Parsing ligands from {input_file}...")
    ligands = parse_ligands(input_file)

    with open(output_file, 'w') as f:
        json.dump(ligands, f, indent=2)

    print(f"✓ Parsed {len(ligands)} ligands")
    print(f"✓ Properties saved to {output_file}")

if __name__ == "__main__":
    main()
