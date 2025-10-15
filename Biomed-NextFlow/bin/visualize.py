#!/usr/bin/env python3
"""
Visualize binding energy results
"""

import sys
import csv

def create_ascii_plot(results, max_width=60):
    """Create ASCII bar chart of binding energies"""

    # Sort by binding energy (most negative = strongest binding)
    sorted_results = sorted(results, key=lambda x: float(x['binding_energy_kcal_mol']))
    top_10 = sorted_results[:10]

    # Find the range for scaling
    min_energy = min(float(r['binding_energy_kcal_mol']) for r in top_10)
    max_energy = max(float(r['binding_energy_kcal_mol']) for r in top_10)
    energy_range = max_energy - min_energy

    plot_lines = []
    plot_lines.append("=" * 70)
    plot_lines.append("BINDING ENERGY VISUALIZATION (Top 10 Candidates)")
    plot_lines.append("=" * 70)
    plot_lines.append("")
    plot_lines.append("Lower (more negative) = Stronger binding")
    plot_lines.append("")

    for i, result in enumerate(top_10, 1):
        energy = float(result['binding_energy_kcal_mol'])
        name = result['name'][:20]  # Truncate long names

        # Scale to bar width
        if energy_range > 0:
            bar_length = int(((energy - min_energy) / energy_range) * max_width)
        else:
            bar_length = max_width // 2

        bar = '█' * max(1, bar_length)

        plot_lines.append(f"{i:2d}. {name:20s} | {bar} {energy:.2f} kcal/mol")

    plot_lines.append("")
    plot_lines.append("=" * 70)

    return "\n".join(plot_lines)

def main():
    if len(sys.argv) != 3:
        print("Usage: visualize.py <binding_csv> <output_dir>")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_dir = sys.argv[2]

    print(f"Loading binding energy data from {csv_file}...")

    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        results = list(reader)

    print(f"✓ Loaded {len(results)} binding results")

    # Create ASCII plot
    plot = create_ascii_plot(results)

    # Save plot as "image"
    plot_file = f"{output_dir}/binding_plot.png"
    with open(plot_file, 'w') as f:
        f.write(plot)

    # Create top candidates summary
    sorted_results = sorted(results, key=lambda x: float(x['binding_energy_kcal_mol']))

    candidates_file = f"{output_dir}/top_candidates.txt"
    with open(candidates_file, 'w') as f:
        f.write("TOP 3 BINDING CANDIDATES\n")
        f.write("=" * 50 + "\n\n")

        for i, result in enumerate(sorted_results[:3], 1):
            f.write(f"Rank {i}: {result['name']}\n")
            f.write(f"  • Binding Energy: {result['binding_energy_kcal_mol']} kcal/mol\n")
            f.write(f"  • Dissociation Constant (Kd): {result['kd_uM']} μM\n")
            f.write(f"  • Molecular Weight: {result['molecular_weight']} Da\n")
            f.write(f"  • LogP: {result['logP']}\n")
            f.write("\n")

    print(f"✓ Visualization saved to {plot_file}")
    print(f"✓ Top candidates saved to {candidates_file}")
    print("\nVisualization Preview:")
    print(plot)

if __name__ == "__main__":
    main()
