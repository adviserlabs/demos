#!/usr/bin/env python3
"""
Biomolecular Dynamics Simulation: Small Molecule-Protein Binding
Mock simulation generating fake data and ASCII visualizations
"""

import random
import time
import sys
import os

class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class MolecularDynamicsSimulator:
    def __init__(self):
        self.proteins = ['ACE2', 'EGFR', 'p53', 'BRCA1', 'TNF-α', 'HER2', 'VEGFR', 'PDGFR', 'IGF1R', 'FGFR', 'MET', 'ALK', 'ROS1', 'BRAF', 'PIK3CA']
        self.ligands = ['Aspirin', 'Ibuprofen', 'Caffeine', 'Morphine', 'Dopamine', 'Serotonin', 'Acetaminophen', 'Naproxen', 'Codeine', 'Theobromine', 'Epinephrine', 'Norepinephrine']
        self.simulation_data = {}
        self.time_steps = 100
        self.binding_threshold = 0.7
        self.tsv_filename = 'biomedMolecularDynamicsSim.tsv'

    def generate_binding_affinity(self, protein, ligand):
        """Generate mock binding affinity based on protein-ligand combination"""
        random.seed(hash(protein + ligand) % 1000)
        base_affinity = random.uniform(0.1, 0.95)
        return round(base_affinity, 3)

    def generate_kinetic_data(self, binding_affinity):
        """Generate binding kinetics over time"""
        kinetics = []
        for t in range(self.time_steps):
            noise = random.uniform(-0.05, 0.05)
            if t < 20:
                # Initial binding phase
                value = binding_affinity * (t / 20) + noise
            elif t < 80:
                # Equilibrium phase
                equilibrium_noise = random.uniform(-0.02, 0.02)
                value = binding_affinity + equilibrium_noise + noise
            else:
                # Dissociation phase
                decay_factor = 1 - ((t - 80) / 20) * 0.3
                value = binding_affinity * decay_factor + noise

            kinetics.append(max(0, min(1, value)))
        return kinetics

    def generate_synthetic_data(self):
        """Generate synthetic molecular dynamics data"""
        print(f"{Color.BOLD}{Color.CYAN}=== Generating Synthetic Data ==={Color.RESET}")
        print(f"{Color.YELLOW}Creating mock binding data for proteins and ligands...{Color.RESET}")
        print()

        for protein in self.proteins:
            self.simulation_data[protein] = {}
            for ligand in self.ligands:
                affinity = self.generate_binding_affinity(protein, ligand)
                kinetics = self.generate_kinetic_data(affinity)
                self.simulation_data[protein][ligand] = {
                    'binding_affinity': affinity,
                    'kinetics': kinetics,
                    'max_binding': max(kinetics),
                    'avg_binding': sum(kinetics) / len(kinetics)
                }

        print(f"{Color.GREEN}✓ Data generation complete! Generated data for {len(self.proteins)} proteins and {len(self.ligands)} ligands{Color.RESET}")
        print()

    def export_to_tsv(self):
        """Export simulation data to TSV file"""
        print(f"{Color.BOLD}{Color.BLUE}=== Exporting Data to TSV ==={Color.RESET}")

        with open(self.tsv_filename, 'w') as f:
            # Write header
            header = ['Protein', 'Ligand', 'Binding_Affinity', 'Max_Binding', 'Avg_Binding']
            kinetic_headers = [f'Time_{i}' for i in range(self.time_steps)]
            f.write('\t'.join(header + kinetic_headers) + '\n')

            # Write data
            for protein in self.proteins:
                for ligand in self.ligands:
                    data = self.simulation_data[protein][ligand]
                    row = [
                        protein,
                        ligand,
                        str(data['binding_affinity']),
                        str(round(data['max_binding'], 3)),
                        str(round(data['avg_binding'], 3))
                    ]
                    kinetic_values = [str(round(val, 3)) for val in data['kinetics']]
                    f.write('\t'.join(row + kinetic_values) + '\n')

        total_rows = len(self.proteins) * len(self.ligands)
        print(f"{Color.GREEN}✓ Exported {total_rows} rows to {self.tsv_filename}{Color.RESET}")
        print()

    def import_from_tsv(self):
        """Import simulation data from TSV file"""
        print(f"{Color.BOLD}{Color.BLUE}=== Importing Data from TSV ==={Color.RESET}")

        if not os.path.exists(self.tsv_filename):
            print(f"{Color.RED}Error: {self.tsv_filename} not found!{Color.RESET}")
            return False

        self.simulation_data = {}
        imported_count = 0

        with open(self.tsv_filename, 'r') as f:
            lines = f.readlines()
            header = lines[0].strip().split('\t')

            # Find kinetic data columns (Time_0, Time_1, etc.)
            kinetic_start_idx = 5  # After Protein, Ligand, Binding_Affinity, Max_Binding, Avg_Binding

            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) < kinetic_start_idx:
                    continue

                protein = parts[0]
                ligand = parts[1]
                binding_affinity = float(parts[2])
                max_binding = float(parts[3])
                avg_binding = float(parts[4])
                kinetics = [float(val) for val in parts[kinetic_start_idx:]]

                if protein not in self.simulation_data:
                    self.simulation_data[protein] = {}

                self.simulation_data[protein][ligand] = {
                    'binding_affinity': binding_affinity,
                    'kinetics': kinetics,
                    'max_binding': max_binding,
                    'avg_binding': avg_binding
                }
                imported_count += 1

        print(f"{Color.GREEN}✓ Imported {imported_count} protein-ligand pairs from {self.tsv_filename}{Color.RESET}")
        print()
        return True

    def display_binding_affinity_matrix(self):
        """Display binding affinity as a colored ASCII matrix"""
        print(f"{Color.BOLD}{Color.BLUE}=== Binding Affinity Matrix ==={Color.RESET}")
        print(f"{Color.WHITE}Values: 0.0 (no binding) → 1.0 (strong binding){Color.RESET}")
        print()

        # Header
        print(f"{Color.BOLD}{'Protein':<10}{Color.RESET}", end="")
        for ligand in self.ligands:
            print(f"{Color.BOLD}{ligand:<14}{Color.RESET}", end="")
        print()

        # Data rows
        for protein in self.proteins:
            print(f"{Color.CYAN}{protein:<10}{Color.RESET}", end="")
            for ligand in self.ligands:
                affinity = self.simulation_data[protein][ligand]['binding_affinity']
                color = self._get_affinity_color(affinity)
                print(f"{color}{affinity:<14.3f}{Color.RESET}", end="")
            print()
        print()

    def _get_affinity_color(self, affinity):
        """Get color based on binding affinity strength"""
        if affinity >= 0.8:
            return Color.RED + Color.BOLD
        elif affinity >= 0.6:
            return Color.YELLOW
        elif affinity >= 0.4:
            return Color.GREEN
        else:
            return Color.WHITE

    def plot_kinetics_chart(self, protein, ligand):
        """Generate ASCII time-series plot for binding kinetics"""
        data = self.simulation_data[protein][ligand]['kinetics']
        affinity = self.simulation_data[protein][ligand]['binding_affinity']

        print(f"{Color.BOLD}{Color.MAGENTA}=== Binding Kinetics: {protein} + {ligand} ==={Color.RESET}")
        print(f"{Color.WHITE}Initial Binding Affinity: {Color.YELLOW}{affinity:.3f}{Color.RESET}")
        print()

        # Chart parameters
        height = 15
        width = 80

        # Scale data to chart dimensions
        max_val = max(data)
        min_val = min(data)
        range_val = max_val - min_val if max_val != min_val else 1

        # Create chart grid
        chart = [[' ' for _ in range(width)] for _ in range(height)]

        # Plot data points
        for i, value in enumerate(data):
            if i >= width:
                break

            normalized = (value - min_val) / range_val
            y_pos = int((1 - normalized) * (height - 1))

            if 0 <= y_pos < height:
                if value >= self.binding_threshold:
                    chart[y_pos][i] = 'O'
                elif value >= 0.5:
                    chart[y_pos][i] = 'o'
                elif value >= 0.3:
                    chart[y_pos][i] = '+'
                else:
                    chart[y_pos][i] = '-'

        # Display chart
        print(f"{Color.WHITE}1.0 |{Color.RESET}", end="")
        for row in chart:
            if row == chart[0]:
                for i, cell in enumerate(row):
                    if cell != ' ':
                        color = Color.RED if data[i] >= 0.8 else Color.YELLOW if data[i] >= 0.6 else Color.GREEN
                        print(f"{color}{cell}{Color.RESET}", end="")
                    else:
                        print(cell, end="")
                print()
            else:
                print(f"{Color.WHITE}    |{Color.RESET}", end="")
                for i, cell in enumerate(row):
                    if cell != ' ':
                        color = Color.RED if data[i] >= 0.8 else Color.YELLOW if data[i] >= 0.6 else Color.GREEN
                        print(f"{color}{cell}{Color.RESET}", end="")
                    else:
                        print(cell, end="")
                print()

        print(f"{Color.WHITE}0.0 |{'_' * width}{Color.RESET}")
        print(f"{Color.WHITE}    0{'':>37}Time (steps){'':>37}100{Color.RESET}")
        print()

        # Add legend
        print(f"{Color.WHITE}Legend: {Color.RED}O{Color.RESET} High (≥0.8)  {Color.YELLOW}o{Color.RESET} Medium (≥0.6)  {Color.GREEN}+{Color.RESET} Low (≥0.3)  {Color.WHITE}-{Color.RESET} Minimal")
        print()

    def generate_summary_statistics(self):
        """Generate and display summary statistics"""
        print(f"{Color.BOLD}{Color.CYAN}=== Simulation Summary Statistics ==={Color.RESET}")
        print()

        all_affinities = []
        strong_binders = []

        for protein in self.proteins:
            for ligand in self.ligands:
                affinity = self.simulation_data[protein][ligand]['binding_affinity']
                all_affinities.append(affinity)
                if affinity >= self.binding_threshold:
                    strong_binders.append((protein, ligand, affinity))

        # Overall statistics
        avg_affinity = sum(all_affinities) / len(all_affinities)
        max_affinity = max(all_affinities)
        min_affinity = min(all_affinities)

        print(f"{Color.WHITE}Total protein-ligand pairs: {Color.YELLOW}{len(all_affinities)}{Color.RESET}")
        print(f"{Color.WHITE}Average binding affinity: {Color.YELLOW}{avg_affinity:.3f}{Color.RESET}")
        print(f"{Color.WHITE}Maximum binding affinity: {Color.GREEN}{max_affinity:.3f}{Color.RESET}")
        print(f"{Color.WHITE}Minimum binding affinity: {Color.RED}{min_affinity:.3f}{Color.RESET}")
        print(f"{Color.WHITE}Strong binders (≥{self.binding_threshold}): {Color.GREEN}{len(strong_binders)}{Color.RESET}")
        print()

        # Top binding pairs
        if strong_binders:
            strong_binders.sort(key=lambda x: x[2], reverse=True)
            print(f"{Color.BOLD}{Color.GREEN}Top 5 Binding Pairs:{Color.RESET}")
            for i, (protein, ligand, affinity) in enumerate(strong_binders[:5], 1):
                print(f"{Color.WHITE}{i}. {Color.CYAN}{protein}{Color.WHITE} + {Color.MAGENTA}{ligand}{Color.WHITE}: {Color.YELLOW}{affinity:.3f}{Color.RESET}")
        print()

    def create_binding_histogram(self):
        """Create ASCII histogram of binding affinities"""
        print(f"{Color.BOLD}{Color.BLUE}=== Binding Affinity Distribution ==={Color.RESET}")
        print()

        all_affinities = []
        for protein in self.proteins:
            for ligand in self.ligands:
                all_affinities.append(self.simulation_data[protein][ligand]['binding_affinity'])

        # Create bins
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_counts = [0] * (len(bins) - 1)
        bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

        for affinity in all_affinities:
            for i in range(len(bins) - 1):
                if bins[i] <= affinity < bins[i + 1] or (i == len(bins) - 2 and affinity == bins[i + 1]):
                    bin_counts[i] += 1
                    break

        # Display histogram
        max_count = max(bin_counts) if bin_counts else 1
        scale = 40 / max_count if max_count > 0 else 1

        for i, (label, count) in enumerate(zip(bin_labels, bin_counts)):
            bar_length = int(count * scale)
            color = [Color.RED, Color.YELLOW, Color.GREEN, Color.CYAN, Color.MAGENTA][i]
            bar = f"{color}{'-' * bar_length}{Color.RESET}"
            print(f"{Color.WHITE}{label:<8} |{bar} {count}{Color.RESET}")

        print()

def main():
    simulator = MolecularDynamicsSimulator()

    # Check command line arguments for mode
    mode = 'both'  # default
    if len(sys.argv) > 1:
        if sys.argv[1] == 'generate':
            mode = 'generate'
        elif sys.argv[1] == 'simulate':
            mode = 'simulate'
        elif sys.argv[1] == 'both':
            mode = 'both'
        else:
            print(f"{Color.RED}Usage: python {sys.argv[0]} [generate|simulate|both]{Color.RESET}")
            print(f"{Color.WHITE}  generate - Generate synthetic data and save to TSV{Color.RESET}")
            print(f"{Color.WHITE}  simulate - Load TSV data and run visualization{Color.RESET}")
            print(f"{Color.WHITE}  both     - Generate data, save to TSV, then run simulation (default){Color.RESET}")
            return

    # Phase 1: Data Generation
    if mode in ['generate', 'both']:
        simulator.generate_synthetic_data()
        simulator.export_to_tsv()

        if mode == 'generate':
            print(f"{Color.BOLD}{Color.GREEN}=== Data Generation Complete ==={Color.RESET}")
            print(f"{Color.WHITE}Run 'python {sys.argv[0]} simulate' to visualize the data{Color.RESET}")
            return

    # Phase 2: Simulation and Visualization
    if mode in ['simulate', 'both']:
        if mode == 'simulate':
            # Load data from TSV
            if not simulator.import_from_tsv():
                return

        print(f"{Color.BOLD}{Color.CYAN}=== Running Biomolecular Simulation ==={Color.RESET}")
        print(f"{Color.YELLOW}Analyzing molecular binding dynamics...{Color.RESET}")
        print()

        # Display results
        simulator.display_binding_affinity_matrix()
        simulator.generate_summary_statistics()
        simulator.create_binding_histogram()

        # Show detailed kinetics for a few interesting pairs
        print(f"{Color.BOLD}{Color.CYAN}=== Detailed Kinetics Analysis ==={Color.RESET}")
        print()

        # Find some interesting pairs to display
        interesting_pairs = []
        proteins_list = list(simulator.simulation_data.keys())
        for protein in proteins_list[:2]:  # Show first 2 proteins
            ligands_list = list(simulator.simulation_data[protein].keys())
            for ligand in ligands_list[:2]:  # Show first 2 ligands
                interesting_pairs.append((protein, ligand))

        for protein, ligand in interesting_pairs:
            simulator.plot_kinetics_chart(protein, ligand)
            time.sleep(0.5)  # Small delay for better readability

        print(f"{Color.BOLD}{Color.GREEN}=== Simulation Complete ==={Color.RESET}")
        print(f"{Color.WHITE}Simulation to demonstrate biomolecular binding affinity dynamics.{Color.RESET}")
        if mode == 'both':
            print(f"{Color.WHITE}Data saved to {simulator.tsv_filename} for future analysis.{Color.RESET}")
        else:
            print(f"{Color.WHITE}Data loaded from {simulator.tsv_filename} for analysis.{Color.RESET}")

if __name__ == "__main__":
    main()
