#!/usr/bin/env python3
"""
Density Functional Theory (DFT) Demo
Simulates electron structure calculations for materials science applications.
Computes band structures, band gaps, and energy levels for semiconductor/catalyst design.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from datetime import datetime

# Ensure output directory exists
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

class DFTSimulator:
    """Simplified DFT simulator using tight-binding approximation"""

    def __init__(self, material_name="Silicon", lattice_constant=5.43):
        self.material_name = material_name
        self.lattice_constant = lattice_constant  # Angstroms
        self.k_points = None
        self.energy_bands = None
        self.band_gap = None
        self.fermi_energy = None

    def generate_k_path(self, n_points=100):
        """Generate k-points along high-symmetry path in Brillouin zone"""
        # Γ → X → W → L → Γ → K (typical fcc path)
        gamma = np.array([0.0, 0.0, 0.0])
        x_point = np.array([0.5, 0.0, 0.5])
        w_point = np.array([0.5, 0.25, 0.75])
        l_point = np.array([0.5, 0.5, 0.5])
        k_point = np.array([0.375, 0.375, 0.75])

        # Create path segments
        segments = [
            (gamma, x_point, "Γ-X"),
            (x_point, w_point, "X-W"),
            (w_point, l_point, "W-L"),
            (l_point, gamma, "L-Γ"),
            (gamma, k_point, "Γ-K")
        ]

        k_path = []
        labels = []
        positions = [0]

        for start, end, label in segments:
            segment = np.linspace(start, end, n_points // len(segments))
            k_path.extend(segment)
            if len(labels) == 0:
                labels.append((0, label.split('-')[0]))
            labels.append((len(k_path) - 1, label.split('-')[1]))
            positions.append(len(k_path) - 1)

        self.k_points = np.array(k_path)
        self.k_labels = labels
        self.k_positions = positions
        return self.k_points

    def calculate_band_structure(self, n_bands=8):
        """
        Calculate electronic band structure using simplified tight-binding model
        Simulates valence and conduction bands
        """
        print(f"Computing band structure for {self.material_name}...")
        n_k = len(self.k_points)
        bands = np.zeros((n_k, n_bands))

        # Tight-binding parameters (simplified)
        t = -2.5  # Hopping parameter (eV)
        epsilon = 0.0  # On-site energy (eV)

        for i, k in enumerate(self.k_points):
            kx, ky, kz = k * 2 * np.pi / self.lattice_constant

            # Simplified dispersion relations for different bands
            for band_idx in range(n_bands):
                if band_idx < n_bands // 2:  # Valence bands
                    offset = -(n_bands // 2 - band_idx) * 1.5
                    bands[i, band_idx] = epsilon + 2 * t * (
                        np.cos(kx) + np.cos(ky) + np.cos(kz)
                    ) + offset - 3.0
                else:  # Conduction bands
                    offset = (band_idx - n_bands // 2 + 1) * 1.2
                    bands[i, band_idx] = epsilon - 2 * t * (
                        np.cos(kx) + np.cos(ky) + np.cos(kz)
                    ) + offset + 4.5

            # Add some realistic noise
            bands[i] += np.random.normal(0, 0.05, n_bands)

        self.energy_bands = bands

        # Calculate band gap (between highest valence and lowest conduction)
        valence_max = np.max(bands[:, :n_bands//2])
        conduction_min = np.min(bands[:, n_bands//2:])
        self.band_gap = conduction_min - valence_max
        self.fermi_energy = (valence_max + conduction_min) / 2

        print(f"Band gap: {self.band_gap:.3f} eV")
        print(f"Fermi energy: {self.fermi_energy:.3f} eV")

        return bands

    def calculate_density_of_states(self, energy_range=(-10, 10), n_points=500):
        """Calculate density of states (DOS)"""
        print("Computing density of states...")

        energies = np.linspace(energy_range[0], energy_range[1], n_points)
        dos = np.zeros(n_points)

        # Gaussian broadening
        sigma = 0.1  # eV

        for band in self.energy_bands.T:
            for E_k in band:
                dos += np.exp(-(energies - E_k)**2 / (2 * sigma**2))

        dos = dos / (sigma * np.sqrt(2 * np.pi))

        self.dos_energies = energies
        self.dos = dos

        return energies, dos

    def plot_band_structure(self):
        """Create band structure diagram"""
        fig, ax = plt.subplots(figsize=(10, 7))

        x = np.arange(len(self.k_points))

        # Plot all bands
        n_bands = self.energy_bands.shape[1]
        for i in range(n_bands):
            color = 'blue' if i < n_bands // 2 else 'red'
            alpha = 0.7
            ax.plot(x, self.energy_bands[:, i], color=color, linewidth=2, alpha=alpha)

        # Add Fermi level
        ax.axhline(y=self.fermi_energy, color='green', linestyle='--',
                   linewidth=2, label=f'Fermi Level ({self.fermi_energy:.2f} eV)')

        # Add vertical lines at high-symmetry points
        for pos in self.k_positions[1:-1]:
            ax.axvline(x=pos, color='gray', linestyle='-', linewidth=0.5)

        # Set x-axis labels
        unique_labels = []
        unique_positions = []
        for pos, label in self.k_labels:
            if pos not in unique_positions:
                unique_positions.append(pos)
                unique_labels.append(label)

        ax.set_xticks(unique_positions)
        ax.set_xticklabels(unique_labels, fontsize=14)

        ax.set_ylabel('Energy (eV)', fontsize=14)
        ax.set_xlabel('Wave Vector k', fontsize=14)
        ax.set_title(f'Electronic Band Structure - {self.material_name}\n'
                     f'Band Gap: {self.band_gap:.3f} eV', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = OUTPUT_DIR / 'band_structure.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved band structure diagram to {filename}")
        plt.close()

    def plot_density_of_states(self):
        """Create density of states plot"""
        fig, ax = plt.subplots(figsize=(8, 7))

        ax.fill_between(self.dos_energies, self.dos, alpha=0.5, color='purple')
        ax.plot(self.dos_energies, self.dos, color='darkviolet', linewidth=2)

        # Add Fermi level
        ax.axvline(x=self.fermi_energy, color='green', linestyle='--',
                   linewidth=2, label=f'Fermi Level ({self.fermi_energy:.2f} eV)')

        # Shade band gap region
        valence_max = np.max(self.energy_bands[:, :self.energy_bands.shape[1]//2])
        conduction_min = np.min(self.energy_bands[:, self.energy_bands.shape[1]//2:])
        ax.axvspan(valence_max, conduction_min, alpha=0.2, color='yellow',
                   label=f'Band Gap ({self.band_gap:.3f} eV)')

        ax.set_xlabel('Energy (eV)', fontsize=14)
        ax.set_ylabel('Density of States (states/eV)', fontsize=14)
        ax.set_title(f'Density of States - {self.material_name}', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = OUTPUT_DIR / 'density_of_states.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved density of states plot to {filename}")
        plt.close()

    def plot_energy_levels(self):
        """Create energy levels diagram"""
        fig, ax = plt.subplots(figsize=(8, 10))

        n_bands = self.energy_bands.shape[1]

        # Calculate average energy for each band
        band_energies = []
        for i in range(n_bands):
            min_e = np.min(self.energy_bands[:, i])
            max_e = np.max(self.energy_bands[:, i])
            avg_e = np.mean(self.energy_bands[:, i])
            band_energies.append((i, min_e, max_e, avg_e))

        # Plot energy levels
        for idx, min_e, max_e, avg_e in band_energies:
            color = 'blue' if idx < n_bands // 2 else 'red'
            label = f'Valence {idx + 1}' if idx < n_bands // 2 else f'Conduction {idx - n_bands // 2 + 1}'

            # Draw horizontal lines for min, max, avg
            ax.plot([0.3, 0.7], [avg_e, avg_e], color=color, linewidth=3, label=label)
            ax.plot([0.35, 0.65], [min_e, min_e], color=color, linewidth=1, linestyle=':', alpha=0.5)
            ax.plot([0.35, 0.65], [max_e, max_e], color=color, linewidth=1, linestyle=':', alpha=0.5)

            # Add text labels
            ax.text(0.75, avg_e, f'{avg_e:.2f} eV', verticalalignment='center', fontsize=10)

        # Add Fermi level
        ax.axhline(y=self.fermi_energy, color='green', linestyle='--',
                   linewidth=2, label=f'Fermi Level')

        # Add band gap annotation
        valence_max = band_energies[n_bands//2 - 1][2]
        conduction_min = band_energies[n_bands//2][1]

        ax.annotate('', xy=(0.85, valence_max), xytext=(0.85, conduction_min),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=2))
        ax.text(0.88, (valence_max + conduction_min) / 2,
                f'Band Gap\n{self.band_gap:.3f} eV',
                verticalalignment='center', fontsize=11, fontweight='bold')

        ax.set_xlim(0, 1)
        ax.set_ylabel('Energy (eV)', fontsize=14)
        ax.set_title(f'Energy Levels - {self.material_name}', fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = OUTPUT_DIR / 'energy_levels.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved energy levels diagram to {filename}")
        plt.close()

    def save_results(self):
        """Save numerical results to files"""
        # Save band structure data
        band_file = OUTPUT_DIR / 'band_structure_data.csv'
        header = ['k_index'] + [f'band_{i+1}' for i in range(self.energy_bands.shape[1])]
        data = np.column_stack([np.arange(len(self.k_points)), self.energy_bands])
        np.savetxt(band_file, data, delimiter=',', header=','.join(header), comments='')
        print(f"Saved band structure data to {band_file}")

        # Save DOS data
        dos_file = OUTPUT_DIR / 'density_of_states_data.csv'
        dos_data = np.column_stack([self.dos_energies, self.dos])
        np.savetxt(dos_file, dos_data, delimiter=',',
                   header='energy_eV,dos_states_per_eV', comments='')
        print(f"Saved DOS data to {dos_file}")

        # Save summary results
        summary = {
            'material': self.material_name,
            'lattice_constant_angstrom': self.lattice_constant,
            'band_gap_eV': float(self.band_gap),
            'fermi_energy_eV': float(self.fermi_energy),
            'n_bands': int(self.energy_bands.shape[1]),
            'n_k_points': int(len(self.k_points)),
            'valence_band_maximum_eV': float(np.max(self.energy_bands[:, :self.energy_bands.shape[1]//2])),
            'conduction_band_minimum_eV': float(np.min(self.energy_bands[:, self.energy_bands.shape[1]//2:])),
            'calculation_date': datetime.now().isoformat()
        }

        summary_file = OUTPUT_DIR / 'dft_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_file}")

        # Create detailed text report
        report_file = OUTPUT_DIR / 'dft_report.txt'
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("DENSITY FUNCTIONAL THEORY (DFT) CALCULATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Material: {self.material_name}\n")
            f.write(f"Lattice Constant: {self.lattice_constant:.2f} Å\n")
            f.write(f"Calculation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("-" * 70 + "\n")
            f.write("ELECTRONIC STRUCTURE PROPERTIES\n")
            f.write("-" * 70 + "\n")
            f.write(f"Band Gap:                    {self.band_gap:.4f} eV\n")
            f.write(f"Fermi Energy:                {self.fermi_energy:.4f} eV\n")
            f.write(f"Valence Band Maximum:        {summary['valence_band_maximum_eV']:.4f} eV\n")
            f.write(f"Conduction Band Minimum:     {summary['conduction_band_minimum_eV']:.4f} eV\n\n")
            f.write("-" * 70 + "\n")
            f.write("CALCULATION DETAILS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Number of Energy Bands:      {self.energy_bands.shape[1]}\n")
            f.write(f"Number of k-points:          {len(self.k_points)}\n")
            f.write(f"k-path: Γ → X → W → L → Γ → K\n\n")
            f.write("-" * 70 + "\n")
            f.write("MATERIAL CLASSIFICATION\n")
            f.write("-" * 70 + "\n")
            if self.band_gap < 0.1:
                material_type = "Metal (no band gap)"
            elif self.band_gap < 2.0:
                material_type = "Semiconductor (small band gap)"
            else:
                material_type = "Insulator (large band gap)"
            f.write(f"Classification: {material_type}\n\n")
            f.write("-" * 70 + "\n")
            f.write("APPLICATIONS\n")
            f.write("-" * 70 + "\n")
            f.write("• Semiconductor device design\n")
            f.write("• Catalyst optimization\n")
            f.write("• Solar cell materials engineering\n")
            f.write("• Optoelectronic device development\n\n")
            f.write("=" * 70 + "\n")

        print(f"Saved detailed report to {report_file}")


def simulate_parallel_computation():
    """Simulate distributed computation on CPU/GPU cluster"""
    print("\n" + "=" * 70)
    print("INITIALIZING PARALLEL DFT COMPUTATION")
    print("=" * 70)

    # Simulate cluster initialization
    n_nodes = 8
    n_gpus_per_node = 4
    total_cores = n_nodes * n_gpus_per_node

    print(f"Cluster Configuration:")
    print(f"  • Compute Nodes:     {n_nodes}")
    print(f"  • GPUs per Node:     {n_gpus_per_node}")
    print(f"  • Total GPU Cores:   {total_cores}")
    print(f"  • Memory per Node:   128 GB")

    print("\nDistributing workload...")
    for i in range(5):
        time.sleep(0.3)
        progress = (i + 1) * 20
        print(f"  [{progress:3d}%] Computing electron density on {i+1}/{5} node groups")

    print("\nSelf-consistent field (SCF) convergence...")
    for iteration in range(1, 6):
        time.sleep(0.2)
        energy_diff = 1.0 / (iteration ** 2)
        print(f"  Iteration {iteration}: ΔE = {energy_diff:.6f} eV")

    print("  ✓ SCF converged!\n")


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("DENSITY FUNCTIONAL THEORY (DFT) SIMULATION")
    print("Materials Science - Electronic Structure Calculation")
    print("=" * 70 + "\n")

    # Simulate parallel computation
    simulate_parallel_computation()

    # Create DFT simulator instance
    print("=" * 70)
    print("BAND STRUCTURE CALCULATION")
    print("=" * 70 + "\n")

    dft = DFTSimulator(material_name="Silicon", lattice_constant=5.43)

    # Generate k-point path
    dft.generate_k_path(n_points=200)

    # Calculate band structure
    dft.calculate_band_structure(n_bands=8)

    # Calculate density of states
    dft.calculate_density_of_states()

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70 + "\n")

    # Create visualizations
    dft.plot_band_structure()
    dft.plot_density_of_states()
    dft.plot_energy_levels()

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70 + "\n")

    # Save results
    dft.save_results()

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*")):
        print(f"  • {file.name}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
