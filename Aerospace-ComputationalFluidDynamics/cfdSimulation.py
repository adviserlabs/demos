#!/usr/bin/env python3
"""
Computational Fluid Dynamics (CFD) Simulation
Simulates airflow around a NACA airfoil using simplified potential flow theory
and panel method to calculate drag/lift coefficients for aircraft optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
from datetime import datetime


class NACAirfoil:
    """Generate NACA 4-digit airfoil coordinates."""

    def __init__(self, naca_digits="2412", num_points=100):
        """
        Initialize NACA airfoil.

        Args:
            naca_digits: 4-digit NACA designation (e.g., "2412")
            num_points: Number of points on airfoil surface
        """
        self.m = int(naca_digits[0]) / 100.0  # Maximum camber
        self.p = int(naca_digits[1]) / 10.0   # Location of maximum camber
        self.t = int(naca_digits[2:]) / 100.0 # Maximum thickness
        self.num_points = num_points
        self.x, self.y = self._generate_coordinates()

    def _generate_coordinates(self):
        """Generate airfoil surface coordinates."""
        # Cosine spacing for better resolution at leading/trailing edges
        beta = np.linspace(0, np.pi, self.num_points // 2)
        x = (1 - np.cos(beta)) / 2

        # Thickness distribution (symmetrical)
        yt = 5 * self.t * (0.2969 * np.sqrt(x) - 0.1260 * x -
                           0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

        # Camber line
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)

        # Forward of maximum camber
        mask = x < self.p
        if self.p > 0:
            yc[mask] = (self.m / self.p**2) * (2 * self.p * x[mask] - x[mask]**2)
            dyc_dx[mask] = (2 * self.m / self.p**2) * (self.p - x[mask])

        # Aft of maximum camber
        mask = x >= self.p
        if self.p < 1:
            yc[mask] = (self.m / (1 - self.p)**2) * ((1 - 2 * self.p) +
                        2 * self.p * x[mask] - x[mask]**2)
            dyc_dx[mask] = (2 * self.m / (1 - self.p)**2) * (self.p - x[mask])

        # Combine camber and thickness
        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # Combine upper and lower surfaces
        x_airfoil = np.concatenate([xu[::-1], xl[1:]])
        y_airfoil = np.concatenate([yu[::-1], yl[1:]])

        return x_airfoil, y_airfoil


class CFDSimulation:
    """Simplified CFD simulation using potential flow and panel method."""

    def __init__(self, airfoil, angle_of_attack=5.0, freestream_velocity=50.0):
        """
        Initialize CFD simulation.

        Args:
            airfoil: NACAirfoil object
            angle_of_attack: Angle of attack in degrees
            freestream_velocity: Freestream velocity in m/s
        """
        self.airfoil = airfoil
        self.alpha = np.deg2rad(angle_of_attack)  # Convert to radians
        self.V_inf = freestream_velocity
        self.rho = 1.225  # Air density at sea level (kg/m³)

        # Create computational mesh
        self.create_mesh()

        # Solve flow field
        self.solve_flow()

    def create_mesh(self):
        """Create computational mesh around airfoil."""
        # Define domain boundaries
        self.x_min, self.x_max = -2, 3
        self.y_min, self.y_max = -2, 2
        self.nx, self.ny = 150, 100

        # Create mesh grid
        x = np.linspace(self.x_min, self.x_max, self.nx)
        y = np.linspace(self.y_min, self.y_max, self.ny)
        self.X, self.Y = np.meshgrid(x, y)

    def solve_flow(self):
        """Solve potential flow around airfoil using simplified panel method."""
        # Freestream velocity components
        U_inf = self.V_inf * np.cos(self.alpha)
        V_inf = self.V_inf * np.sin(self.alpha)

        # Initialize velocity field with freestream
        self.U = np.ones_like(self.X) * U_inf
        self.V = np.ones_like(self.Y) * V_inf

        # Add circulation effect (simplified vortex panel method)
        # This simulates the bound vortex on the airfoil
        gamma = 4 * np.pi * self.V_inf * np.sin(self.alpha)  # Circulation strength

        for i in range(self.nx):
            for j in range(self.ny):
                x, y = self.X[j, i], self.Y[j, i]

                # Check if point is inside airfoil (skip if inside)
                if self._is_inside_airfoil(x, y):
                    self.U[j, i] = 0
                    self.V[j, i] = 0
                    continue

                # Add velocity induced by circulation (bound vortex)
                dx = x - 0.25  # Vortex at quarter-chord
                dy = y
                r_sq = dx**2 + dy**2 + 0.01  # Smoothing factor

                # Induced velocity from vortex
                self.U[j, i] -= gamma * dy / (2 * np.pi * r_sq)
                self.V[j, i] += gamma * dx / (2 * np.pi * r_sq)

        # Calculate pressure field using Bernoulli equation
        V_mag = np.sqrt(self.U**2 + self.V**2)
        self.pressure = self.rho * (self.V_inf**2 - V_mag**2) / 2

        # Calculate pressure coefficient
        self.Cp = self.pressure / (0.5 * self.rho * self.V_inf**2)

    def _is_inside_airfoil(self, x, y):
        """Check if point is inside airfoil using ray casting."""
        if x < 0 or x > 1:
            return False

        # Simple check: point is inside if between upper and lower surfaces
        x_airfoil = self.airfoil.x
        y_airfoil = self.airfoil.y

        # Find closest x-coordinate on airfoil
        idx = np.argmin(np.abs(x_airfoil - x))

        # Get upper and lower bounds
        y_upper = np.max(y_airfoil[np.abs(x_airfoil - x) < 0.05])
        y_lower = np.min(y_airfoil[np.abs(x_airfoil - x) < 0.05])

        return y_lower <= y <= y_upper

    def calculate_forces(self):
        """Calculate lift and drag coefficients."""
        # Surface pressure integration (simplified)
        x_surf = self.airfoil.x
        y_surf = self.airfoil.y

        # Sample pressure at airfoil surface
        pressures = []
        for xs, ys in zip(x_surf, y_surf):
            # Find nearest mesh point
            i = np.argmin(np.abs(self.X[0, :] - xs))
            j = np.argmin(np.abs(self.Y[:, 0] - ys))
            pressures.append(self.Cp[j, i])

        pressures = np.array(pressures)

        # Integrate pressure to get forces
        # Calculate panel normals
        dx = np.diff(x_surf)
        dy = np.diff(y_surf)

        # Extend to match array size
        dx = np.append(dx, dx[-1])
        dy = np.append(dy, dy[-1])

        # Normal vectors (perpendicular to surface)
        length = np.sqrt(dx**2 + dy**2)
        nx = -dy / length
        ny = dx / length

        # Force coefficients (simplified integration)
        Cn = np.sum(pressures * length)  # Normal force coefficient

        # Decompose into lift and drag
        self.Cl = Cn * np.cos(self.alpha)  # Lift coefficient
        self.Cd = 0.02 + 0.05 * np.sin(self.alpha)**2  # Drag coefficient (simplified)

        # Lift-to-drag ratio
        self.L_D_ratio = self.Cl / self.Cd if self.Cd > 0 else 0

        return self.Cl, self.Cd


class CFDVisualizer:
    """Visualize CFD simulation results."""

    def __init__(self, simulation, output_dir="output"):
        """
        Initialize visualizer.

        Args:
            simulation: CFDSimulation object
            output_dir: Directory to save output files
        """
        self.sim = simulation
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def plot_pressure_field(self):
        """Plot pressure coefficient field."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot pressure contours
        levels = np.linspace(-2, 1, 20)
        contour = ax.contourf(self.sim.X, self.sim.Y, self.sim.Cp,
                              levels=levels, cmap='RdBu_r', extend='both')

        # Plot airfoil
        ax.plot(self.sim.airfoil.x, self.sim.airfoil.y, 'k-', linewidth=2)
        ax.fill(self.sim.airfoil.x, self.sim.airfoil.y, 'white')

        # Formatting
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        ax.set_title(f'Pressure Coefficient (Cp) - α = {np.rad2deg(self.sim.alpha):.1f}°')
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.8, 0.8)

        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Cp')

        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'pressure_field.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

    def plot_velocity_field(self):
        """Plot velocity field with streamlines."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate velocity magnitude
        V_mag = np.sqrt(self.sim.U**2 + self.sim.V**2)

        # Plot velocity magnitude
        contour = ax.contourf(self.sim.X, self.sim.Y, V_mag,
                              levels=20, cmap='viridis')

        # Plot streamlines
        ax.streamplot(self.sim.X, self.sim.Y, self.sim.U, self.sim.V,
                     color='white', linewidth=0.5, density=1.5, arrowsize=0.8)

        # Plot airfoil
        ax.plot(self.sim.airfoil.x, self.sim.airfoil.y, 'r-', linewidth=2)
        ax.fill(self.sim.airfoil.x, self.sim.airfoil.y, 'gray')

        # Formatting
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        ax.set_title(f'Velocity Field and Streamlines - α = {np.rad2deg(self.sim.alpha):.1f}°')
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.8, 0.8)

        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Velocity Magnitude (m/s)')

        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'velocity_field.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

    def plot_surface_pressure(self):
        """Plot pressure distribution on airfoil surface."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Sample pressure at airfoil surface
        x_surf = self.sim.airfoil.x
        y_surf = self.sim.airfoil.y
        Cp_surf = []

        for xs, ys in zip(x_surf, y_surf):
            i = np.argmin(np.abs(self.sim.X[0, :] - xs))
            j = np.argmin(np.abs(self.sim.Y[:, 0] - ys))
            Cp_surf.append(self.sim.Cp[j, i])

        # Separate upper and lower surfaces
        mid_idx = len(x_surf) // 2
        x_upper = x_surf[:mid_idx]
        Cp_upper = Cp_surf[:mid_idx]
        x_lower = x_surf[mid_idx:]
        Cp_lower = Cp_surf[mid_idx:]

        # Plot
        ax.plot(x_upper, Cp_upper, 'b-', linewidth=2, label='Upper Surface')
        ax.plot(x_lower, Cp_lower, 'r-', linewidth=2, label='Lower Surface')

        # Formatting
        ax.set_xlabel('x/c')
        ax.set_ylabel('Cp')
        ax.set_title(f'Surface Pressure Distribution - α = {np.rad2deg(self.sim.alpha):.1f}°')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.invert_yaxis()  # Negative Cp is suction (upward)

        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'surface_pressure.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

    def plot_airfoil_geometry(self):
        """Plot airfoil geometry."""
        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot airfoil
        ax.plot(self.sim.airfoil.x, self.sim.airfoil.y, 'b-', linewidth=2)
        ax.fill(self.sim.airfoil.x, self.sim.airfoil.y, 'lightblue', alpha=0.5)

        # Mark chord line
        ax.plot([0, 1], [0, 0], 'k--', linewidth=1, label='Chord Line')

        # Mark quarter-chord (aerodynamic center)
        ax.plot(0.25, 0, 'ro', markersize=8, label='Quarter-Chord')

        # Formatting
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        ax.set_title(f'NACA Airfoil Geometry')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'airfoil_geometry.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

    def plot_lift_drag_polar(self, alpha_range=None):
        """Plot lift-drag polar for different angles of attack."""
        if alpha_range is None:
            alpha_range = np.linspace(-5, 15, 21)

        Cl_values = []
        Cd_values = []

        print("\nCalculating lift-drag polar...")
        for alpha in alpha_range:
            sim = CFDSimulation(self.sim.airfoil, angle_of_attack=alpha,
                               freestream_velocity=self.sim.V_inf)
            Cl, Cd = sim.calculate_forces()
            Cl_values.append(Cl)
            Cd_values.append(Cd)

        # Plot Cl vs alpha
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        ax1.plot(alpha_range, Cl_values, 'b-o', linewidth=2)
        ax1.set_xlabel('Angle of Attack (°)')
        ax1.set_ylabel('Lift Coefficient (Cl)')
        ax1.set_title('Lift Coefficient vs. Angle of Attack')
        ax1.grid(True, alpha=0.3)

        # Plot Cd vs alpha
        ax2.plot(alpha_range, Cd_values, 'r-o', linewidth=2)
        ax2.set_xlabel('Angle of Attack (°)')
        ax2.set_ylabel('Drag Coefficient (Cd)')
        ax2.set_title('Drag Coefficient vs. Angle of Attack')
        ax2.grid(True, alpha=0.3)

        # Plot drag polar (Cl vs Cd)
        ax3.plot(Cd_values, Cl_values, 'g-o', linewidth=2)
        ax3.set_xlabel('Drag Coefficient (Cd)')
        ax3.set_ylabel('Lift Coefficient (Cl)')
        ax3.set_title('Lift-Drag Polar')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'lift_drag_polar.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

        return alpha_range, Cl_values, Cd_values

    def save_results(self, alpha_range=None, Cl_values=None, Cd_values=None):
        """Save numerical results to file."""
        filename = os.path.join(self.output_dir, 'simulation_results.txt')

        with open(filename, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CFD SIMULATION RESULTS\n")
            f.write("Computational Fluid Dynamics - Airfoil Analysis\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Simulation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("AIRFOIL CONFIGURATION:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Maximum Camber:          {self.sim.airfoil.m * 100:.2f}%\n")
            f.write(f"  Camber Location:         {self.sim.airfoil.p * 100:.2f}% chord\n")
            f.write(f"  Maximum Thickness:       {self.sim.airfoil.t * 100:.2f}% chord\n\n")

            f.write("FLOW CONDITIONS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Freestream Velocity:     {self.sim.V_inf:.2f} m/s\n")
            f.write(f"  Angle of Attack:         {np.rad2deg(self.sim.alpha):.2f}°\n")
            f.write(f"  Air Density:             {self.sim.rho:.3f} kg/m³\n")
            f.write(f"  Reynolds Number:         ~{self.sim.V_inf * 1.0 / 1.5e-5:.0f}\n\n")

            f.write("AERODYNAMIC COEFFICIENTS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Lift Coefficient (Cl):   {self.sim.Cl:.4f}\n")
            f.write(f"  Drag Coefficient (Cd):   {self.sim.Cd:.4f}\n")
            f.write(f"  Lift-to-Drag Ratio:      {self.sim.L_D_ratio:.2f}\n\n")

            # Dynamic pressure and forces (for 1m² wing area)
            q = 0.5 * self.sim.rho * self.sim.V_inf**2
            f.write("FORCES (per m² wing area):\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Dynamic Pressure:        {q:.2f} Pa\n")
            f.write(f"  Lift Force:              {self.sim.Cl * q:.2f} N/m²\n")
            f.write(f"  Drag Force:              {self.sim.Cd * q:.2f} N/m²\n\n")

            if alpha_range is not None and Cl_values is not None:
                f.write("LIFT-DRAG POLAR DATA:\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'Alpha (°)':>10} {'Cl':>12} {'Cd':>12} {'L/D':>12}\n")
                f.write("-" * 70 + "\n")
                for alpha, Cl, Cd in zip(alpha_range, Cl_values, Cd_values):
                    L_D = Cl / Cd if Cd > 0 else 0
                    f.write(f"{alpha:>10.1f} {Cl:>12.4f} {Cd:>12.4f} {L_D:>12.2f}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("OPTIMIZATION INSIGHTS:\n")
            f.write("=" * 70 + "\n")
            f.write("For optimal fuel efficiency, maximize L/D ratio:\n")
            f.write("  - Higher L/D = less drag for same lift = less fuel consumption\n")
            f.write("  - Typical cruise: operate near maximum L/D angle of attack\n")
            f.write("  - Reduce drag through airfoil shape, surface smoothness\n")
            f.write("=" * 70 + "\n")

        print(f"Saved: {filename}")


def main():
    """Main simulation function."""
    print("=" * 70)
    print("COMPUTATIONAL FLUID DYNAMICS (CFD) SIMULATION")
    print("Airflow Analysis for Aircraft Optimization")
    print("=" * 70)

    # Create airfoil (NACA 2412 - common for general aviation)
    print("\n[1/5] Generating NACA 2412 airfoil geometry...")
    airfoil = NACAirfoil(naca_digits="2412", num_points=100)
    print(f"      Generated {len(airfoil.x)} surface points")

    # Create CFD simulation
    print("\n[2/5] Running CFD simulation...")
    angle_of_attack = 5.0  # degrees
    freestream_velocity = 50.0  # m/s (typical cruise speed)
    simulation = CFDSimulation(airfoil, angle_of_attack, freestream_velocity)
    print(f"      Mesh size: {simulation.nx} x {simulation.ny} = {simulation.nx * simulation.ny} points")

    # Calculate forces
    print("\n[3/5] Calculating lift and drag coefficients...")
    Cl, Cd = simulation.calculate_forces()
    print(f"      Lift Coefficient (Cl):  {Cl:.4f}")
    print(f"      Drag Coefficient (Cd):  {Cd:.4f}")
    print(f"      Lift-to-Drag Ratio:     {simulation.L_D_ratio:.2f}")

    # Create visualizations
    print("\n[4/5] Generating visualizations...")
    visualizer = CFDVisualizer(simulation)
    visualizer.plot_airfoil_geometry()
    visualizer.plot_pressure_field()
    visualizer.plot_velocity_field()
    visualizer.plot_surface_pressure()

    # Generate lift-drag polar
    print("\n[5/5] Computing lift-drag polar...")
    alpha_range, Cl_values, Cd_values = visualizer.plot_lift_drag_polar()

    # Save results
    visualizer.save_results(alpha_range, Cl_values, Cd_values)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {os.path.abspath(visualizer.output_dir)}/")
    print("\nGenerated files:")
    print("  - airfoil_geometry.png      : Airfoil shape and geometry")
    print("  - pressure_field.png        : Pressure coefficient contours")
    print("  - velocity_field.png        : Velocity magnitude and streamlines")
    print("  - surface_pressure.png      : Pressure distribution on surface")
    print("  - lift_drag_polar.png       : Performance curves vs. angle of attack")
    print("  - simulation_results.txt    : Detailed numerical results")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
