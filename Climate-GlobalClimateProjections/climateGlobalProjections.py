#!/usr/bin/env python3
"""
Global Climate Projections Demo

Models atmospheric/ocean dynamics over decades on HPC clusters;
outputs temperature/rainfall maps to predict regional climate change impacts.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import json

# Create output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ClimateModel:
    """Simulates global climate projections with atmospheric/ocean dynamics."""

    def __init__(self, grid_size=50, years=100, base_temp=15.0):
        """
        Initialize the climate model.

        Args:
            grid_size: Resolution of the spatial grid (lat/lon)
            years: Number of years to simulate
            base_temp: Base global temperature in Celsius
        """
        self.grid_size = grid_size
        self.years = years
        self.base_temp = base_temp
        self.start_year = 2020

        # Create spatial grids (latitude, longitude)
        self.lat = np.linspace(-90, 90, grid_size)
        self.lon = np.linspace(-180, 180, grid_size)
        self.LAT, self.LON = np.meshgrid(self.lat, self.lon)

        # Initialize climate variables
        self.temperature_history = []
        self.rainfall_history = []
        self.co2_levels = []

    def simulate_atmospheric_dynamics(self, year_idx):
        """
        Simulate atmospheric temperature patterns with ocean coupling.

        Models:
        - Latitudinal temperature gradient
        - Ocean heat transport
        - CO2-driven warming trend
        - Natural variability
        """
        # CO2 forcing (increasing over time)
        co2_forcing = 0.02 * year_idx  # +2°C per century
        self.co2_levels.append(400 + year_idx * 2.5)  # ppm

        # Base temperature pattern (warmer at equator, colder at poles)
        lat_gradient = self.base_temp - 0.5 * np.abs(self.LAT)

        # Ocean heat transport effects (warming poles faster)
        polar_amplification = 0.03 * year_idx * (1 - np.cos(np.radians(self.LAT)) ** 2)

        # Atmospheric circulation patterns (El Niño-like oscillations)
        elnino_cycle = 2.0 * np.sin(2 * np.pi * year_idx / 7.0)
        ocean_pattern = elnino_cycle * np.sin(np.radians(self.LAT)) * np.cos(np.radians(self.LON / 2))

        # Natural variability
        noise = np.random.normal(0, 0.3, self.LAT.shape)

        # Combined temperature field
        temperature = lat_gradient + co2_forcing + polar_amplification + ocean_pattern + noise

        return temperature

    def simulate_rainfall_patterns(self, temperature, year_idx):
        """
        Simulate rainfall patterns based on temperature and atmospheric dynamics.

        Models:
        - Hadley cell circulation
        - Monsoon systems
        - Climate change impacts on precipitation
        """
        # Base rainfall pattern (ITCZ and mid-latitude storm tracks)
        itcz = 2000 * np.exp(-((self.LAT - 5) ** 2) / 100)  # Intertropical Convergence Zone
        mid_lat_storms = 800 * np.exp(-((np.abs(self.LAT) - 45) ** 2) / 100)

        # Climate change impact: wet gets wetter, dry gets drier
        climate_change_factor = 1 + 0.07 * (temperature - self.base_temp) / 10

        # Monsoon variability
        monsoon = 500 * np.maximum(0, np.sin(np.radians(self.LAT))) * \
                  np.sin(np.radians(self.LON / 4 + year_idx * 30))

        # Desertification in subtropics
        subtropical_drying = -0.5 * year_idx * np.exp(-((np.abs(self.LAT) - 25) ** 2) / 50)

        # Combined rainfall pattern (mm/year)
        rainfall = (itcz + mid_lat_storms + monsoon) * climate_change_factor + subtropical_drying
        rainfall = np.maximum(rainfall, 0)  # No negative rainfall

        # Add variability
        rainfall += np.random.normal(0, 50, rainfall.shape)
        rainfall = np.maximum(rainfall, 0)

        return rainfall

    def run_simulation(self):
        """Run the full climate simulation over the specified time period."""
        print(f"Starting climate simulation: {self.start_year} to {self.start_year + self.years}")
        print(f"Grid resolution: {self.grid_size}x{self.grid_size}")
        print(f"Modeling atmospheric and ocean dynamics...\n")

        for year_idx in range(self.years):
            # Simulate atmospheric dynamics
            temperature = self.simulate_atmospheric_dynamics(year_idx)
            self.temperature_history.append(temperature)

            # Simulate coupled rainfall patterns
            rainfall = self.simulate_rainfall_patterns(temperature, year_idx)
            self.rainfall_history.append(rainfall)

            if (year_idx + 1) % 20 == 0 or year_idx == 0:
                avg_temp = np.mean(temperature)
                avg_rainfall = np.mean(rainfall)
                print(f"Year {self.start_year + year_idx}: "
                      f"Avg Temp = {avg_temp:.2f}°C, "
                      f"Avg Rainfall = {avg_rainfall:.0f} mm/year")

        print("\nSimulation complete!")
        return self.temperature_history, self.rainfall_history


def plot_global_maps(model):
    """Create temperature and rainfall map visualizations."""
    print("\nGenerating global climate maps...")

    # Create comparison: initial vs final state
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Global Climate Projections: Temperature and Rainfall Changes',
                 fontsize=16, fontweight='bold')

    # Temperature - Initial
    im1 = axes[0, 0].contourf(model.LON, model.LAT, model.temperature_history[0],
                              levels=20, cmap='RdYlBu_r')
    axes[0, 0].set_title(f'Temperature {model.start_year} (°C)', fontweight='bold')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0, 0])

    # Temperature - Final
    im2 = axes[0, 1].contourf(model.LON, model.LAT, model.temperature_history[-1],
                              levels=20, cmap='RdYlBu_r')
    axes[0, 1].set_title(f'Temperature {model.start_year + model.years} (°C)', fontweight='bold')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[0, 1])

    # Rainfall - Initial
    im3 = axes[1, 0].contourf(model.LON, model.LAT, model.rainfall_history[0],
                              levels=20, cmap='YlGnBu')
    axes[1, 0].set_title(f'Rainfall {model.start_year} (mm/year)', fontweight='bold')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[1, 0])

    # Rainfall - Final
    im4 = axes[1, 1].contourf(model.LON, model.LAT, model.rainfall_history[-1],
                              levels=20, cmap='YlGnBu')
    axes[1, 1].set_title(f'Rainfall {model.start_year + model.years} (mm/year)', fontweight='bold')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'global_climate_maps.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_change_maps(model):
    """Create maps showing the change in temperature and rainfall."""
    print("Generating climate change impact maps...")

    temp_change = model.temperature_history[-1] - model.temperature_history[0]
    rainfall_change = model.rainfall_history[-1] - model.rainfall_history[0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f'Regional Climate Change Impacts ({model.start_year}-{model.start_year + model.years})',
                 fontsize=16, fontweight='bold')

    # Temperature change
    im1 = axes[0].contourf(model.LON, model.LAT, temp_change,
                           levels=15, cmap='RdYlBu_r')
    axes[0].set_title('Temperature Change (°C)', fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    cbar1 = plt.colorbar(im1, ax=axes[0])

    # Rainfall change
    im2 = axes[1].contourf(model.LON, model.LAT, rainfall_change,
                           levels=15, cmap='BrBG')
    axes[1].set_title('Rainfall Change (mm/year)', fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    cbar2 = plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'climate_change_impacts.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_time_series(model):
    """Create time series plots of global averages."""
    print("Generating time series projections...")

    years = np.arange(model.start_year, model.start_year + model.years)

    # Calculate global averages
    avg_temps = [np.mean(temp) for temp in model.temperature_history]
    avg_rainfall = [np.mean(rain) for rain in model.rainfall_history]

    # Calculate regional averages (Arctic, Tropics, Antarctica)
    arctic_temps = [np.mean(temp[model.LAT > 66.5]) for temp in model.temperature_history]
    tropic_temps = [np.mean(temp[np.abs(model.LAT) < 23.5]) for temp in model.temperature_history]
    antarctic_temps = [np.mean(temp[model.LAT < -66.5]) for temp in model.temperature_history]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Global Climate Projections: Time Series Analysis',
                 fontsize=16, fontweight='bold')

    # Global average temperature
    axes[0, 0].plot(years, avg_temps, 'r-', linewidth=2, label='Global Average')
    axes[0, 0].set_title('Global Average Temperature', fontweight='bold')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Regional temperature comparison
    axes[0, 1].plot(years, arctic_temps, 'b-', linewidth=2, label='Arctic (>66.5°N)')
    axes[0, 1].plot(years, tropic_temps, 'r-', linewidth=2, label='Tropics (±23.5°)')
    axes[0, 1].plot(years, antarctic_temps, 'c-', linewidth=2, label='Antarctic (<66.5°S)')
    axes[0, 1].set_title('Regional Temperature Trends', fontweight='bold')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Global rainfall
    axes[1, 0].plot(years, avg_rainfall, 'b-', linewidth=2)
    axes[1, 0].set_title('Global Average Rainfall', fontweight='bold')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Rainfall (mm/year)')
    axes[1, 0].grid(True, alpha=0.3)

    # CO2 levels
    axes[1, 1].plot(years, model.co2_levels, 'g-', linewidth=2)
    axes[1, 1].set_title('Atmospheric CO₂ Concentration', fontweight='bold')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('CO₂ (ppm)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'time_series_projections.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def export_data(model):
    """Export climate data to files for further analysis."""
    print("\nExporting climate data...")

    # Export summary statistics
    summary = {
        'simulation_info': {
            'start_year': model.start_year,
            'end_year': model.start_year + model.years,
            'grid_size': model.grid_size,
            'base_temperature': model.base_temp
        },
        'initial_conditions': {
            'avg_temperature': float(np.mean(model.temperature_history[0])),
            'avg_rainfall': float(np.mean(model.rainfall_history[0])),
            'co2_concentration': float(model.co2_levels[0])
        },
        'final_conditions': {
            'avg_temperature': float(np.mean(model.temperature_history[-1])),
            'avg_rainfall': float(np.mean(model.rainfall_history[-1])),
            'co2_concentration': float(model.co2_levels[-1])
        },
        'changes': {
            'temperature_increase': float(np.mean(model.temperature_history[-1]) -
                                         np.mean(model.temperature_history[0])),
            'rainfall_change': float(np.mean(model.rainfall_history[-1]) -
                                    np.mean(model.rainfall_history[0])),
            'co2_increase': float(model.co2_levels[-1] - model.co2_levels[0])
        }
    }

    summary_path = os.path.join(OUTPUT_DIR, 'climate_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # Export time series data
    years = list(range(model.start_year, model.start_year + model.years))
    avg_temps = [float(np.mean(temp)) for temp in model.temperature_history]
    avg_rainfall = [float(np.mean(rain)) for rain in model.rainfall_history]

    timeseries_path = os.path.join(OUTPUT_DIR, 'timeseries_data.csv')
    with open(timeseries_path, 'w') as f:
        f.write('Year,Temperature_C,Rainfall_mm,CO2_ppm\n')
        for i, year in enumerate(years):
            f.write(f'{year},{avg_temps[i]:.2f},{avg_rainfall[i]:.1f},{model.co2_levels[i]:.1f}\n')
    print(f"Saved: {timeseries_path}")

    # Export final temperature and rainfall grids
    np.save(os.path.join(OUTPUT_DIR, 'final_temperature_grid.npy'),
            model.temperature_history[-1])
    np.save(os.path.join(OUTPUT_DIR, 'final_rainfall_grid.npy'),
            model.rainfall_history[-1])
    print(f"Saved: temperature and rainfall grid data (.npy files)")


def main():
    """Main execution function."""
    print("=" * 70)
    print("GLOBAL CLIMATE PROJECTIONS DEMO")
    print("=" * 70)
    print("\nModeling atmospheric/ocean dynamics over decades...")
    print("Predicting regional climate change impacts\n")

    # Initialize and run the climate model
    model = ClimateModel(grid_size=50, years=100, base_temp=15.0)
    model.run_simulation()

    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    plot_global_maps(model)
    plot_change_maps(model)
    plot_time_series(model)

    # Export data
    print("\n" + "=" * 70)
    print("EXPORTING DATA")
    print("=" * 70)
    export_data(model)

    # Summary
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print(f"\nKey findings:")
    print(f"  - Temperature increase: {np.mean(model.temperature_history[-1]) - np.mean(model.temperature_history[0]):.2f}°C")
    print(f"  - Polar amplification: Arctic warming faster than global average")
    print(f"  - Rainfall changes: Regional variations in precipitation patterns")
    print(f"  - CO₂ increase: {model.co2_levels[-1] - model.co2_levels[0]:.1f} ppm")
    print("\nVisualization files generated:")
    print("  - global_climate_maps.png")
    print("  - climate_change_impacts.png")
    print("  - time_series_projections.png")
    print("\nData files generated:")
    print("  - climate_summary.json")
    print("  - timeseries_data.csv")
    print("  - final_temperature_grid.npy")
    print("  - final_rainfall_grid.npy")
    print()


if __name__ == "__main__":
    main()
