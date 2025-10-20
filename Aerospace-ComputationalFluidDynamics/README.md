# Aerospace Computational Fluid Dynamics
Simulates airflow around vehicles/wings on mesh-based clusters; outputs
drag/lift coefficients to optimize aircraft fuel efficiency.

## Running the Demonstration
`adviser run "pip install -r requirements.txt && python cfdSimulation.py"`

```
======================================================================
COMPUTATIONAL FLUID DYNAMICS (CFD) SIMULATION
Airflow Analysis for Aircraft Optimization
======================================================================

[1/5] Generating NACA 2412 airfoil geometry...
      Generated 99 surface points

[2/5] Running CFD simulation...
      Mesh size: 150 x 100 = 15000 points

[3/5] Calculating lift and drag coefficients...
      Lift Coefficient (Cl):  0.8627
      Drag Coefficient (Cd):  0.0204
      Lift-to-Drag Ratio:     42.33

[4/5] Generating visualizations...
Saved: output/airfoil_geometry.png
Saved: output/pressure_field.png
Saved: output/velocity_field.png
Saved: output/surface_pressure.png

[5/5] Computing lift-drag polar...

Calculating lift-drag polar...
Saved: output/lift_drag_polar.png
Saved: output/simulation_results.txt

======================================================================
SIMULATION COMPLETE!
======================================================================

All results saved to: /home/ubuntu/sky_workdir/output/

Generated files:
  - airfoil_geometry.png      : Airfoil shape and geometry
  - pressure_field.png        : Pressure coefficient contours
  - velocity_field.png        : Velocity magnitude and streamlines
  - surface_pressure.png      : Pressure distribution on surface
  - lift_drag_polar.png       : Performance curves vs. angle of attack
  - simulation_results.txt    : Detailed numerical results

======================================================================
```
