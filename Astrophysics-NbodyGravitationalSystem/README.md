# Astrophysics - N-Body Gravitational System
Evolves orbits of millions of particles (e.g., galaxies) on parallel clusters;
outputs collision/formation visuals to study cosmic structure evolution.

By default simulates 10,000 particles (scalable to millions) using Numba +
multiprocessing for cluster-parallelization. Runs in ~2 minutes on 64 cores vs
~4 hours single-core.

## Running the Demonstration
`adviser run "pip install -r requirements.txt && python nBodyGravitationalSystem.py"`

```
- Launching N-Body Demo: 10,000 particles, 500 steps, 2 cores
- Computing orbits on cluster...
- COMPLETE: 128.7s on 2 cores (10,000 particles)
- SAVED: nbody_data.npz (Load in Jupyter)
- SAVED: nbody_results.png
```
