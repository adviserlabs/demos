import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time

# Parameters
nx, ny = 41, 41  # Grid points
lx, ly = 1.0, 1.0  # Cavity dimensions
dx, dy = lx / (nx - 1), ly / (ny - 1)  # Grid spacing
Re = 100  # Reynolds number
Pr = 0.71  # Prandtl number
nu = 0.01  # Kinematic viscosity
dt = 0.001  # Time step
n_iter = 1000  # Number of time steps
n_iter = 200  # Number of time steps
tol = 1e-5  # Tolerance for streamfunction iteration
n_workers = 8  # Number of threads (adjust based on your CPU)

# Grid
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)
X, Y = np.meshgrid(x, y)

# Arrays
psi = np.zeros((ny, nx))  # Streamfunction
omega = np.zeros((ny, nx))  # Vorticity
T = np.zeros((ny, nx))  # Temperature

# Boundary conditions for temperature
T[-1, :] = 1.0  # Top wall (hot)
T[0, :] = 0.0   # Bottom wall (cold)

def solve_streamfunction(psi, omega, dx, dy, tol=1e-5, max_iter=1000):
    """Solve streamfunction equation using successive over-relaxation."""
    psi_new = psi.copy()
    for _ in range(max_iter):
        psi_old = psi_new.copy()
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                psi_new[i, j] = 0.25 * (psi_old[i+1, j] + psi_old[i-1, j] +
                                        psi_old[i, j+1] + psi_old[i, j-1] +
                                        dx**2 * omega[i, j])
        psi_new[0, :] = 0.0  # Bottom
        psi_new[-1, :] = 0.0  # Top
        psi_new[:, 0] = 0.0  # Left
        psi_new[:, -1] = 0.0  # Right
        if np.max(np.abs(psi_new - psi_old)) < tol:
            break
    return psi_new

def vorticity_worker(i, psi, omega, u, v, dx, dy, nu, dt):
    """Worker function for vorticity update for row i."""
    omega_new = omega[i].copy()
    for j in range(1, nx-1):
        conv_x = u[i, j] * (omega[i, j+1] - omega[i, j-1]) / (2 * dx)
        conv_y = v[i, j] * (omega[i+1, j] - omega[i-1, j]) / (2 * dy)
        diff = nu * ((omega[i+1, j] - 2*omega[i, j] + omega[i-1, j]) / dy**2 +
                     (omega[i, j+1] - 2*omega[i, j] + omega[i, j-1]) / dx**2)
        omega_new[j] = omega[i, j] + dt * (-conv_x - conv_y + diff)
    return i, omega_new

def update_vorticity(psi, omega, dx, dy, nu, dt, n_workers):
    """Update vorticity using multithreading."""
    omega_new = omega.copy()
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    u[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * dy)  # u = d(psi)/dy
    v[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2 * dx)  # v = -d(psi)/dx
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(vorticity_worker, i, psi, omega, u, v, dx, dy, nu, dt)
                   for i in range(1, ny-1)]
        for future in futures:
            i, row = future.result()
            omega_new[i, :] = row
    
    # Boundary conditions for vorticity
    omega_new[0, :] = (psi[1, :] - psi[0, :]) / (dy**2)  # Bottom
    omega_new[-1, :] = (psi[-2, :] - psi[-1, :] - 2*dy) / (dy**2)  # Top (u=1)
    omega_new[:, 0] = (psi[:, 1] - psi[:, 0]) / (dx**2)  # Left
    omega_new[:, -1] = (psi[:, -2] - psi[:, -1]) / (dx**2)  # Right
    
    return omega_new

def temperature_worker(i, psi, T, u, v, dx, dy, nu, Pr, dt):
    """Worker function for temperature update for row i."""
    T_new = T[i].copy()
    for j in range(1, nx-1):
        conv_x = u[i, j] * (T[i, j+1] - T[i, j-1]) / (2 * dx)
        conv_y = v[i, j] * (T[i+1, j] - T[i-1, j]) / (2 * dy)
        diff = (nu / Pr) * ((T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dy**2 +
                            (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dx**2)
        T_new[j] = T[i, j] + dt * (-conv_x - conv_y + diff)
    return i, T_new

def update_temperature(T, psi, dx, dy, nu, Pr, dt, n_workers):
    """Update temperature using multithreading."""
    T_new = T.copy()
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    u[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * dy)
    v[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2 * dx)
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(temperature_worker, i, psi, T, u, v, dx, dy, nu, Pr, dt)
                   for i in range(1, ny-1)]
        for future in futures:
            i, row = future.result()
            T_new[i, :] = row
    
    # Boundary conditions for temperature
    T_new[-1, :] = 1.0  # Top (hot)
    T_new[0, :] = 0.0   # Bottom (cold)
    T_new[:, 0] = T_new[:, 1]  # Left (insulated)
    T_new[:, -1] = T_new[:, -2]  # Right (insulated)
    
    return T_new

# Time-stepping loop with timing
start_time = time.time()
for n in range(n_iter):
    psi = solve_streamfunction(psi, omega, dx, dy, tol)
    omega = update_vorticity(psi, omega, dx, dy, nu, dt, n_workers)
    T = update_temperature(T, psi, dx, dy, nu, Pr, dt, n_workers)
    if n % 100 == 0:
        print(f"Iteration {n}/{n_iter}")
print(f"Simulation time: {time.time() - start_time:.2f} seconds")

# Compute velocities for plotting
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
u[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * dy)
v[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2 * dx)

# Visualization
plt.figure(figsize=(12, 5))

# Streamfunction (velocity field)
plt.subplot(1, 2, 1)
plt.contourf(X, Y, psi, levels=20, cmap='jet')
plt.colorbar(label='Streamfunction')
plt.streamplot(X, Y, u, v, color='k')
plt.title('Velocity Field (Streamlines)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('square')

# Temperature
plt.subplot(1, 2, 2)
plt.contourf(X, Y, T, levels=20, cmap='hot')
plt.colorbar(label='Temperature')
plt.title('Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('square')

plt.tight_layout()
#plt.show()
print("Saving to output/ directory")
plt.savefig('output/Adviser-ThermalLidDrivenCavitySimulation-Plot.png')

