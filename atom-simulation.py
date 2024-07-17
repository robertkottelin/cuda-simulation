import numpy as np
from numba import cuda, float64
import math
import matplotlib.pyplot as plt

# Constants
k_e = 8.9875517873681764e9  # Coulomb's constant (N m^2 C^-2)
G_strong = 1e4  # Arbitrary strong force constant for simplicity
G_weak = 1e2  # Arbitrary weak force constant for simplicity
mass_proton = 1.6726219e-27  # kg
mass_neutron = 1.6750e-27  # kg
mass_electron = 9.10938356e-31  # kg
charge_proton = 1.60217662e-19  # C
charge_electron = -1.60217662e-19  # C
desired_temp = 1.0  # Desired temperature
tau = 0.1  # Time constant for temperature control
epsilon = 1e-30  # Small value to prevent division by zero

# Simulation parameters
N_protons = 100
N_neutrons = 100
N_electrons = 100
N_particles = N_protons + N_neutrons + N_electrons
L = 1e-10  # Size of the simulation box (m)
dt = 1e-22  # Time step (s)
steps = 100000  # Number of simulation steps

# Initialize positions, velocities, and forces
def initialize_positions():
    positions = np.zeros((N_particles, 3), dtype=np.float64)
    # Place protons and neutrons close together
    for i in range(N_protons):
        positions[i] = [L/2 + (i % 2) * 1e-15, L/2, L/2]
    for i in range(N_neutrons):
        positions[N_protons + i] = [L/2, L/2 + (i % 2) * 1e-15, L/2]
    # Place electrons at some distance
    for i in range(N_electrons):
        positions[N_protons + N_neutrons + i] = [L/2, L/2 + 5e-11 + i * 1e-11, L/2]
    return positions

positions = initialize_positions()
initial_positions = positions.copy()
velocities = np.zeros((N_particles, 3), dtype=np.float64)  # Initialize with zero velocities
forces = np.zeros((N_particles, 3), dtype=np.float64)
masses = np.array([mass_proton] * N_protons + [mass_neutron] * N_neutrons + [mass_electron] * N_electrons)
charges = np.array([charge_proton] * N_protons + [0] * N_neutrons + [charge_electron] * N_electrons)

# CUDA kernel for computing forces
@cuda.jit
def compute_forces(positions, velocities, forces, masses, charges, N_particles, L, k_e, G_strong, G_weak, epsilon):
    i = cuda.grid(1)
    if i < N_particles:
        force = cuda.local.array(3, dtype=float64)
        force[0] = 0.0
        force[1] = 0.0
        force[2] = 0.0
        for j in range(N_particles):
            if i != j:
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                dx -= L * round(dx / L)
                dy -= L * round(dy / L)
                dz -= L * round(dz / L)
                r2 = dx * dx + dy * dy + dz * dz + epsilon  # Add epsilon to avoid division by zero
                r = math.sqrt(r2)
                
                # Electromagnetic force
                if charges[i] != 0 and charges[j] != 0:
                    f_e = k_e * charges[i] * charges[j] / r2
                    force[0] += f_e * dx / r
                    force[1] += f_e * dy / r
                    force[2] += f_e * dz / r

                # Strong nuclear force (simplified)
                if r < 1e-15:
                    f_s = -G_strong * (1 - r / 1e-15)  # Attractive with repulsive core
                    force[0] += f_s * dx / r
                    force[1] += f_s * dy / r
                    force[2] += f_s * dz / r

                # Weak nuclear force (simplified, repulsive at very short distances)
                if r < 1e-17:
                    f_w = G_weak / (r2 + epsilon)
                    force[0] += f_w * dx / r
                    force[1] += f_w * dy / r
                    force[2] += f_w * dz / r

                # Pauli exclusion principle (repulsive force for very short distances)
                if r < 1e-15:
                    f_pauli = G_strong / (r2 + epsilon)
                    force[0] += f_pauli * dx / r
                    force[1] += f_pauli * dy / r
                    force[2] += f_pauli * dz / r
        forces[i, 0] = force[0]
        forces[i, 1] = force[1]
        forces[i, 2] = force[2]

# CUDA kernel for updating positions and velocities with reflective boundary conditions
@cuda.jit
def update_positions_velocities(positions, velocities, forces, masses, dt, L, N_particles):
    i = cuda.grid(1)
    if i < N_particles:
        for k in range(3):
            velocities[i, k] += forces[i, k] * dt / masses[i]
            positions[i, k] += velocities[i, k] * dt
            # Reflective boundary conditions
            if positions[i, k] < 0:
                positions[i, k] = -positions[i, k]
                velocities[i, k] = -velocities[i, k]
            elif positions[i, k] > L:
                positions[i, k] = 2*L - positions[i, k]
                velocities[i, k] = -velocities[i, k]

# CUDA kernel for applying Berendsen thermostat
@cuda.jit
def apply_berendsen_thermostat(velocities, current_temp, desired_temp, tau, dt, N_particles):
    scale = cuda.local.array(1, dtype=float64)
    scale[0] = math.sqrt(1.0 + (desired_temp / current_temp - 1.0) * dt / tau)
    i = cuda.grid(1)
    if i < N_particles:
        for k in range(3):
            velocities[i, k] *= scale[0]

# Allocate memory on GPU
positions_device = cuda.to_device(positions)
velocities_device = cuda.to_device(velocities)
forces_device = cuda.to_device(forces)
masses_device = cuda.to_device(masses)
charges_device = cuda.to_device(charges)

# Set up CUDA grid and block dimensions
threads_per_block = 256
blocks_per_grid = (N_particles + (threads_per_block - 1)) // threads_per_block

# Run the simulation
for step in range(steps):
    compute_forces[blocks_per_grid, threads_per_block](positions_device, velocities_device, forces_device, masses_device, charges_device, N_particles, L, k_e, G_strong, G_weak, epsilon)
    update_positions_velocities[blocks_per_grid, threads_per_block](positions_device, velocities_device, forces_device, masses_device, dt, L, N_particles)
    
    # Debugging: Print positions every 10000 steps
    if step % 10000 == 0:
        positions_host = positions_device.copy_to_host()
        print(f"Step {step}:")
        print(positions_host)

    # Compute current temperature
    velocities = velocities_device.copy_to_host()
    kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * velocities ** 2)
    current_temp = (2.0 / 3.0) * kinetic_energy / N_particles
    velocities_device = cuda.to_device(velocities)
    
    # Apply Berendsen thermostat
    apply_berendsen_thermostat[blocks_per_grid, threads_per_block](velocities_device, current_temp, desired_temp, tau, dt, N_particles)

# Copy final data back to host
positions = positions_device.copy_to_host()
velocities = velocities_device.copy_to_host()

# Visualization of the initial and final positions
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.scatter(initial_positions[:N_protons, 0], initial_positions[:N_protons, 1], s=10, c='red', label='Protons')
if N_neutrons > 0:
    plt.scatter(initial_positions[N_protons:N_protons+N_neutrons, 0], initial_positions[N_protons:N_protons+N_neutrons, 1], s=10, c='green', label='Neutrons')
plt.scatter(initial_positions[-N_electrons:, 0], initial_positions[-N_electrons:, 1], s=10, c='blue', label='Electrons')
plt.xlim(0, L)
plt.ylim(0, L)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Initial positions of subatomic particles')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(positions[:N_protons, 0], positions[:N_protons, 1], s=10, c='red', label='Protons')
if N_neutrons > 0:
    plt.scatter(positions[N_protons:N_protons+N_neutrons, 0], positions[N_protons:N_protons+N_neutrons, 1], s=10, c='green', label='Neutrons')
plt.scatter(positions[-N_electrons:, 0], positions[-N_electrons:, 1], s=10, c='blue', label='Electrons')
plt.xlim(0, L)
plt.ylim(0, L)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Final positions of subatomic particles')
plt.legend()

plt.show()
