import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from multiprocessing import Pool, cpu_count, Manager
import os
import time

# Constants
hbar = 1
I1, I2 = 1, 1  # Moments of inertia
g = 5  # Gravity term
N = 30 # Number of basis states per rotor (grid size for theta)
Np = 30  # Momentum grid size

# Define angular and momentum grids
theta_vals = np.linspace(-np.pi, np.pi, N, endpoint=False)
p_vals = np.linspace(-5, 5, Np)  # Momentum range

def coherent_state(theta1, theta2, p1, p2, theta_vals, ground_state):
    """Compute the overlap of the ground state with a Gaussian coherent state."""
    sigma = np.sqrt(hbar / 2)  # Width of the coherent state
    
    # Compute Gaussian wave packet for both theta1 and theta2
    gaussian1 = np.exp(-((theta_vals - theta1) ** 2) / (2 * sigma**2))
    gaussian2 = np.exp(-((theta_vals - theta2) ** 2) / (2 * sigma**2))
    
    # Create 2D Gaussian
    gaussian_2d = np.outer(gaussian1, gaussian2)
    norm_factor = np.sqrt(np.sum(gaussian_2d**2))
    
    # Compute overlap with ground state
    coherent_overlap = np.abs(np.sum(ground_state * gaussian_2d)) / norm_factor
    return coherent_overlap**2 / (np.pi * hbar)

def husimi_function_point(args):
    """Compute a single point of the Husimi function."""
    theta1, theta2, p1, p2, shared_dict = args
    ground_state_array = shared_dict["ground_state"]
    
    if ground_state_array is None:
        raise ValueError("Ground state wavefunction not initialized in subprocess!")
    
    return (theta1, theta2, p1, p2, coherent_state(theta1, theta2, p1, p2, theta_vals, ground_state_array))

def compute_husimi_parallel(eigenvectors):
    """Parallel computation of the Husimi function."""
    with Manager() as manager:
        shared_dict = manager.dict()
        ground_state = np.array(eigenvectors[:, 0]).reshape((N, N))  # Reshape to (N, N)
        shared_dict["ground_state"] = ground_state
        
        args_list = [(theta1, theta2, p1, p2, shared_dict) for theta1 in theta_vals for theta2 in theta_vals for p1 in p_vals for p2 in p_vals]
        
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(husimi_function_point, args_list)
    
    husimi_values = np.array([r[4] for r in results]).reshape((N, N, Np, Np))
    return husimi_values

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'  # Prevents multi-threading issues
    
    print("Diagonalizing Hamiltonian...")
    start = time.time()
    eigenvalues, eigenvectors = eigh(np.random.rand(N*N, N*N))  # Placeholder Hamiltonian diagonalization
    end = time.time()
    print(f"Hamiltonian diagonalized in {end - start:.2f} seconds")
    
    print("Computing Husimi function in parallel...")
    start = time.time()
    husimi_vals = compute_husimi_parallel(eigenvectors)
    end = time.time()
    print(f"Husimi function computed in {end - start:.2f} seconds")
    
    # Plot the Husimi function
    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    theta2_slices = [0, np.pi/2, np.pi, 3*np.pi/2]
    p2_fixed = 0  # Fix p2 at 0

    for ax, theta2 in zip(axes.flat, theta2_slices):
        k = np.searchsorted(theta_vals, theta2) % N
        l = np.searchsorted(p_vals, p2_fixed) % Np
        
        cp = ax.contourf(theta_vals, p_vals, husimi_vals[:, :, k, l].T, levels=100, cmap='viridis')
        fig.colorbar(cp, ax=ax)
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel(r'$p_1$')
        ax.set_title(rf'Husimi Function at $\theta_2 = {theta2:.2f}, p_2 = {p2_fixed}$')
    
    plt.tight_layout()
    plt.show()