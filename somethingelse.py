import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import quad, nquad
from multiprocessing import Pool, cpu_count, Manager
import os
import time

# Constants
hbar = 1
I1, I2 = 1, 1  # Moments of inertia
g = 1  # Gravity term
N = 5 # Number of basis states per rotor (grid size for theta)
Np = 5  # Momentum grid size

# Define angular and momentum grids
theta_vals = np.linspace(-np.pi, np.pi, N, endpoint=False)
p_vals = np.linspace(-5, 5, Np)  # Momentum range

ground_state_array = None  # Global storage for wavefunction values

# Construct the Hamiltonian matrix
def construct_hamiltonian(N):
    """Constructs the Hamiltonian matrix for the quantum double pendulum."""
    size = N * N  # Total basis size
    H = np.zeros((size, size))

    # Define basis indices (m1, m2) for rotor states
    for m1 in range(N):
        for m2 in range(N):
            idx = m1 * N + m2  # Flattened index
            
            # Kinetic energy terms
            H[idx, idx] = (m1**2 / (2 * I1)) + (m2**2 / (2 * I2))
            
            # Potential energy terms (cosine terms in matrix form)
            if m1 > 0:
                H[idx, (m1 - 1) * N + m2] += -g * I1 / 2  # Cos(theta1)
                H[(m1 - 1) * N + m2, idx] += -g * I1 / 2
            if m2 > 0:
                H[idx, m1 * N + (m2 - 1)] += -g * I2 / 2  # Cos(theta2)
                H[m1 * N + (m2 - 1), idx] += -g * I2 / 2

    return H

# Diagonalize the Hamiltonian to get eigenvalues and eigenstates
def diagonalize_hamiltonian(N):
    H = construct_hamiltonian(N)
    eigenvalues, eigenvectors = eigh(H)  # Solve H|psi> = E|psi>
    return eigenvalues, eigenvectors

# Precompute the ground state wavefunction values
def precompute_ground_state(eigenvectors, shared_dict):
    """Compute and store the ground state wavefunction in a shared dictionary."""
    print("Type of eigenvectors:", type(eigenvectors))
    print("Shape of eigenvectors:", np.shape(eigenvectors))  # Should be (N, N)
    eigenvectors = np.array(eigenvectors)  # Convert to NumPy array if needed
    ground_state = eigenvectors[:, 0]  # Take the first eigenvector (lowest energy state)
    #print(ground_state)
    shared_dict["ground_state"] = ground_state.reshape((N, N))

def wigner_function_point(args):
    """Compute a single point of the Wigner function using the shared ground state."""
    theta1, theta2, p1, p2, shared_dict = args  # Unpack arguments
    ground_state_array = shared_dict["ground_state"]  # Retrieve shared memory variable
    
    if ground_state_array is None:
        raise ValueError("Ground state wavefunction not initialized in subprocess!")
    
    # Compute Wigner function (Modify this based on your exact implementation)
    def integrand(xi1, xi2):
        i1 = np.searchsorted(theta_vals, theta1 - xi1 / 2) % N
        j1 = np.searchsorted(theta_vals, theta2 - xi2 / 2) % N
        i2 = np.searchsorted(theta_vals, theta1 + xi1 / 2) % N
        j2 = np.searchsorted(theta_vals, theta2 + xi2 / 2) % N
        
        psi1 = ground_state_array[i1, j1]
        psi2 = np.conj(ground_state_array[i2, j2])
        return np.exp(2j * (p1 * xi1 + p2 * xi2) / hbar) * psi1 * psi2

    integral, _ = nquad(lambda xi1, xi2: integrand(xi1, xi2).real, [[-4, 4], [-4, 4]])


    return (theta1, p1, theta2, p2, (1 / (np.pi**2 * hbar**2)) * integral)


def compute_wigner_parallel(eigenvectors):
    """Parallel computation of the Wigner function."""
    with Manager() as manager:
        shared_dict = manager.dict()  # Create shared memory space

        eigenvectors = np.array(eigenvectors)  # Convert to NumPy array if needed
        ground_state = eigenvectors[:, 0]  # Take the first eigenvector (lowest energy state)
        #print(ground_state)
        shared_dict["ground_state"] = ground_state.reshape((N, N))
        
        # Debugging check
        if "ground_state" not in shared_dict:
            raise ValueError("Ground state was not successfully stored in shared dictionary!")

        args_list = [(theta1, theta2, p1, p2, shared_dict) for theta1 in theta_vals for theta2 in theta_vals for p1 in p_vals for p2 in p_vals]
        
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(wigner_function_point, args_list)
            
    
    wigner_values_only = np.array([r[4] for r in results])  # Extract the Wigner function values
    print(f"Extracted Wigner values shape: {wigner_values_only.shape}, Expected: {N**4}")  # Debug

    if len(wigner_values_only) == N**4:
        return wigner_values_only.reshape((N, N, N, N))
    else:
        print(f"Mismatch in expected size: {len(wigner_values_only)} instead of {N**4}.")
        return wigner_values_only  # Return without reshaping for debugging

# Ensure this block runs only in the main script
if __name__ == "__main__":
    """

    os.environ['OMP_NUM_THREADS'] = '1'  # Prevents multi-threading issues

    with Manager() as manager:
        shared_dict = manager.dict()  # Create shared memory object

        # Step 2: Compute and store the ground state in shared_dict
        eigenvalues, eigenvectors = diagonalize_hamiltonian(N)  # Get eigenvalues & eigenvectors

        # Step 3: Compute Wigner function in parallel
        wigner_vals = compute_wigner_parallel(eigenvectors)
    """

    os.environ['OMP_NUM_THREADS'] = '1'  # Prevents multi-threading issues

    start_total = time.time()  # Start total timer

    with Manager() as manager:
        shared_dict = manager.dict()  # Create shared memory object

        print("Diagonalizing Hamiltonian...")
        start = time.time()
        eigenvalues, eigenvectors = diagonalize_hamiltonian(N)  # Get eigenvalues & eigenvectors
        end = time.time()
        print(f"Hamiltonian diagonalized in {end - start:.2f} seconds")

        print("Computing Wigner function in parallel...")
        start = time.time()
        wigner_vals = compute_wigner_parallel(eigenvectors)
        end = time.time()
        print(f"Wigner function computed in {end - start:.2f} seconds")

    end_total = time.time()
    print(f"Total execution time: {end_total - start_total:.2f} seconds")

    # Select fixed p2 values and plot Wigner function for different theta2
    fig, axes = plt.subplots(2, 2, figsize=(12,10))

    theta2_slices = [0, np.pi/2, np.pi, 3*np.pi/2]  # Different values of theta2
    p2_fixed = 0  # Fix p2 at 0

    for ax, theta2 in zip(axes.flat, theta2_slices):
        k = np.searchsorted(theta_vals, theta2) % N  # Find index for theta2
        l = np.searchsorted(p_vals, p2_fixed) % Np  # Find index for p2

        # 2D Contour plot for Wigner function at fixed (theta2, p2)
        cp = ax.contourf(theta_vals, p_vals, wigner_vals[:, :, k, l].T, levels=100, cmap='RdBu')
        fig.colorbar(cp, ax=ax)
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel(r'$p_1$')
        ax.set_title(rf'Wigner Function at $\theta_2 = {theta2:.2f}, p_2 = {p2_fixed}$')

    plt.tight_layout()
    plt.show()
