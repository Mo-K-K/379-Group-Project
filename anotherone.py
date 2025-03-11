import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from multiprocessing import Pool, cpu_count

# Constants
hbar = 1
N = 10  # Reduced grid size for angles
Np = 10  # Reduced grid size for momenta

# Define angular and momentum grids
theta_vals = np.linspace(0, 2*np.pi, N, endpoint=False)
p_vals = np.linspace(-5, 5, Np)  # Momentum range

# Define the trial wavefunction
def psi_trial(m1, m2, theta1, theta2):
    """Trial wavefunction for the quantum double pendulum."""
    return (1 / (2 * np.pi)) * np.exp(1j * m1 * theta1) * np.exp(1j * m2 * theta2)

# Compute the Wigner function for a single point
def wigner_function_point(args):
    """Computes the Wigner function for a given (theta1, p1, theta2, p2) pair."""
    m1, m2, theta1, p1, theta2, p2 = args
    
    def integrand(xi1, xi2):
        psi1 = psi_trial(m1, m2, theta1 - xi1 / 2, theta2 - xi2 / 2)
        psi2 = np.conj(psi_trial(m1, m2, theta1 + xi1 / 2, theta2 + xi2 / 2))
        return np.exp(2j * (p1 * xi1 + p2 * xi2) / hbar) * psi1 * psi2

    integral, _ = quad(lambda xi1: quad(lambda xi2: integrand(xi1, xi2).real, -5, 5)[0], -5, 5)
    return (theta1, p1, theta2, p2, (1 / (np.pi**2 * hbar**2)) * integral)

# Use parallel processing to compute the Wigner function
def compute_wigner_parallel(m1, m2):
    print("computing tings")
    args_list = [(m1, m2, theta1, p1, theta2, p2) 
                 for theta1 in theta_vals for p1 in p_vals
                 for theta2 in theta_vals for p2 in p_vals]
    
    print("arglist done")

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(wigner_function_point, args_list)
    

    print("starting for loop")
    # Convert results to an array
    wigner_vals = np.zeros((N, Np, N, Np))
    for theta1, p1, theta2, p2, value in results:
        i = np.searchsorted(theta_vals, theta1) % N
        j = np.searchsorted(p_vals, p1) % Np
        k = np.searchsorted(theta_vals, theta2) % N
        l = np.searchsorted(p_vals, p2) % Np
        wigner_vals[i, j, k, l] = value
    return wigner_vals

# Ensure this block runs only in the main script
if __name__ == "__main__":
    # Run computation for m1 = m2 = 1 using parallel processing
    m1, m2 = 1, 1
    wigner_vals = compute_wigner_parallel(m1, m2)

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
