import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

m = 1.0
length = 1.0
g = 9.81
h_bar = 1
N = 10000

theta_vals = np.linspace(-np.pi, np.pi, N, endpoint=False)
L_vals = np.linspace(-np.pi, np.pi, N, endpoint=False)
phi_vals = np.linspace(-np.pi, np.pi, N, endpoint=False)


def psi(theta):
    A_1, A_2, B_1, B_2 = 1, 1, 1, 1
    k_1, k_2, j_1, j_2 = 1, 1, 1, 1

    return A_1 * np.cos(k_1 * theta) + B_1 * np.sin((2 * j_1 - 1) * theta * 0.5) + A_2 * np.cos(
        k_2 * theta) + B_2 * np.sin((2 * j_2 - 1) * theta * 0.5)


def psi_mod(x):
    x_mod = ((x + np.pi) % (2 * np.pi)) - np.pi
    return psi(x_mod)


W = np.zeros((N, N), dtype=complex)

start_time = time.time()

for i, th in enumerate(tqdm(theta_vals, desc="Computing Wigner function", unit="row")):
    psi_product = psi_mod(th + phi_vals) * psi_mod(th - phi_vals)
    phase_factor = np.exp(-2j * np.outer(L_vals, phi_vals))
    integrand = phase_factor * psi_product
    integral = np.trapz(integrand, phi_vals, axis=1)
    W[i, :] = (1 / np.pi) * integral

elapsed_time = time.time() - start_time
print(f"Computation completed in {elapsed_time:.2f} seconds.")

W = np.real(W)

# Plot and save the Wigner function
plt.figure(figsize=(8, 6))
plt.imshow(W.T, extent=[-np.pi, np.pi, -np.pi, np.pi],
           aspect='auto', origin='lower', cmap='RdBu_r')
plt.xlabel(r'$\theta$')
plt.ylabel('L')
plt.title('Wigner Function Phase Space')
plt.colorbar(label=r'$W(\theta, L)$')
plt.savefig("wigner_function.png", dpi=300, bbox_inches='tight')
print("Plot saved as 'wigner_function.png'.")
plt.show()
