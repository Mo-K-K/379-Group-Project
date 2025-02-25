import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

m = 1.0
length = 1.0
g = 9.81
h_bar = 1.0
N = 10000
dt = 0.001
total_time = 10

theta = np.linspace(-np.pi, np.pi, N, endpoint=False)
d_theta = theta[1] - theta[0]
L_vals = fftfreq(N, d=d_theta) * 2 * np.pi
T_evol = np.exp(-1j * dt * (L_vals ** 2 / (2 * m * length ** 2)) / h_bar)
V_evol = np.exp(-1j * dt * (-g * m * length * np.cos(theta)) / h_bar)
psi = np.exp(-0.5 * (theta / 0.5) ** 2)

norm = np.sqrt(np.sum(np.abs(psi) ** 2) * d_theta)
psi /= norm
num_steps = int(total_time / dt)

for i in range(num_steps):
    psi = fft(psi)
    psi *= T_evol
    psi = ifft(psi)
    psi *= V_evol

plt.plot(theta, np.abs(psi) ** 2)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$|\psi(\theta)|^2$')
plt.title('Final Probability Density')
plt.show()
