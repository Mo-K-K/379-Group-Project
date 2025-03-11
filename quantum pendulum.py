import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import fft, ifft, fftfreq

m = 1.0
length = 1.0
g = 9.81
h_bar = 1.0
N = 100000
dt = 0.0001
total_time = 10

theta = np.linspace(-np.pi, np.pi, N, endpoint=False)
d_theta = theta[1] - theta[0]
L_vals = fftfreq(N, d=d_theta) * 2 * np.pi
T_evol = np.exp(-1j * dt * (L_vals ** 2 / (2 * m * length ** 2)) / h_bar)
V_evol = np.exp(-1j * dt * (-g * m * length * np.cos(theta)) / h_bar)
A_1, A_2, B_1, B_2 = 1, 1, 1, 1
k_1, k_2, j_1, j_2 = 1, 1, 1, 1

psi = A_1 * np.cos(k_1*theta) + B_1 * np.sin((2*j_1 - 1)*theta*0.5) + A_2 * np.cos(k_2*theta) + B_2 * np.sin((2*j_2 - 1)*theta*0.5)

norm = np.sqrt(np.sum(np.abs(psi) ** 2) * d_theta)
psi /= norm
num_steps = int(total_time / dt)

probability_density_list = []

for i in range(num_steps):
    psi = fft(psi)
    psi *= T_evol
    psi = ifft(psi)
    psi *= V_evol

    if i % 100 == 0:
        probability_density_list.append(np.abs(psi)**2)

# Slow down playback by repeating frames
slowdown_factor = 4  # Increase this value to slow down further
extended_prob_density_list = []
for frame in probability_density_list:
    extended_prob_density_list.extend([frame] * slowdown_factor)

# Animation
fig, ax = plt.subplots()
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(0, np.max(probability_density_list))
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$|\psi(\theta)|^2$')
ax.set_title('Probability Density Evolution')
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(theta, extended_prob_density_list[frame])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(extended_prob_density_list), init_func=init, blit=True, interval=1000/60)
ani.save('probability_density_evolution.gif', writer='pillow', fps=60)
plt.show()

#plt.plot(theta, np.abs(psi) ** 2)
#plt.xlabel(r'$\theta$')
#plt.ylabel(r'$|\psi(\theta)|^2$')
#plt.title('Final Probability Density')
#plt.show()
