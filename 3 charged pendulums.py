import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import *
from matplotlib.animation import PillowWriter
from tqdm import tqdm
import sys

d = 3 # dist. between 
L = 1 # Assuming L is the length of rod of each pendulum
k = 1 # 1/4πɛ₀
g = 10 # grav const. NOTE: g > 0
m = np.array([1,1,1]) # masses
Q = 10*np.array([1,1,-1]) # charges
theta = np.array([pi/2,0,-pi/2]) # angles (θᵢ)
omega = np.array([0,0,0]) # angular (ωᵢ)

dt = 0.001 # time step
t = 0 # initial time - editing this does not "skip" time
T = 10 # end time (T/FPS)

frames = int(T/dt)
FPS = 30
aniTime = 30

fa = lambda y: y
def fb(theta):
    d2theta = np.zeros(3)
    for i in range(3):  
        A = 0
        for j in range(3):
            if j != i:
                u = L*( sin(theta[j]) - sin(theta[i])) + d*(j-i)
                du = -L*cos(theta[i])

                v = L*( cos(theta[i]) - cos(theta[j]) ) # GOOFY AHHH LINE
                dv = -L*sin(theta[i])

                r2 = u**2 + v**2
                rn32 = r2**(-3/2)
                d1r = (-1/2)*( 2*u*du + 2*v*dv )*rn32

                A += k*Q[i]*Q[j]*d1r

        B = A - m[i]*g*L*sin(theta[i])
        d2theta[i] = B/(m[i]*(L**2))
    return d2theta

# RK4 algorithm
def RK4(xn,yn,t,dt=dt): 
    ka1 = fa(yn)
    kb1 = fb(xn)

    ka2 = fa(yn + kb1*dt/2)
    kb2 = fb(xn + ka1*dt/2)

    ka3 = fa(yn + kb2*dt/2)
    kb3 = fb(xn + ka2*dt/2)

    ka4 = fa(yn + kb3*dt)
    kb4 = fb(xn + ka3*dt)

    xN = xn + (dt/6)*(ka1 + 2*ka2 + 2*ka3 + ka4)
    yN = yn + (dt/6)*(kb1 + 2*kb2 + 2*kb3 + kb4)
    return xN, yN, t+dt

def normalize_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi  # Keeps within (-π, π]

pbar = tqdm(total=frames, desc="Processing")

# Loop n -> N
M = [[theta, omega, t]] 
while t < T - dt:
    theta, omega, t = RK4(theta, omega, t)  # Use the previous state for the RK4 step
    theta = normalize_angle(theta)  # Normalize the angle if needed
    M.append([theta, omega, t])  # Store the updated values
    pbar.update(1)

pbar.close()
sys.stdout.flush()







'''RENDERING'''
###########################################################################

D = []
S = aniTime*FPS
step = frames / S

for i in range(S):
    frame_index = round(i * step)  # Pick evenly spaced frames
    D.append(M[frame_index])


X, Y, T = zip(*D)  # Unpack into separate lists
X = np.array([np.array(xi) for xi in X]).T  # Convert to (3, N) array
Y = np.array([np.array(yi) for yi in Y]).T  # Convert to (3, N) array
T = np.array(T)  # Convert to (N,) array

for i in range(3):
    x_diff = np.diff(X[i])
    discontinuities = np.where(np.abs(x_diff) > np.pi - 0.1)[0]
    X[i] = X[i].astype(float)
    for idx in discontinuities:
        X[i][idx] = np.nan
# Plot
plt.figure( )
plt.xlim(-pi,pi)
for i in range(3):
    plt.plot(X[i], Y[i], label=f'Pendulum {i+1}')
plt.xlabel('θ')
plt.ylabel('ω')
plt.title('Phase Space of Three Charged Pendulums')
plt.legend()
plt.grid()
plt.savefig(f"images/2.png", dpi=300, bbox_inches='tight')
plt.show()


# print("Rendering...")
# # Set up the figure
# fig, ax = plt.subplots()
# ax.grid(True)

# ax.set_xlim(-1*pi,pi)

# if isinf(Y.max()) or isnan(Y.max()):
#     ax.set_ylim(ymin=-50,ymax=50)
# else:
#     ax.set_ylim(ymin=Y.min(),ymax=Y.max())
#     ax.set_xlim(xmin=X.min(),xmax=X.max())

# ax.set_xlabel('θ')
# ax.set_ylabel('ω')
# ax.set_title('Animated Phase Space')

# # Create separate lines for each pendulum
# lines = [ax.plot([], [], '-', lw=2, label=f'Pendulum {i+1}')[0] for i in range(3)]
# points = [ax.plot([], [], 'o')[0] for _ in range(3)]  # Moving points

# ax.legend()

# # Initialize function
# def init():
#     for line, point in zip(lines, points):
#         line.set_data([], [])
#         point.set_data([], [])
#     return lines + points

# def update(frame):
#     for i in range(3):  # Iterate over the three particles
#         x_trail = X[i, :frame+1]
#         y_trail = Y[i, :frame+1]

#         # Compute finite differences
#         x_diff = np.diff(x_trail)

#         # Detect only the discontinuities at ±π
#         discontinuities = np.where(np.abs(x_diff) > np.pi - 0.1)[0]  # Slight tolerance to catch jumps

#         x_trail = x_trail.astype(float)

#         # Insert NaNs where π to -π jumps occur
#         for idx in discontinuities:
#             x_trail[idx] = np.nan

#         lines[i].set_data(x_trail, y_trail)  # Update trail
#         points[i].set_data([X[i, frame]], [Y[i, frame]])  # Update moving point

#     return lines + points

# # Animate
# ani = animation.FuncAnimation(fig, update, frames=len(T), init_func=init, blit=True, interval=dt/1000, repeat=True)
# # ani.save(r"animations/test5.gif", writer=PillowWriter(fps=FPS))

# plt.show()
# # print("Done.")
