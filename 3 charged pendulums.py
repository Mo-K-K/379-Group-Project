import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import *
from matplotlib.animation import FFMpegWriter

d = 4 # dist. between 
L = 1 # Assuming L is the length of rod of each pendulum
k = 1 # 4πɛ₀ (divides Q Q)
g = 10 # grav const. NOTE: is g positive or negative?
m = np.array([1,1,2]) # masses
Q = 1*np.array([1,0,1]) # charges
theta = np.array([pi/2,0,-pi/2]) # angles (θᵢ)
omega = np.array([0,0,0]) # angular (ωᵢ)
dt = 0.001 # time step
t = 0 # initial time - editing this does not "skip" time
T = 10 # end time (T/FPS)

frames = T/dt
print(f"Steps: {frames}")
FPS = 30
writer = FFMpegWriter(fps=FPS)

C = -d*np.array([[ 0,  1,  2], # NOTE: MAKE SURE TO CHECK THAT WE HAVE THE CORRECT CONVENTION HERE
                [-1,  0,  1], # AS IN: MAKE SURE THAT WE HAVE C[i,j] = C_ij and not = C_ji
                [-2, -1,  0]]) # OK ACTUALLY REMINDER, WE HAVE C[i-1,j-1] = C_ij, since A[0] is 1st entry

xn = theta # translating for ease of use in code
yn = omega # ^^ NOTE: Ben is stinky

# RK4 setup 
fa = lambda y: y
def fb(xn, yn, t, dt=dt, Q=Q, L=L, g=g):

    d2theta = np.zeros(3) # d²θᵢ/dt² aka dω/dt
    # xN = xn

    for i in [0,1,2]:
        for j in [0,1,2]:
            if j != i:
                '''EULER-LAGRANGE EQUATION OF MOTION'''
                d2theta[i] += 2*L*( L - L*cos(xn[i]-xn[j]) + C[i,j]*( sin(xn[j]) - sin(xn[i]) + C[i,j]/2*L ) ) # Step 1
                
                # ANTI EXPLOSION LINES
                d2theta[i] = 0 if isinf(d2theta[i]) else d2theta[i] # NOTE: removes inf errors but assumes that Q = 0
                if d2theta[i] <= 0:   # accounts for floating point errors where d2theta = -0
                    # d2theta[i] = 0
                    d2theta = abs(d2theta)

                d2theta[i] **= (-3/2) # Step 2

                d2theta[i] *= (1/2)*( Q[i]*Q[j]*k ) # Step 3 NOTE: changed -1/2 to 1/2

                d2theta[i] *= 2*L*( L*sin( xn[i] - xn[j] ) - C[i,j]*cos(xn[i]) ) # Step 4

                d2theta[i] -= g*L*m[i]*sin(xn[i]) # Step 5
                # print(d2theta)
                
    d2theta /= 2
    return d2theta

# RK4 algorithm
def RK4(xn,yn,t,dt=dt): 
    ka1 = fa(yn)
    kb1 = fb(xn, yn, t)

    ka2 = fa(yn + kb1*dt/2)
    kb2 = fb(xn + ka1*dt/2, yn + kb1*dt/2, t + dt/2)

    ka3 = fa(yn + kb2*dt/2)
    kb3 = fb(xn + ka2*dt/2, yn + kb2*dt/2, t + dt/2)

    ka4 = fa(yn + kb3*dt)
    kb4 = fb(xn + ka3*dt, yn + kb3*dt, t + dt)

    xN = xn + (dt/6)*(ka1 + 2*ka2 + 2*ka3 + ka4)
    yN = yn + (dt/6)*(kb1 + 2*kb2 + 2*kb3 + kb4)
    return xN,yN,t

def normalize_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi  # Keeps within (-π, π]

# def energy(xn, yn, m=m, L=L, g=g): # NOTE: NOT INCLUDING ELECTRIC POTENTIAL!!!
#     for i in range(len(xn)):
#         T = 0.5 * m[i] * L**2 * yn[i]**2  # Kinetic energy for each pendulum
#         U = m[i] * g * L * np.cos(xn[i])  # Potential energy for each pendulum

#     E = T + U  # Total energy   
#     return E

print("Processing...")

# Loop n -> N
# Initialize with the first set of values
M = [[xn, yn, t]]
while t < T - dt:
    xn, yn, t = RK4(xn, yn, t)  # Use the previous state for the RK4 step
    t += dt  # Update time
    xn = normalize_angle(xn)  # Normalize the angle if needed
    M.append([xn, yn, t])  # Store the updated values


# print([E[0],E[-1]])

'''RENDERING'''
###########################################################################

D = []
S = 900
step = frames / S

for i in range(S):
    frame_index = round(i * step)  # Pick evenly spaced frames
    D.append(M[frame_index])


X, Y, T = zip(*D)  # Unpack into separate lists
X = np.array([np.array(xi) for xi in X]).T  # Convert to (3, N) array
Y = np.array([np.array(yi) for yi in Y]).T  # Convert to (3, N) array
T = np.array(T)  # Convert to (N,) array

# print(Y) 

print("Rendering...")
# Set up the figure
fig, ax = plt.subplots()

ax.set_xlim(-1*pi,pi)
ax.set_xlim(xmin=X.min(),xmax=X.max())
ax.set_ylim(ymin=Y.min(),ymax=Y.max())

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Animated Phase Space')

# Create separate lines for each pendulum
lines = [ax.plot([], [], '-', lw=2, label=f'Pendulum {i+1}')[0] for i in range(3)]
points = [ax.plot([], [], 'o')[0] for _ in range(3)]  # Moving points

ax.legend()

# Initialize function
def init():
    for line, point in zip(lines, points):
        line.set_data([], [])
        point.set_data([], [])
    return lines + points

def update(frame):
    for i in range(3):  # Iterate over the three particles
        x_trail = X[i, :frame+1]
        y_trail = Y[i, :frame+1]

        # Compute finite differences
        x_diff = np.diff(x_trail)

        # Detect only the discontinuities at ±π
        discontinuities = np.where(np.abs(x_diff) > np.pi - 0.1)[0]  # Slight tolerance to catch jumps

        x_trail = x_trail.astype(float)

        # Insert NaNs where π to -π jumps occur
        for idx in discontinuities:
            x_trail[idx] = np.nan

        lines[i].set_data(x_trail, y_trail)  # Update trail
        points[i].set_data([X[i, frame]], [Y[i, frame]])  # Update moving point

    return lines + points

# Animate
ani = animation.FuncAnimation(fig, update, frames=len(T), init_func=init, blit=True, interval=dt/1000, repeat=True)
# ani.save("animation.mp4", writer=writer)

plt.show()
print("Done.")
