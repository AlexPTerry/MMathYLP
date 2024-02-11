import numpy as np
import numpy.linalg as la

def kinetic_energy(p, m):
    # Calculate the squared 2-norm of each momentum vector, then sum with mass consideration
    ke = np.sum(np.sum(p**2, axis=0) / (2 * m))  # axis=0 sums across rows for each column vector
    return ke

def potential_energy(q, f_c):
    energy = 0
    k = q.shape[1]  # Number of particles
    for i in range(k):
        for j in range(i + 1, k):
            r_ij = np.linalg.norm(q[:, i] - q[:, j])
            energy += f_c(r_ij)
    return energy

def v_t(q, t, e_t):
    # Assuming e(t) is a scalar or a vector function that properly broadcasts over q's dimensions
    return np.sum(e_t(t) * q)

# def f_c(x):
#     return 1/x # Straightforward to differentiate

# def df_c(x):
#     return -1/x**2

def f_c(x):
    return -1/x**2 # Straightforward to differentiate

def df_c(x):
    return 2/x**3

def e(t):
    return 0 # Consider conservative system initially

def H(p, q, m, f_c, e_t, t):
    return kinetic_energy(p, m) + potential_energy(q, f_c) + v_t(q, t, e_t)

def dH_dp_i(p, m, i):
    return p[:, i] / m[i]

def dH_dq_i(q, m, i):
    d, k = q.shape
    total = np.zeros(d)
    for j in range(k):
        if i == j:
            continue
        x = q[:, i] - q[:, j]
        norm_x = la.norm(x)
        total += x/norm_x * df_c(norm_x) * m[i] * m[j]
    return total

def dH_dp(p, m):
    d, k = p.shape
    diff_p = np.zeros_like(p)
    for i in range(k):
        diff_p[:, i] = dH_dp_i(p, m, i)
    return diff_p

def dH_dq(q, m):
    d, k = q.shape
    diff_q = np.zeros_like(q)
    for i in range(k):
        diff_q[:, i] = dH_dq_i(q, m, i)
    return diff_q
    

def leapfrog_integration(q0, p0, m, dt, total_time):
    """
    Perform leapfrog integration.
    """
    steps = int(total_time / dt)
    q, p = q0.copy(), p0.copy()
    

    # Initialize arrays to store trajectories for plotting
    trajectories = np.zeros((q.shape[0], q.shape[1], steps + 1))
    trajectories[:, :, 0] = q
    
    momentums = np.zeros((p.shape[0], p.shape[1], steps + 1))
    momentums[:, :, 0] = p
    
    for step in range(steps):
        p_half = p - 0.5 * dt * dH_dq(q, m)
        q = q + dt * dH_dp(p_half, m)
        p = p_half - 0.5 * dt * dH_dq(q, m)
        
        # Store positions for plotting
        trajectories[:, :, step + 1] = q
        momentums[:, :, step + 1] = p
    
    return trajectories, momentums

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def plot_trajectories(trajectories):
    """
    Plot the trajectories of particles in a 2D system.
    """
    fig, ax = plt.subplots()
    k = trajectories.shape[1]  # Number of particles
    for i in range(k):
        ax.plot(trajectories[0, i, :], trajectories[1, i, :], label=f'Particle {i+1}')
    ax.legend()
    plt.show()


m = 1.05*np.array([1, 1])  # Mass of orbiting body
q0 = 0.5*np.array([[-1, 0], [1, 0]]).T  # Initial position (right on the x-axis, at unit distance from origin)
v0 = np.array([[0, -1], [0, 1]]).T  # Initial velocity (perpendicular to the radius for circular orbit)

# m = np.array([1, 1, 1])  # Mass of orbiting body
# q0 = 0.2*np.array([[-1, 0], [1, 0], [0, 0]]).T  # Initial position (right on the x-axis, at unit distance from origin)
# v0 = 10*np.array([[0, -1], [0, 1], [0, 0]]).T  # Initial velocity (perpendicular to the radius for circular orbit)

# m = np.array([200, 1])  # Mass of orbiting body
# q0 = 0.5*np.array([[0, 0], [3, 0]]).T  # Initial position (right on the x-axis, at unit distance from origin)
# v0 = np.array([[0, 0], [0, 10.8]]).T  # Initial velocity (perpendicular to the radius for circular orbit)

# Convert velocity to momentum
p0 = m * v0


# Time step and total simulation time
dt = 0.001
total_time = 2*np.pi  # One orbital period, simplified

trajectories, momentums = leapfrog_integration(q0, p0, m, dt, total_time)




fig, ax = plt.subplots()
lines = []
for _ in range(trajectories.shape[1]):  # Create a line for each particle
    line, = ax.plot([], [], 'o-', lw=2)
    lines.append(line)

ax.set_xlim(np.min(trajectories[0, :, :]), np.max(trajectories[0, :, :]))
ax.set_ylim(np.min(trajectories[1, :, :]), np.max(trajectories[1, :, :]))
ax.set_xlabel('x')
ax.set_ylabel('y')

trajectories = trajectories[:,:,::10]

# def init():
#     for line in lines:
#         line.set_data([], [])
#     return lines

# def animate(i):
#     for j, line in enumerate(lines):
#         line.set_data(trajectories[0, j, :i+1], trajectories[1, j, :i+1])
#     return lines
tail_length = 200  

def init():
    ax.clear()
    ax.set_xlim(np.min(trajectories[0, :, :]), np.max(trajectories[0, :, :]))
    ax.set_ylim(np.min(trajectories[1, :, :]), np.max(trajectories[1, :, :]))
    return ax,

def animate(i):
    ax.clear()
    ax.set_xlim(np.min(trajectories[0, :, :]), np.max(trajectories[0, :, :]))
    ax.set_ylim(np.min(trajectories[1, :, :]), np.max(trajectories[1, :, :]))
    for j in range(0, trajectories.shape[1], 1):
        # Determine the start index for the tail
        start_idx = max(i - tail_length, 0)
        # Compute fading alphas
        alphas = np.linspace(0.1, 1, i - start_idx + 1)
        for k, alpha in zip(range(start_idx, i + 1, 1), alphas):
            ax.plot(trajectories[0, j, k:k+2], trajectories[1, j, k:k+2], 'o-', lw=2, alpha=alpha, color=f"C{j}")
    return ax,


ani = FuncAnimation(fig, animate, frames=trajectories.shape[2], init_func=init, blit=True, interval=20)

plt.show()