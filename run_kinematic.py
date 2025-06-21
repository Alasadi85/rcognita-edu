import numpy as np
import matplotlib.pyplot as plt

# Kinematic Controller
def kinematic_controller(x, goal, k_rho=1.0, k_alpha=3.0, k_beta=-1.0):
    dx = goal[0] - x[0]
    dy = goal[1] - x[1]
    rho = np.sqrt(dx**2 + dy**2)
    alpha = -x[2] + np.arctan2(dy, dx)
    beta = -x[2] - alpha
    v = k_rho * rho
    omega = k_alpha * alpha + k_beta * beta
    return np.array([v, omega])

# Initialize
x = np.array([0.0, 0.0, 0.0])
goal = np.array([2.0, 2.0])
trajectory = [x.copy()]
errors = []

# Simulate
for t in range(100):
    u = kinematic_controller(x, goal)
    dt = 0.1
    x[0] += u[0] * np.cos(x[2]) * dt
    x[1] += u[0] * np.sin(x[2]) * dt
    x[2] += u[1] * dt
    trajectory.append(x.copy())
    errors.append(np.linalg.norm(goal - x[:2]))

trajectory = np.array(trajectory)

# Plot trajectory
plt.figure()
plt.plot(trajectory[:,0], trajectory[:,1], label="Path")
plt.plot(goal[0], goal[1], 'rx', label="Goal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Kinematic Trajectory")
plt.legend()
plt.grid()
plt.savefig("trajectory_kinematic.png")

# Plot error
plt.figure()
plt.plot(errors)
plt.title("Tracking Error")
plt.xlabel("Time Step")
plt.ylabel("Error")
plt.grid()
plt.savefig("error_kinematic.png")

print("Simulation finished. Plots saved.")
