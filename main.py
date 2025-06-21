import numpy as np
import matplotlib.pyplot as plt
from controllers import N_CTRL

# Simulation parameters
dt = 0.1
T = 20
steps = int(T / dt)

# Initial pose: [x, y, theta]
pose = np.array([2.0, 2.0, np.pi / 2])
goal = np.array([0.0, 0.0])

# Controller
controller = N_CTRL()

# Logs
trajectory = [pose.copy()]
vs = []
ws = []

for _ in range(steps):
    v, w = controller.pure_loop(pose)
    vs.append(v)
    ws.append(w)

    # Update pose using unicycle model
    x, y, theta = pose
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += w * dt
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    pose = np.array([x, y, theta])
    trajectory.append(pose.copy())

    if np.linalg.norm(pose[:2] - goal) < 0.05:
        break

# Convert to arrays
trajectory = np.array(trajectory)

# Plotting
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Robot Path')
plt.plot(goal[0], goal[1], 'ro', label='Goal')
plt.title("Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(vs)
plt.title("Linear Velocity v")
plt.xlabel("Time Step")
plt.ylabel("v")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(ws)
plt.title("Angular Velocity w")
plt.xlabel("Time Step")
plt.ylabel("ω")
plt.grid(True)

plt.tight_layout()
#plt.savefig("kinematic_controller_results.png")
# Instead of plt.show()
plt.savefig("trajectory_plot.png")
print("Plot saved as trajectory_plot.png")

