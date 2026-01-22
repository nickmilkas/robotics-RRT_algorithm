import numpy as np
import matplotlib.pyplot as plt


# Question A
def diff_kinematics(x: np.ndarray, u: np.ndarray):
    r = 0.1
    d = 0.25
    stand = r / (2 * d)
    omega = (u[1, 0] - u[0, 0]) * stand
    vel = (u[0, 0] + u[1, 0]) * stand * d
    if u[1, 0] > 0.5 or u[0, 0] > 0.5:
        message = "Give smaller u_values values. Must be no more than 0.5!"
        print(message)  # Changed to print to avoid breaking the array flow, or handle error upstream
        return np.zeros_like(x)  # Return zero velocity to prevent crash if error
    else:
        x_dot = np.array([[omega],
                          [vel * np.cos(x[0, 0])],
                          [vel * np.sin(x[0, 0])]])

        return x_dot


# Question B
def visualize_transitions_with_euler(u_val, xy_val, steps):
    pos = find_pos(u_val, xy_val, steps)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    d = 0.25

    def rot(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    for i in range(len(pos)):
        position = pos[i]
        bpb = np.array([[-d, -d], [d, -d], [d, d], [-d, d], [-d, -d]]).T
        bpw = rot(position[0, 0]) @ bpb + position[1:3, :]

        rect = plt.Polygon(bpw.T, edgecolor='black', fill=False)
        ax.add_patch(rect)

    # --- Dynamic Limits Logic ---
    # Extract all x and y coordinates from the path
    # pos shape is usually (steps, 3, 1), so we want all rows, index 1 (x) and 2 (y)
    all_x = pos[:, 1, 0]
    all_y = pos[:, 2, 0]

    margin = 1.0  # Buffer space around the path

    # Calculate limits based on data range + margin
    plt.xlim(np.min(all_x) - margin, np.max(all_x) + margin)
    plt.ylim(np.min(all_y) - margin, np.max(all_y) + margin)
    # ----------------------------

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Euler Method Transitions')
    plt.grid(True)
    plt.show()


def find_pos(u_val, x_val, steps):
    dt = 0.1
    p = x_val
    all_poses = [p]
    for index in range(steps):
        v = diff_kinematics(p, u_val)
        p = p + v * dt
        all_poses += [p]
    return np.array(all_poses)


# Question 3
def f_rk4_one_step(x0, u0):
    dt = 0.1
    f1 = diff_kinematics(x0, u0)
    f2 = diff_kinematics(x0 + 0.5 * dt * f1, u0)
    f3 = diff_kinematics(x0 + 0.5 * dt * f2, u0)
    f4 = diff_kinematics(x0 + dt * f3, u0)
    x_pos = x0 + dt * (f1 + 2. * f2 + 2. * f3 + f4) / 6.
    return x_pos


def visualize_transitions_with_rk4(u_val, xy_val, steps):
    states = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    d = 0.25
    position = xy_val

    # Pre-append the initial state so it is included in the plot
    states.append(position)

    for i in range(steps):
        pos = f_rk4_one_step(position, u_val)
        states.append(pos)
        position = pos
    states = np.array(states)

    def rot(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    for i in range(len(states)):
        position = states[i]
        bpb = np.array([[-d, -d], [d, -d], [d, d], [-d, d], [-d, -d]]).T
        bpw = rot(position[0, 0]) @ bpb + position[1:3, :]

        rect = plt.Polygon(bpw.T, edgecolor='black', fill=False)
        ax.add_patch(rect)

    # --- Dynamic Limits Logic ---
    all_x = states[:, 1, 0]
    all_y = states[:, 2, 0]

    margin = 1.0

    plt.xlim(np.min(all_x) - margin, np.max(all_x) + margin)
    plt.ylim(np.min(all_y) - margin, np.max(all_y) + margin)
    # ----------------------------

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('RK4 Method Transitions')
    plt.grid(True)
    plt.show()