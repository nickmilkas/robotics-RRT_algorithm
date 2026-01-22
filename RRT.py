import numpy as np
import Modeling


def state_to_tuple(xx):
    x = np.asarray(xx, dtype=float).reshape((-1, 1))
    return tuple(float(x[i, 0]) for i in range(x.shape[0]))


def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def angle_diff(a: float, b: float) -> float:
    return abs(wrap_to_pi(a - b))


def distance_xy(node: np.ndarray, x_target: np.ndarray) -> float:
    node = np.asarray(node).reshape((3, 1))
    x_target = np.asarray(x_target).reshape((3, 1))
    return float(np.linalg.norm(node[1:3, 0] - x_target[1:3, 0]))


def cost_function(node: np.ndarray, x_target: np.ndarray, w_pos: float = 1.0, w_theta: float = 0.2) -> float:
    dpos = distance_xy(node, x_target)
    dtheta = angle_diff(float(node[0, 0]), float(x_target[0, 0]))
    return float(w_pos * dpos + w_theta * dtheta)


def sample_state(goal: np.ndarray, bounds, goal_bias: float) -> np.ndarray:
    x_min, x_max, y_min, y_max = bounds
    if np.random.rand() < goal_bias:
        return np.asarray(goal, dtype=float).reshape((3, 1))
    theta = np.random.uniform(-np.pi, np.pi)
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    return np.array([[theta], [x], [y]], dtype=float)


def is_in_bounds(x: np.ndarray, bounds) -> bool:
    x_min, x_max, y_min, y_max = bounds
    return (x_min <= x[1, 0] <= x_max) and (y_min <= x[2, 0] <= y_max)


def is_collision(x: np.ndarray, obstacles, robot_radius: float) -> bool:
    if not obstacles:
        return False
    px = float(x[1, 0])
    py = float(x[2, 0])
    for radius, ox, oy in obstacles:
        dx = px - float(ox)
        dy = py - float(oy)
        if dx * dx + dy * dy <= (float(radius) + robot_radius) ** 2:
            return True
    return False


def rollout(x0: np.ndarray, u_seq, obstacles, robot_radius: float, bounds):
    x = np.asarray(x0, dtype=float).reshape((3, 1))
    visited = []
    for u in u_seq:
        x = Modeling.f_rk4_one_step(x, u)
        x = np.asarray(x, dtype=float).reshape((3, 1))
        x[0, 0] = wrap_to_pi(float(x[0, 0]))

        if not is_in_bounds(x, bounds):
            break
        if is_collision(x, obstacles, robot_radius):
            break

        visited.append(x)
    return visited


def greedy_rollout(x0: np.ndarray, x_target: np.ndarray, horizon_steps: int, obstacles, robot_radius: float, bounds):
    x = np.asarray(x0, dtype=float).reshape((3, 1))
    x_target = np.asarray(x_target, dtype=float).reshape((3, 1))
    visited = []

    base = 0.45
    k_turn = 0.8

    for _ in range(int(max(1, horizon_steps))):
        dx = float(x_target[1, 0] - x[1, 0])
        dy = float(x_target[2, 0] - x[2, 0])
        bearing = np.arctan2(dy, dx)
        err = wrap_to_pi(bearing - float(x[0, 0]))

        uR = base + k_turn * err
        uL = base - k_turn * err

        uL = float(np.clip(uL, -0.5, 0.5))
        uR = float(np.clip(uR, -0.5, 0.5))
        u = np.array([[uL], [uR]], dtype=float)

        x = Modeling.f_rk4_one_step(x, u)
        x = np.asarray(x, dtype=float).reshape((3, 1))
        x[0, 0] = wrap_to_pi(float(x[0, 0]))

        if not is_in_bounds(x, bounds):
            break
        if is_collision(x, obstacles, robot_radius):
            break

        visited.append(x)

    return visited


def connect(x_start: np.ndarray,
            x_target: np.ndarray,
            number_of_samples: int,
            horizon_steps: int,
            obstacles,
            robot_radius: float,
            bounds,
            early_stop_cost: float):
    best_cost = float("inf")
    best_edge = None

    # Greedy attempt first
    greedy = greedy_rollout(x_start, x_target, horizon_steps, obstacles, robot_radius, bounds)
    if greedy:
        costs = [cost_function(s, x_target) for s in greedy]
        idx = int(np.argmin(costs))
        best_cost = float(costs[idx])
        best_edge = greedy[:idx + 1]
        if best_cost <= early_stop_cost:
            return best_edge[-1], best_edge

    K = int(max(1, min(number_of_samples, 120)))
    H = int(max(1, horizon_steps))

    for _ in range(K):
        u_seq = []
        for _t in range(H):
            if np.random.rand() < 0.7:
                uL = np.random.uniform(0.0, 0.5)
                uR = np.random.uniform(0.0, 0.5)
            else:
                uL = np.random.uniform(-0.5, 0.5)
                uR = np.random.uniform(-0.5, 0.5)
            u_seq.append(np.array([[uL], [uR]], dtype=float))

        visited = rollout(x_start, u_seq, obstacles, robot_radius, bounds)
        if not visited:
            continue

        costs = [cost_function(s, x_target) for s in visited]
        idx = int(np.argmin(costs))
        c = float(costs[idx])

        if c < best_cost:
            best_cost = c
            best_edge = visited[:idx + 1]

        if best_cost <= early_stop_cost:
            break

    if best_edge is None:
        return None, []
    return best_edge[-1], best_edge


def goal_reached(x: np.ndarray, goal: np.ndarray, pos_tol: float, theta_tol: float) -> bool:
    dpos = distance_xy(x, goal)
    dtheta = angle_diff(float(x[0, 0]), float(goal[0, 0]))
    return (dpos <= pos_tol) and (dtheta <= theta_tol)


def nearest_node_idx(node_data: np.ndarray, x_sample: np.ndarray) -> int:
    xs = float(x_sample[1, 0])
    ys = float(x_sample[2, 0])
    thetas = float(x_sample[0, 0])

    dx = node_data[:, 1] - xs
    dy = node_data[:, 2] - ys
    dpos = np.hypot(dx, dy)

    dtheta = np.abs(((node_data[:, 0] - thetas + np.pi) % (2 * np.pi)) - np.pi)

    score = 2.0 * dpos + 0.5 * dtheta
    return int(np.argmin(score))


def reconstruct_path(start_key, end_key, parents, edges):
    chain = []
    cur = end_key
    while cur != start_key and cur is not None:
        chain.append(cur)
        cur = parents.get(cur, None)

    if cur != start_key:
        return [start_key]

    chain.reverse()
    path = [start_key]
    for node_key in chain:
        path.extend(edges[node_key])
    return path


def rrt_core(xy_start, xy_goal, max_iters, number_of_samples, obstacles=None):
    xy_start = np.asarray(xy_start, dtype=float).reshape((3, 1))
    xy_goal = np.asarray(xy_goal, dtype=float).reshape((3, 1))

    bounds = (-1.0, 12.0, -1.0, 12.0)

    goal_bias = 0.10
    horizon_steps = 10
    robot_radius = 0.10

    pos_tol = 0.25
    theta_tol = 0.35
    early_stop_cost = 0.60

    start_key = state_to_tuple(xy_start)

    nodes = [start_key]
    node_data = np.array([start_key], dtype=float)
    node_states = {start_key: xy_start}
    parents = {start_key: None}
    edges = {start_key: []}

    best_key = start_key
    best_goal_cost = cost_function(xy_start, xy_goal)

    for _ in range(int(max_iters)):
        x_sample = sample_state(xy_goal, bounds, goal_bias)

        nearest_key = nodes[nearest_node_idx(node_data, x_sample)]
        x_nearest = node_states[nearest_key]

        x_new, edge_states = connect(
            x_nearest, x_sample,
            number_of_samples=number_of_samples,
            horizon_steps=horizon_steps,
            obstacles=obstacles,
            robot_radius=robot_radius,
            bounds=bounds,
            early_stop_cost=early_stop_cost
        )

        if x_new is None:
            continue

        new_key = state_to_tuple(x_new)
        if new_key in parents:
            continue

        parents[new_key] = nearest_key
        node_states[new_key] = x_new
        edges[new_key] = [state_to_tuple(s) for s in edge_states]

        nodes.append(new_key)
        node_data = np.vstack([node_data, np.array([new_key], dtype=float)])

        g_cost = cost_function(x_new, xy_goal)
        if g_cost < best_goal_cost:
            best_goal_cost = g_cost
            best_key = new_key

        if goal_reached(x_new, xy_goal, pos_tol=pos_tol, theta_tol=theta_tol):
            return True, reconstruct_path(start_key, new_key, parents, edges)

    return False, reconstruct_path(start_key, best_key, parents, edges)


def RRT(xy_start, xy_goal, max_iters, number_of_samples):
    return rrt_core(xy_start, xy_goal, max_iters, number_of_samples, obstacles=None)


def RRT_with_obstacles(xy_start, xy_goal, max_iters, number_of_samples, obs_positions=None):
    return rrt_core(xy_start, xy_goal, max_iters, number_of_samples, obstacles=obs_positions)
