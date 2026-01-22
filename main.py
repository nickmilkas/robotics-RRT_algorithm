from Differential_Drive_Robot_Path import *
from Modeling import *

def run_default_path_planning():
    # Defaults from your assignment script
    x_begin = np.array([[np.pi / 2], [1], [4]], dtype=float)
    x_finish = np.array([[-np.pi / 2], [6], [8]], dtype=float)
    obstacles = [(0.2, 3, 7), (0.2, 3.7, 6)]

    iterations = 300
    number_of_samples = 200

    plot_path_with_obstacles(x_begin, x_finish, iterations, number_of_samples, obstacles)

def run_modeling_demo():
    u_values = np.array([[0.42], [0.43]])
    x_values = np.array([[np.pi / 12], [1], [7]])
    visualize_transitions_with_rk4(u_values, x_values, steps=200)


if __name__ == "__main__":
    # run_modeling_demo()
    run_default_path_planning()
