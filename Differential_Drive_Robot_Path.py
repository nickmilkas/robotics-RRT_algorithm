import numpy as np
import matplotlib.pyplot as plt
import RRT


def plot_path_without_obstacles(xy_begin, xy_finish, iterations, number_of_samples, step_size=25):
    valid, path_to_go = RRT.RRT(xy_begin, xy_finish, iterations, number_of_samples)

    if not valid:
        print("Planner did not reach the goal. Plotting the best partial path found.")

    for i in range(0, len(path_to_go) - 1, step_size):
        point = path_to_go[i]
        next_point = path_to_go[min(i + step_size, len(path_to_go) - 1)]
        plt.plot([point[1], next_point[1]], [point[2], next_point[2]], color='b')

        rect_outer = plt.Rectangle((point[1] - 0.1, point[2] - 0.1), 0.2, 0.2, color='b', fill=False)
        rect_inner = plt.Rectangle((point[1] - 0.05, point[2] - 0.05), 0.1, 0.1, color='b', fill=True)
        plt.gca().add_patch(rect_outer)
        plt.gca().add_patch(rect_inner)

    plt.plot(xy_begin[1], xy_begin[2], 'ro', label='Start')
    plt.plot(xy_finish[1], xy_finish[2], 'go', label='Goal')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Path without Obstacles')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def plot_path_with_obstacles(xy_begin, xy_finish, iterations, number_of_samples, obst, step_size=30):
    valid, path_to_go = RRT.RRT_with_obstacles(
        xy_begin, xy_finish, iterations, number_of_samples, obst
    )

    if not valid:
        print("Planner did not reach the goal (with obstacles). Plotting the best partial path found.")

    for i in range(0, len(path_to_go) - 1, step_size):
        point = path_to_go[i]
        next_point = path_to_go[min(i + step_size, len(path_to_go) - 1)]
        plt.plot([point[1], next_point[1]], [point[2], next_point[2]], color='b')

        rect_outer = plt.Rectangle((point[1] - 0.1, point[2] - 0.1), 0.2, 0.2, color='b', fill=False)
        rect_inner = plt.Rectangle((point[1] - 0.05, point[2] - 0.05), 0.1, 0.1, color='b', fill=True)
        plt.gca().add_patch(rect_outer)
        plt.gca().add_patch(rect_inner)

    for radius, x_center, y_center in obst:
        circle = plt.Circle((x_center, y_center), radius, color='r', fill=False)
        plt.gca().add_artist(circle)

    plt.plot(xy_begin[1], xy_begin[2], 'ro', label='Start')
    plt.plot(xy_finish[1], xy_finish[2], 'go', label='Goal')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Path with Obstacles')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

