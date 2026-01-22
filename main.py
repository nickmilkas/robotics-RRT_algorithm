import numpy as np
from Modeling import visualize_transitions_with_euler, visualize_transitions_with_rk4


# Initializing

u_values = np.array([[0.42], [0.43]])
x_values = np.array([[np.pi / 12], [1], [7]])

visualize_transitions_with_euler(u_values, x_values, 12000)

visualize_transitions_with_rk4(u_values, x_values, 12000)