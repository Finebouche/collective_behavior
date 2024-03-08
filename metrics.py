import numpy as np


def calculate_dos(loc_x, loc_y):
    if not loc_x or not loc_y:  # If no agents are present
        return 0

    loc_x = np.array(loc_x)
    loc_y = np.array(loc_y)
    distances = np.sqrt((loc_x[:, np.newaxis] - loc_x[np.newaxis, :]) ** 2 + (loc_y[:, np.newaxis] - loc_y[np.newaxis, :]) ** 2)
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)
    sum_min_distances = np.sum(min_distances)

    return sum_min_distances