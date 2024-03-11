import numpy as np


def calculate_dos(loc_x, loc_y):
    loc_x = np.array(loc_x)
    loc_y = np.array(loc_y)

    # Check if the arrays are not empty and have the same length
    if loc_x.size == 0 or loc_y.size == 0:
        raise ValueError("Both loc_x and loc_y must be non-empty arrays.")
    if loc_x.shape[0] != loc_y.shape[0]:
        raise ValueError("Arrays loc_x and loc_y must have the same length.")

    distances = np.sqrt((loc_x[:, np.newaxis] - loc_x[np.newaxis, :]) ** 2 + (loc_y[:, np.newaxis] - loc_y[np.newaxis, :]) ** 2)
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)
    sum_min_distances = np.sum(min_distances)

    return sum_min_distances