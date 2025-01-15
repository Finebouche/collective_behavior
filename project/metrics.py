import numpy as np
from scipy.spatial import distance_matrix


def calculate_dos(loc_x, loc_y):
    """
    Degree of Sparsity (DoS):
    - A measure of how spread out (or dense) agents are.
    - In this example: the sum of each agent's distance to its closest neighbor.
      (Often you'd take the average or another statistic, but we'll preserve the 'sum' behavior.)
    """
    # Edge case: If fewer than 2 agents, return 0
    if len(loc_x) < 2 or len(loc_y) < 2:
        return 0.0

    # Combine x and y coordinates into an (N,2) array
    locations = np.column_stack((loc_x, loc_y))
    dist_mat = distance_matrix(locations, locations)
    # Ignore self-distances by setting them to np.inf
    np.fill_diagonal(dist_mat, np.inf)

    # Take the minimum distance per agent
    min_dists = np.min(dist_mat, axis=1)
    return np.sum(min_dists)


def calculate_doa(headings):
    """
    Degree of Alignment (DoA):
    - A measure of how aligned agents are, based on headings in [0, 2π).
    - Here, we sum each agent's minimum circular difference to any other agent.
      (Again, you might want an average or another metric, but we'll preserve the existing logic.)
    """
    # If fewer than 2 agents, no meaningful alignment
    if len(headings) < 2:
        return 0.0

    headings = np.asarray(headings, dtype=float)

    # Create pairwise angle differences using broadcasting
    diff_mat = np.abs(headings[:, None] - headings)
    # For circular angles, the difference is min(angle_diff, 2π - angle_diff)
    diff_mat = np.minimum(diff_mat, 2 * np.pi - diff_mat)

    # Ignore self-comparisons
    np.fill_diagonal(diff_mat, np.inf)

    # For each agent, find the minimum angular difference across others
    min_diffs = np.min(diff_mat, axis=1)
    return np.sum(min_diffs)