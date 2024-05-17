
import numpy as np


def calculate_dos(loc_x, loc_y):
    # Dos : degree of sparsity
    # It is a measure of the density of the agents in the environment
    # It is calculated as the average normalized distance to the nearest neighborhood of all conspecifics in an episode
    if len(loc_x) < 2 or len(loc_y) < 2:  # If 1 or 0 agent
        return 0

    locations = np.vstack((loc_x, loc_y)).T

    # Calculate the pairwise distances using np.linalg.norm
    distances = np.linalg.norm(locations[:, np.newaxis] - locations, axis=2)
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)
    sum_min_distances = np.sum(min_distances)

    return sum_min_distances


def calculate_doa(headings):
    # Calculates the Degree of Alignment (DoA) among agents based on their headings.

    if len(headings) < 2:  # Need at least two agents to calculate DoA
        return 0

    headings = np.array(headings)
    n = len(headings)
    alignments = np.zeros((n, n))

    # Calculate pairwise angular differences, taking the circular nature into account
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = abs(headings[i] - headings[j])
                # Considering circular nature of angles: the difference should be in the range [0, π]
                alignments[i, j] = min(diff, 2*np.pi - diff)
            else:
                alignments[i, j] = np.inf  # Ignore self-comparison

    # Take the minimum alignment for each agent, and then sum these minimums
    min_alignments = np.min(alignments, axis=1)
    sum_alignments = np.sum(min_alignments)

    return sum_alignments
