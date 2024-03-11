
import numpy as np


def calculate_dos(loc_x, loc_y):
    # Dos : degree of sparsity
    # It is a measure of the density of the agents in the environment
    # It is calculated as the average normalized distance to the nearest neighborhood of all conspecifics in an episode
    if not loc_x or not loc_y:  # If no agents are present
        return 0

    locations = np.vstack((loc_x, loc_y)).T

    # Calculate the pairwise distances using np.linalg.norm
    distances = np.linalg.norm(locations[:, np.newaxis] - locations, axis=2)
    sorted_distances = np.sort(distances, axis=1)
    # Pick the second-smallest distance (first one is zero for self-distance)
    min_distances = sorted_distances[:, 1]
    sum_min_distances = np.sum(min_distances)

    return sum_min_distances


def calculate_doa(headings):
    # Calculates the Degree of Alignment (DoA) among agents based on their headings.

    if len(headings) < 2:  # Need at least two agents to calculate DoA
        return 0

    headings = np.array(headings)
    # Ensure headings are in radians for np.cos computation
    alignments = headings[:, np.newaxis] - headings[np.newaxis, :]
    sorted_alignments = np.sort(alignments, axis=1)
    # Pick the second-smallest alignments (first one is zero for self-alignments)
    min_alignments = sorted_alignments[:, 1]
    sum_alignments = np.sum(min_alignments)

    return sum_alignments
s