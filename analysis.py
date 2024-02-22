import numpy as np


def calculate_dos(observation, D):
    """
    Calculate the Degree of Sparsity (DoS) for a swarm of agents.
    :param agents_positions: A list containing the positions of the agents for each time step.

                             Each element is a list of tuples representing the positions (x, y) of N agents at time t.
    :param D: The environment size (maximum possible distance between any two agents).
    :return: The Degree of Sparsity.
    """

    # Initialize the sum of minimum distances.
    sum_min_distances = 0.0

    loc_x = observation["loc_x"]
    loc_y = observation["loc_y"]
    #  T: The episode length (number of time steps).
    T = observation["loc_x"].shape[0]
    # N: The total number of agents
    N = observation["loc_x"].shape[1]

    # Iterate over each time step.
    for t in range(T):
        # Get the positions of agents at time step t.
        positions_at_t = [(loc_x[t, i], loc_y[t, i]) for i in range(N)]
        # Calculate the distance to the nearest neighbor for each agent.
        for j in range(N):
            # Get the position of the j-th agent at time t.
            xj = np.array(positions_at_t[j])

            # Initialize the minimum distance for the j-th agent as infinity.
            min_distance = np.inf
            # Iterate over all other agents to find the nearest neighbor.
            for k in range(N):
                if k != j:
                    # Get the position of the k-th agent at time t.
                    xk = np.array(positions_at_t[k])
                    # Calculate the Euclidean distance between the j-th and k-th agents.
                    distance = np.linalg.norm(xj - xk)
                    # Update the minimum distance if this distance is smaller.
                    if distance < min_distance:
                        min_distance = distance
            # Add the minimum distance to the sum.
            sum_min_distances += min_distance
    # Calculate the average normalized distance (DoS).
    dos = (1 / (T * N * D)) * sum_min_distances

    return dos
