from scipy.spatial import distance_matrix
import numpy as np
import pandas as pd
import networkx as nx

def generate_runner_dataframe(trainer):
    assert trainer is not None

    # Fetch episode states
    episode_states = trainer.fetch_episode_states(
        ["loc_x", "loc_y", "still_in_the_game"]
    )
    env = trainer.cuda_envs.env

    episode_states["loc_x"] = episode_states["loc_x"]/ env.stage_size
    episode_states["loc_y"] = episode_states["loc_y"]/ env.stage_size
    assert isinstance(episode_states, dict)

    # Create empty pandas dataframe
    df = pd.DataFrame(columns=["time_step", "agent_id", "x", "y", "still_in_the_game"])

    # Loop over time steps and agents to initialize lines
    for t in range(len(episode_states["loc_x"])):
        for i in range(len(episode_states["loc_x"][0])):
            if i not in env.predators:  # only include runners
                df = df.append({
                    "time_step": t,
                    "agent_id": i,
                    "x": episode_states["loc_x"][t, i],
                    "y": episode_states["loc_y"][t, i],
                    "still_in_the_game": episode_states["still_in_the_game"][t, i]
                }, ignore_index=True)

    # Calculate velocity and orientation
    df["delta_x"] = df.groupby("agent_id")["x"].diff()
    df["delta_y"] = df.groupby("agent_id")["y"].diff()
    df["velocity"] = np.sqrt(df["delta_x"] ** 2 + df["delta_y"] ** 2)
    df["orientation"] = np.arctan2(df["delta_y"], df["delta_x"]) * 180 / np.pi

    return df


def calculate_milling_parameters(df, threshold, min_group_size):
    num_time_steps = df["time_step"].nunique()
    max_agents = df.groupby("time_step").size().max()
    pairwise_dists = np.zeros((num_time_steps, max_agents, max_agents))

    for t in range(num_time_steps):
        agents_t = df[df["time_step"] == t][["x", "y"]].values
        n_agents_t = agents_t.shape[0]
        pairwise_dists_t = np.zeros((max_agents, max_agents))
        if n_agents_t > 1:
            pairwise_dists_t[:n_agents_t, :n_agents_t] = distance_matrix(agents_t, agents_t)
        pairwise_dists[t] = pairwise_dists_t

    groups = []

    for t in range(num_time_steps):
        # Find pairs of agents that are close enough to be considered part of the same group
        close_pairs = np.where(pairwise_dists[t] <= threshold)
        close_pairs = np.sort(np.stack(close_pairs, axis=-1), axis=-1)
        close_pairs = np.unique(close_pairs, axis=0)

        # Build a graph of the close pairs of agents
        graph = nx.Graph()
        graph.add_edges_from(close_pairs)

        # Identify connected components of the graph
        connected_components = list(nx.connected_components(graph))

        # Filter out groups that are smaller than the minimum size
        groups_t = [list(component) for component in connected_components if len(component) >= min_group_size]
        groups.append(groups_t)

    for t in range(num_time_steps):
        groups_t = groups[t]
        num_groups_t = len(groups_t)

        if num_groups_t == 0:
            result.append({
                "time_step": t,
                "num_groups": 0,
                "mean_group_size": 0,
                "std_group_size": 0,
                "mean_group_speed": 0,
                "std_group_speed": 0,
                "mean_group_angular_velocity": 0,
                "std_group_angular_velocity": 0
            })
            continue
        # Calculate orientation difference between consecutive rows
        df["orientation_diff"] = df["orientation"].diff()
        df.dropna(inplace=True)

        # Identify time steps where orientation difference exceeds threshold
        df["orientation_change"] = abs(df["orientation_diff"]) > threshold
        df["orientation_change"] = df["orientation_change"].astype(int)

        # Identify groups of consecutive time steps where orientation change occurs
        df["group"] = df["orientation_change"].diff()
        df.loc[0, "group"] = 1
        df["group_start"] = df["group"] == 1
        df["group_end"] = df["group"] == -1
        df.drop("group", axis=1, inplace=True)
        df["group"] = df["group_start"].cumsum()

        # Identify groups that meet minimum size requirement
        group_sizes = df.groupby("group").size()
        valid_groups = group_sizes[group_sizes >= min_group_size].index
        df = df[df["group"].isin(valid_groups)]

        # Calculate mean and standard deviation of group angular velocity
        group_orientations_t = [df.iloc[groups_t[i]]["orientation"].values for i in range(num_groups_t)]
        group_orientations_t = np.stack(group_orientations_t)
        group_ang_velocities_t = np.stack([np.mean(np.diff(group_orientations_t[i])) for i in range(num_groups_t)])
        mean_group_ang_velocity_t = np.mean(group_ang_velocities_t)
        std_group_ang_velocity_t = np.std(group_ang_velocities_t)

        # Identify time steps where angular velocity exceeds threshold
        df["ang_velocity"] = df["orientation_diff"].rolling(window=5).sum()
        df["ang_velocity_change"] = abs(df["ang_velocity"]) > mean_group_ang_velocity_t + std_group_ang_velocity_t
        df["ang_velocity_change"] = df["ang_velocity_change"].astype(int)

        # Identify groups of consecutive time steps where angular velocity change occurs
        df["group"] = df["ang_velocity_change"].diff()
        df.loc[0, "group"] = 1
        df["group_start"] = df["group"] == 1
        df["group_end"] = df["group"] == -1
        df.drop("group", axis=1, inplace=True)
        df["group"] = df["group_start"].cumsum()

        # Identify groups that meet minimum size requirement
        group_sizes = df.groupby("group").size()
        valid_groups = group_sizes[group_sizes >= min_group_size].index
        df = df[df["group"].isin(valid_groups)]

        return df
    return None


def calculate_group_properties(df, threshold, min_group_size):
    num_time_steps = df["time_step"].nunique()
    max_agents = df.groupby("time_step").size().max()
    pairwise_dists = np.zeros((num_time_steps, max_agents, max_agents))
    group_properties = pd.DataFrame(
        columns=["group_id", "group_size", "mean_group_speed", "mean_group_ang_velocity", "timestep"])

    for t in range(num_time_steps):
        agents_t = df[df["time_step"] == t][["x", "y"]].values
        n_agents_t = agents_t.shape[0]
        pairwise_dists_t = np.zeros((max_agents, max_agents))
        if n_agents_t > 1:
            pairwise_dists_t[:n_agents_t, :n_agents_t] = distance_matrix(agents_t, agents_t)
        pairwise_dists[t] = pairwise_dists_t

    # Identify groups of agents that are milling together at each time step
    groups = []
    for t in range(num_time_steps):
        # Find pairs of agents that are close enough to be considered part of the same group
        close_pairs = np.where(pairwise_dists[t] <= threshold)
        close_pairs = np.sort(np.stack(close_pairs, axis=-1), axis=-1)
        close_pairs = np.unique(close_pairs, axis=0)

        # Build a graph of the close pairs of agents
        graph = nx.Graph()
        graph.add_edges_from(close_pairs)

        # Identify connected components of the graph
        connected_components = list(nx.connected_components(graph))

        # Filter out groups that are smaller than the minimum size
        groups_t = [list(component) for component in connected_components if len(component) >= min_group_size]
        groups.append(groups_t)

    # Calculate milling parameters for each time step
    result = []
    for t in range(num_time_steps):
        groups_t = groups[t]
        num_groups_t = len(groups_t)

        if num_groups_t == 0:
            result.append({
                "time_step": t,
                "num_groups": 0,
                "mean_group_size": 0,
                "std_group_size": 0,
                "mean_group_speed": 0,
                "std_group_speed": 0,
                "mean_group_angular_velocity": 0,
                "std_group_angular_velocity": 0
            })
            continue

        # Calculate group properties
        group_sizes_t = [len(group) for group in groups_t]
        group_positions_t = [df.iloc[groups_t[i]][["x", "y"]].values for i in range(num_groups_t)]

        # Calculate mean and standard deviation of group size
        mean_group_size_t = np.mean(group_sizes_t)
        std_group_size_t = np.std(group_sizes_t)

        # Calculate mean and standard deviation of group speed
        group_velocities_t = np.stack(
            [np.mean(np.diff(group_positions_t[i], axis=0), axis=0) for i in range(num_groups_t)])
        group_speeds_t = np.sqrt(np.sum(group_velocities_t ** 2, axis=-1))
        mean_group_speed_t = np.mean(group_speeds_t)
        std_group_speed_t = np.std(group_speeds_t)

        # Calculate mean and standard deviation of group angular velocity
        group_orientations_t = [df.iloc[groups_t[i]]["orientation"].values for i in range(num_groups_t)]
        max_len = max(len(a) for a in group_orientations_t)
        group_orientations_t = [np.pad(a, (0, max_len - len(a)), mode='constant', constant_values=np.nan) for a in
                                group_orientations_t]
        group_orientations_t = np.stack(group_orientations_t)

        group_ang_velocities_t = np.stack([np.mean(np.diff(group_orientations_t[i])) for i in range(num_groups_t)])
        mean_group_ang_velocity_t = np.mean(group_ang_velocities_t)
        std_group_ang_velocity_t = np.std(group_ang_velocities_t)

        # Add group properties to dataframe
        group_properties_t = pd.DataFrame({
            "group_id": np.arange(num_groups_t),
            "group_size": group_sizes_t,
            "mean_group_speed": group_speeds_t,
            "mean_group_ang_velocity": group_ang_velocities_t
        })

        # Add timestep to group properties
        group_properties_t["timestep"] = t

        # Append group properties to master dataframe
        group_properties = pd.concat([group_properties, group_properties_t], ignore_index=True)

    return group_properties