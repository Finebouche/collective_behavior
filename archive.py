def _simulate_one_vectorized_step(self, action_dict=None):
    # Not worth it for small number of agents ! Might not be up to date !

    # Filter agents still in the game
    active_agents = [agent for agent in self.agents if agent.still_in_game]

    # Vectorized attributes
    loc_x = np.array([agent.loc_x for agent in active_agents])
    loc_y = np.array([agent.loc_y for agent in active_agents])
    speed_x = np.array([agent.speed_x for agent in active_agents])
    heading = np.array([agent.heading for agent in active_agents])
    speed_y = np.array([agent.speed_y for agent in active_agents])
    radius = np.array([agent.radius for agent in active_agents])
    agent_types = np.array([agent.agent_type for agent in active_agents])

    # DRAGGING FORCE
    speed_magnitude = np.sqrt(speed_x ** 2 + speed_y ** 2)

    # Calculate the dragging force amplitude based on the chosen type of friction
    if self.friction_regime == "linear":
        dragging_force_amplitude = speed_magnitude * self.dragging_force_coefficient
    elif self.friction_regime == "quadratic":
        dragging_force_amplitude = speed_magnitude ** 2 * self.dragging_force_coefficient
    else:
        dragging_force_amplitude = speed_magnitude ** 1.4 * self.dragging_force_coefficient
    dragging_force_orientation = np.arctan2(speed_y, speed_x) - np.pi
    acceleration_x = dragging_force_amplitude * np.cos(dragging_force_orientation)
    acceleration_y = dragging_force_amplitude * np.sin(dragging_force_orientation)

    # ACCELERATION FORCE
    if action_dict is not None:
        # put action dictionary in a list
        actions = [action_dict.get(agent.agent_id, (0, 0)) for agent in self.agents if agent.still_in_game]
        self_force_amplitude, self_force_orientation = zip(*actions)
        heading = (heading + self_force_orientation) % (2 * np.pi)
        acceleration_x += self_force_amplitude * np.cos(heading)
        acceleration_y += self_force_amplitude * np.sin(heading)

    # CONTACT FORCES WITH AGENTS
    # Calculate distances between agents
    delta_x = loc_x[:, np.newaxis] - loc_x
    delta_y = loc_y[:, np.newaxis] - loc_y
    distances = np.sqrt(delta_x ** 2 + delta_y ** 2)

    # Calculate minimum distances for collision (matrix of sum of radii)
    dist_min_matrix = radius[:, np.newaxis] + radius

    # Identify collisions (where distance < sum of radii)
    collisions = distances < dist_min_matrix

    # Calculate penetration (use np.logaddexp for numerical stability)
    penetration = np.logaddexp(0, -(distances - dist_min_matrix) / self.contact_margin) * self.contact_margin
    force_magnitude = self.contact_force_coefficient * penetration
    force_direction_x = np.where(distances != 0, delta_x / (distances + self.eps), 0)
    force_direction_y = np.where(distances != 0, delta_y / (distances + self.eps), 0)

    # Calculate forces for each possible collision (N x N)
    force_x = np.zeros_like(collisions, dtype=np.float64)  # Specify dtype as np.float64 otherwise it will be bool
    force_y = np.zeros_like(collisions, dtype=np.float64)
    force_x[collisions] += force_magnitude[collisions] * force_direction_x[collisions]
    force_y[collisions] += force_magnitude[collisions] * force_direction_y[collisions]
    contact_force = np.stack((np.sum(force_x, axis=1), np.sum(force_y, axis=1)), axis=-1)
    acceleration_x += contact_force[:, 0]
    acceleration_y += contact_force[:, 1]

    # CONTACT FORCES WITH WALLS
    touching_edge_x = np.logical_or(loc_x < radius, loc_x > self.stage_size - radius)
    touching_edge_y = np.logical_or(loc_y < radius, loc_y > self.stage_size - radius)

    contact_force_x = np.zeros_like(loc_x)
    contact_force_y = np.zeros_like(loc_y)

    # Compute contact force for x dimension
    x_less_than_radius = loc_x < radius
    force_x_left = radius - loc_x  # Force when touching on the left
    force_x_right = loc_x - self.stage_size + radius  # Force when touching on the right
    # Apply element-wise conditional logic
    contact_force_x[touching_edge_x] = self.wall_contact_force_coefficient * np.where(
        x_less_than_radius[touching_edge_x], force_x_left[touching_edge_x], force_x_right[touching_edge_x]
    )

    # Compute contact force for y dimension
    y_less_than_radius = loc_y < radius
    force_y_bottom = radius - loc_y  # Force when touching at the bottom
    force_y_top = loc_y - self.stage_size + radius  # Force when touching at the top
    # Apply element-wise conditional logic
    contact_force_y[touching_edge_y] = self.wall_contact_force_coefficient * np.where(
        y_less_than_radius[touching_edge_y], force_y_bottom[touching_edge_y], force_y_top[touching_edge_y]
    )

    # Update acceleration with contact forces
    acceleration_x += np.sign(self.stage_size / 2 - loc_x) * contact_force_x
    acceleration_y += np.sign(self.stage_size / 2 - loc_y) * contact_force_y

    # Compute new speeds
    speed_x += acceleration_x / (radius ** 3 * self.agent_density)
    speed_y += acceleration_y / (radius ** 3 * self.agent_density)
    speed_magnitude = np.sqrt(speed_x ** 2 + speed_y ** 2)
    max_speeds = np.where(agent_types == 0, self.max_speed_prey, self.max_speed_predator)
    over_limit = speed_magnitude > max_speeds
    speed_x[over_limit] *= max_speeds[over_limit] / speed_magnitude[over_limit]
    speed_y[over_limit] *= max_speeds[over_limit] / speed_magnitude[over_limit]

    # Update positions
    loc_x += speed_x
    loc_y += speed_y

    # Handle periodic boundaries
    if self.periodical_boundary:
        loc_x %= self.stage_size
        loc_y %= self.stage_size
    else:
        # Ensure agents do not go out of bounds and adjust speed to 0 if they hit the boundaries
        out_of_bounds_x = np.logical_or(loc_x < radius / 2, loc_x > self.stage_size - radius / 2)
        out_of_bounds_y = np.logical_or(loc_y < radius / 2, loc_y > self.stage_size - radius / 2)

        loc_x[out_of_bounds_x] = np.clip(loc_x[out_of_bounds_x], radius[out_of_bounds_x],
                                         self.stage_size - radius[out_of_bounds_x])
        speed_x[out_of_bounds_x] = 0
        loc_y[out_of_bounds_y] = np.clip(loc_y[out_of_bounds_y], radius[out_of_bounds_y],
                                         self.stage_size - radius[out_of_bounds_y])
        speed_y[out_of_bounds_y] = 0

    # UPDATE THE STATES OF ACTIVE AGENTS
    for i, agent in enumerate(active_agents):
        agent.loc_x = loc_x[i]
        agent.loc_y = loc_y[i]
        agent.speed_x = speed_x[i]
        agent.speed_y = speed_y[i]
        agent.heading = heading[i]
