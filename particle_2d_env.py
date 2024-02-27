import numpy as np
import math

from ray.rllib.env import MultiAgentEnv
from gymnasium import spaces
from gymnasium.utils import seeding
from ray.rllib.env.env_context import EnvContext


def sign(x):
    if x == 0:
        return 0
    elif x > 0:
        return 1
    else:
        return -1


def ComputeDistance(agent1, agent2):
    return math.sqrt(
        ((agent1.loc_x - agent2.loc_x) ** 2)
        + ((agent1.loc_y - agent2.loc_y) ** 2)
    )


def ComputeAngle(agent1, agent2):
    # Compute relative heading between the two agents
    direction = math.atan2(
        agent2.loc_y - agent1.loc_y,
        agent2.loc_x - agent1.loc_x
    ) - agent1.heading

    # Normalize the direction to the range [-pi, pi]
    direction = (direction + np.pi) % (2 * np.pi) - np.pi
    return direction


# function that check if variable is an array, then choose a random int in the interval
# return the number and the max_value
def random_int_in_interval(variable):
    if isinstance(variable, list):
        return np.random.randint(variable[0], variable[1]), variable[1]
    else:
        return variable, variable


class ParticuleAgent:
    def __init__(self, id=None, radius=None, agent_type=None,
                 loc_x=None, loc_y=None, heading=None,
                 speed_x=None, speed_y=None,
                 still_in_game=True):
        self.agent_id = id
        self.agent_type = agent_type
        self.radius = radius
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.heading = heading
        self.still_in_game = still_in_game


class Particle2dEnvironment(MultiAgentEnv):
    def __init__(self, config: EnvContext):

        super().__init__()
        self.float_dtype = np.float32
        self.int_dtype = np.int32
        self.eps = self.float_dtype(1e-10)

        self.np_random = np.random
        seed = config.get('seed')
        if seed is not None:
            self.np_random, seed = seeding.np_random(seed)

        # ENVIRONMENT
        self.timestep = 0
        self.episode_length = config.get('episode_length')
        assert self.episode_length > 0
        # stage size in an interval or a fixed value
        stage_size = config.get('stage_size')
        self.stage_size, _ = random_int_in_interval(stage_size)
        assert self.stage_size > 1
        self.grid_diagonal = self.stage_size * np.sqrt(2)

        # PHYSICS
        self.dragging_force_coefficient = config.get('dragging_force_coefficient')
        self.contact_force_coefficient = config.get('contact_force_coefficient')
        self.contact_margin = config.get('contact_margin')
        self.friction_regime = config.get('friction_regime')
        # assert that the value is either linear, quadratic or intermediate:
        assert self.friction_regime in ["linear", "quadratic", "intermediate"]

        self.wall_contact_force_coefficient = config.get('wall_contact_force_coefficient')
        self.periodical_boundary = config.get('periodical_boundary')
        if self.periodical_boundary is True:
            self.wall_contact_force_coefficient = None
        self.max_speed_prey = config.get('max_speed_prey')
        self.max_speed_predator = config.get('max_speed_predator')

        # AGENTS (PREYS AND PREDATORS)
        # random number of preys
        ini_num_preys = config.get('num_preys')
        self.ini_num_preys, self.max_num_preys = random_int_in_interval(ini_num_preys)
        self.num_preys = self.ini_num_preys
        # random number of predators
        num_predators = config.get('num_predators')
        self.num_predators, self.max_num_predators = random_int_in_interval(num_predators)
        assert self.num_preys > 0
        assert self.num_predators >= 0
        self.num_agents = self.num_preys + self.num_predators
        self.max_num_agents = self.max_num_preys + self.max_num_predators

        self.prey_radius = config.get('prey_radius')
        self.predator_radius = config.get('predator_radius')
        assert 0 <= self.prey_radius <= 1
        assert 0 <= self.predator_radius <= 1

        self.agent_density = config.get('agent_density')
        assert 0 < self.agent_density

        self.agents = []
        for i in range(self.max_num_agents):
            if i < self.num_agents:
                if i < self.num_preys:
                    agent_type, radius = 0, self.prey_radius  # for preys
                else:
                    agent_type, radius = 1, self.predator_radius  # for predators
                still_in_game = True
            else:
                agent_type, radius, still_in_game = None, None, False

            self.agents.append(ParticuleAgent(id=i, agent_type=agent_type, radius=radius, still_in_game=still_in_game))

        self._agent_ids = {agent.agent_id for agent in self.agents}  # Used by RLlib
        self.prey_consumed = config.get('prey_consumed')

        # ACTIONS (ACCELERATION AND TURN)
        self.max_acceleration_prey = config.get('max_acceleration_prey')
        self.max_acceleration_predator = config.get('max_acceleration_predator')
        self.max_turn = config.get('max_turn')

        self.action_space = spaces.Dict({
            agent.agent_id: spaces.Box(low=np.array([0, -self.max_turn]),
                                       high=np.array([
                                           self.max_acceleration_prey if agent.agent_type == 0 else self.max_acceleration_predator,
                                           self.max_turn]), shape=(2,),
                                       dtype=self.float_dtype) for agent in self.agents
        })

        # OBSERVATION SETTINGS
        self.max_seeing_angle = config.get('max_seeing_angle')
        if not 0 < self.max_seeing_angle <= np.pi:
            self.max_seeing_angle = np.pi

        self.max_seeing_distance = config.get('max_seeing_distance')
        if not 0 < self.max_seeing_distance <= self.grid_diagonal:
            self.max_seeing_distance = self.grid_diagonal

        self.sort_by_distance = config.get('sort_by_distance')

        self.num_other_agents_observed = config.get('num_other_agents_observed')
        if not 0 < self.num_other_agents_observed <= self.num_agents - 1 or self.num_other_agents_observed == "all":
            self.num_other_agents_observed = self.max_num_agents - 1

        self.use_polar_coordinate = config.get('use_polar_coordinate')
        self.use_speed_observation = config.get('use_speed_observation')

        # Number of observed properties
        self.num_observed_properties = 4  # heading, position and type are always observed
        if self.use_speed_observation:  # speed is observed when use_speed_observation is True
            self.num_observed_properties += 2
        self.observation_size = 5 + self.num_observed_properties * self.num_other_agents_observed

        self.observation_space = spaces.Dict({
            agent.agent_id: spaces.Box(
                low=-1, high=1,
                shape=(self.observation_size,),
                dtype=self.float_dtype) for agent in self.agents
        })

        # REWARDS
        self.starving_penalty_for_predator = config.get('starving_penalty_for_predator')
        self.eating_reward_for_predator = config.get('eating_reward_for_predator')
        self.collective_eating_reward_for_predator = config.get('collective_eating_reward_for_predator')
        self.surviving_reward_for_prey = config.get('surviving_reward_for_prey')
        self.death_penalty_for_prey = config.get('death_penalty_for_prey')
        self.collective_death_penalty_for_prey = config.get('collective_death_penalty_for_prey')
        self.edge_hit_penalty = config.get('edge_hit_penalty')
        self.energy_cost_penalty_coef = config.get('energy_cost_penalty_coef')

    def _observation_pos(self, agent, other_agent):
        if not self.use_polar_coordinate:
            return (other_agent.loc_x - agent.loc_x) / self.grid_diagonal, (
                    other_agent.loc_y - agent.loc_y) / self.grid_diagonal
        else:
            dist = ComputeDistance(agent, other_agent)
            direction = ComputeAngle(agent, other_agent)
            return dist / self.grid_diagonal, direction / (2 * np.pi)

    def _generate_observation(self, agent):
        """
        Generate and return the observations for every agent.
        """
        # initialize obs as an empty list of correct size
        obs = np.zeros(self.observation_size, dtype=self.float_dtype)

        # SELF OBSERVATION
        # Distance to walls
        obs[0] = agent.loc_x / self.grid_diagonal
        obs[1] = agent.loc_y / self.grid_diagonal
        # modulo 2pi to avoid large values
        obs[2] = agent.heading % (2 * np.pi) / (2 * np.pi)
        # speed
        # add speed normalized by max speed for prey or predator
        max_speed = self.max_speed_prey if agent.agent_type == 0 else self.max_speed_predator

        if not self.use_polar_coordinate:
            obs[3] = agent.speed_x / max_speed
            obs[4] = agent.speed_y / max_speed
        else:
            obs[3] = math.sqrt(agent.speed_x ** 2 + agent.speed_y ** 2) / max_speed
            obs[4] = math.atan2(agent.speed_y, agent.speed_x) % (2 * np.pi) / (2 * np.pi)

        # OTHER AGENTS
        # Remove the agent itself and the agents that are not in the game
        other_agents = [
            other_agent for other_agent in self.agents
            if other_agent is not agent and other_agent.still_in_game == 1
               and abs(ComputeAngle(agent, other_agent)) < self.max_seeing_angle
               and ComputeDistance(agent, other_agent) < self.max_seeing_distance
        ]
        # keep only the closest agents
        other_agents = other_agents[:self.num_other_agents_observed]

        # Observation of the other agents
        if self.sort_by_distance:
            # Sort the agents by distance
            other_agents = sorted(other_agents, key=lambda other_agent: ComputeDistance(agent, other_agent))

        for j, other in enumerate(other_agents):
            # count the number of already observed properties
            base_index = 5 + j * self.num_observed_properties
            obs[base_index], obs[base_index + 1] = self._observation_pos(agent, other)  # relative position
            obs[base_index + 2] = ((other.heading - agent.heading) % (2 * np.pi) - np.pi) / np.pi  # relative heading
            if self.use_speed_observation:
                # add speed normalized by max speed for prey or predator
                max_speed = self.max_speed_prey if other.agent_type == 0 else self.max_speed_predator

                if not self.use_polar_coordinate:
                    obs[base_index + 3] = (other.speed_x - agent.speed_x) / max_speed
                    obs[base_index + 4] = (other.speed_y - agent.speed_y) / max_speed
                else:
                    obs[base_index + 3] = math.sqrt(
                        (other.speed_x - agent.speed_x) ** 2 + (other.speed_y - agent.speed_y) ** 2
                    ) / max_speed
                    obs[base_index + 4] = math.atan2(other.speed_y - agent.speed_y,
                                                     other.speed_x - agent.speed_x
                                                     ) - agent.heading

            obs[base_index + (self.num_observed_properties - 1)] = other.agent_type

        return obs

    def _get_observation_list(self):
        return {agent.agent_id: self._generate_observation(agent) for agent in self.agents if agent.still_in_game == 1}

    def reset(self, seed=None, options=None):
        # Reset time to the beginning
        self.timestep = 0
        self.num_preys = self.ini_num_preys

        # Vectorized operations for random values
        # len(self.agents) can be bigger than self.num_agents
        random_values = self.np_random.random(size=(len(self.agents), 3))
        loc_x = random_values[:, 0] * self.stage_size
        loc_y = random_values[:, 1] * self.stage_size
        headings = random_values[:, 2] * 2 * np.pi

        # Assigning the vectorized values to agents
        for i, agent in enumerate(self.agents):
            agent.loc_x, agent.loc_y = loc_x[i], loc_y[i]
            agent.speed_x, agent.speed_y = 0.0, 0.0
            agent.heading = headings[i]
            agent.still_in_game = int(i < self.num_agents)

        observation_list = self._get_observation_list()
        return observation_list, {}

    def step(self, action_list):
        self.timestep += 1

        self._simulate_one_step(action_list)

        observation_list = self._get_observation_list()

        reward_list = self._get_reward(action_list)
        terminated, truncated = self._get_done()

        return observation_list, reward_list, terminated, truncated, {}

    def render(self):
        # TODO
        raise NotImplementedError()

    def _simulate_one_step(self, action_dict=None):

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
            actions = [action_dict.get(agent.agent_id, (0, 0)) for agent in self.agents  if agent.still_in_game]
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

            loc_x[out_of_bounds_x] = np.clip(loc_x[out_of_bounds_x], radius[out_of_bounds_x], self.stage_size - radius[out_of_bounds_x])
            speed_x[out_of_bounds_x] = 0
            loc_y[out_of_bounds_y] = np.clip(loc_y[out_of_bounds_y], radius[out_of_bounds_y], self.stage_size - radius[out_of_bounds_y])
            speed_y[out_of_bounds_y] = 0

        # UPDATE THE STATES OF ACTIVE AGENTS
        for i, agent in enumerate(active_agents):
            agent.loc_x = loc_x[i]
            agent.loc_y = loc_y[i]
            agent.speed_x = speed_x[i]
            agent.speed_y = speed_y[i]
            agent.heading = heading[i]

    def _get_reward(self, action_list):
        # Initialize rewards
        reward_list = {agent.agent_id: 0 for agent in self.agents}

        predator_agents = [other_agent for other_agent in self.agents if other_agent.agent_type == 1]

        for agent in self.agents:
            if agent.still_in_game:
                # AGENT EATEN OR NOT
                if agent.agent_type == 0:  # 0 for prey, is_prey
                    reward_list[agent.agent_id] += self.surviving_reward_for_prey

                    # Check if the agent is touching a predator
                    for predator_agent in predator_agents:
                        dist = ComputeDistance(predator_agent, agent)
                        eating_distance = predator_agent.radius + agent.radius
                        if dist < eating_distance:
                            # The prey is eaten
                            reward_list[predator_agent.agent_id] += self.eating_reward_for_predator
                            reward_list[agent.agent_id] += self.death_penalty_for_prey
                            if self.prey_consumed:
                                self.num_preys -= 1
                                agent.still_in_game = 0

                            # collective penalty for preys
                            for prey_agent in self.agents:
                                if prey_agent.agent_type == 0 and prey_agent.still_in_game == 1:
                                    reward_list[prey_agent.agent_id] += self.collective_death_penalty_for_prey

                            # collective rewards for predators
                            for predator_agent_2 in predator_agents:
                                reward_list[predator_agent_2.agent_id] += self.collective_eating_reward_for_predator

                else:  # is_predator
                    reward_list[agent.agent_id] += self.starving_penalty_for_predator

                # ENERGY EFFICIENCY
                # Add the energy efficiency penalty
                # set the energy cost penalty
                self_force_amplitude, self_force_orientation = action_list.get(agent.agent_id)

                energy_cost_penalty = -(
                        abs(self_force_amplitude)
                        + abs(self_force_orientation)
                ) * self.energy_cost_penalty_coef
                reward_list[agent.agent_id] += energy_cost_penalty

                # WALL avoidance
                # Check if the agent is touching the edge
                if self.periodical_boundary is False:
                    is_touching_edge_x = (
                            agent.loc_x < agent.radius
                            or agent.loc_x > self.stage_size - agent.radius
                    )
                    is_touching_edge_y = (
                            agent.loc_y < agent.radius
                            or agent.loc_y > self.stage_size - agent.radius
                    )

                    if is_touching_edge_x or is_touching_edge_y:
                        reward_list[agent.agent_id] += self.edge_hit_penalty

        return reward_list

    def _get_done(self):
        # Natural ending
        # True when a prey is eaten (agent.still_in_game == 0) or episode ends because all preys have been eaten (self.num_preys == 0)
        terminated = {agent.agent_id: self.num_preys == 0 or agent.still_in_game == 0 for agent in self.agents}
        terminated['__all__'] = self.num_preys == 0
        # Premature ending (because of time limit)
        truncated = {agent.agent_id: self.timestep >= self.episode_length for agent in self.agents}
        truncated['__all__'] = self.timestep >= self.episode_length

        return terminated, truncated
