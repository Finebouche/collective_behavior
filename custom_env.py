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


class CustomEnvironment(MultiAgentEnv):
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
        self.preparation_length = config.get('preparation_length')

        # random stage size
        stage_size = config.get('stage_size')
        self.stage_size, _ = random_int_in_interval(stage_size)
        assert self.stage_size > 1
        self.grid_diagonal = self.stage_size * np.sqrt(2)

        # AGENTS
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
            if i < self.num_preys and i < self.num_agents:
                agent_type = 0  # for preys
                radius = self.prey_radius
            elif i >= self.num_preys and i < self.num_agents:
                agent_type = 1  # for predators
                radius = self.predator_radius
            else:
                agent_type = 0
                radius = 0
            self.agents.append(ParticuleAgent(id=i, agent_type=agent_type, radius=radius))

        self._agent_ids = {agent.agent_id for agent in self.agents}  # Used by RLlib

        # PHYSICS
        self.dragging_force_coefficient = config.get('dragging_force_coefficient')
        self.contact_force_coefficient = config.get('contact_force_coefficient')
        self.wall_contact_force_coefficient = config.get('wall_contact_force_coefficient')
        self.periodical_boundary = config.get('periodical_boundary')
        if self.periodical_boundary is True:
            self.wall_contact_force_coefficient = None

        # ACTIONS
        self.max_acceleration = config.get('max_acceleration')
        self.max_turn = config.get('max_turn')

        self.action_space = spaces.Dict({
            agent.agent_id: spaces.Box(low=np.array([0, -self.max_turn]),
                                       high=np.array([self.max_acceleration, self.max_turn]), shape=(2,),
                                       dtype=self.float_dtype) for agent in self.agents
        })

        # OBSERVATION
        self.num_other_agents_observed = config.get('num_other_agents_observed')
        if not 0 < self.num_other_agents_observed <= self.num_agents - 1 or self.num_other_agents_observed == "all":
            self.num_other_agents_observed = self.max_num_agents - 1
        self.max_seeing_angle = config.get('max_seeing_angle')
        if not 0 < self.max_seeing_angle <= np.pi:
            self.max_seeing_angle = np.pi
        self.max_seeing_distance = config.get('max_seeing_distance')
        if not 0 < self.max_seeing_distance <= self.grid_diagonal:
            self.max_seeing_distance = self.grid_diagonal

        self.use_polar_coordinate = config.get('use_polar_coordinate')

        # The observation space is a dict of Box spaces, one per agent.
        self.observation_space = spaces.Dict({
            agent.agent_id: spaces.Box(low=-1, high=1, shape=(4 * self.num_other_agents_observed + 3,),
                                       dtype=self.float_dtype) for agent in self.agents
        })

        # REWARDS
        self.starving_penalty_for_predator = config.get('starving_penalty_for_predator')
        self.eating_reward_for_predator = config.get('eating_reward_for_predator')
        self.surviving_reward_for_prey = config.get('surviving_reward_for_prey')
        self.death_penalty_for_prey = config.get('death_penalty_for_prey')
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
        obs = np.zeros(4 * self.num_other_agents_observed + 3, dtype=self.float_dtype)

        # Generate observation for each agent
        obs[0] = agent.loc_x / self.grid_diagonal
        obs[1] = agent.loc_y / self.grid_diagonal
        # modulo 2pi to avoid large values
        obs[2] = agent.heading % (2 * np.pi) / (2 * np.pi)

        j = 3  # start adding at this index after adding the initial properties

        # Remove the agent itself and the agents that are not in the game
        other_agents = [other_agent for other_agent in self.agents if
                        other_agent is not agent and other_agent.still_in_game == 1]

        # Observation of the other agents
        if self.num_other_agents_observed < self.num_agents:
            # Sort the agents by distance
            other_agents = sorted(other_agents, key=lambda other_agent: ComputeDistance(agent, other_agent))

        number_of_observed_agent = 0

        for other in other_agents:
            dist = ComputeDistance(agent, other)
            if dist < self.max_seeing_distance:
                direction = ComputeAngle(agent, other)
                if abs(direction) < self.max_seeing_angle:
                    obs[j], obs[j + 1] = self._observation_pos(agent, other)                      # relative position
                    obs[j + 2] = ((other.heading - agent.heading) % (2 * np.pi) - np.pi) / np.pi  # relative heading
                    obs[j + 3] = other.agent_type
                    j += 4
            number_of_observed_agent += 1
            if number_of_observed_agent == self.num_other_agents_observed:
                break

        return obs

    def _get_observation_list(self):
        return {agent.agent_id: self._generate_observation(agent) for agent in self.agents if agent.still_in_game == 1}

    def reset(self, seed=None, options=None):
        # Reset time to the beginning
        self.timestep = 0
        self.num_preys = self.ini_num_preys

        for i, agent in enumerate(self.agents):
            agent.loc_x = self.np_random.random() * self.stage_size
            agent.loc_y = self.np_random.random() * self.stage_size
            agent.speed_x = 0.0
            agent.speed_y = 0.0
            agent.heading = self.np_random.random() * 2 * np.pi
            if i < self.num_agents:
                agent.still_in_game = 1  # True = 1 and False = 0
            else:
                agent.still_in_game = 0

        observation_list = self._get_observation_list()
        return observation_list, {}

    def step(self, action_list):
        self.timestep += 1

        self._simulate_one_step(action_list)

        observation_list = self._get_observation_list()

        reward_list = self._get_reward(action_list)
        dones, truncateds = self._get_done()

        return observation_list, reward_list, dones, truncateds, {}

    def render(self):
        # TODO
        raise NotImplementedError()

    def _simulate_one_step(self, action_list):
        for agent in self.agents:
            if agent.still_in_game == 1:
                # get the actions for this agent
                self_force_amplitude, self_force_orientation = action_list.get(agent.agent_id)

                agent.heading = (agent.heading + self_force_orientation) % (2 * np.pi)
                acceleration_x = self_force_amplitude * math.cos(agent.heading)
                acceleration_y = self_force_amplitude * math.sin(agent.heading)

                # DRAGGING FORCE
                dragging_force_amplitude = math.sqrt(
                    agent.speed_x ** 2 + agent.speed_y ** 2) ** 2 * self.dragging_force_coefficient
                # opposed to the speed direction of previous step
                dragging_force_orientation = math.atan2(agent.speed_y, agent.speed_x) - math.pi
                acceleration_x += dragging_force_amplitude * math.cos(dragging_force_orientation)
                acceleration_y += dragging_force_amplitude * math.sin(dragging_force_orientation)

                # BUMP INTO OTHER AGENTS
                # contact_force
                if self.contact_force_coefficient > 0:
                    for other_agent in self.agents:
                        if agent.agent_type == other_agent.agent_type and agent.agent_id != other_agent.agent_id:
                            dist = ComputeDistance(agent, other_agent)
                            if dist < other_agent.radius + agent.radius:
                                # compute the contact force
                                contact_force_amplitude = self.contact_force_coefficient * (
                                        other_agent.radius + agent.radius - dist)
                                contact_force_orientation = math.atan2(
                                    other_agent.loc_y - agent.loc_y,
                                    other_agent.loc_x - agent.loc_x
                                ) - math.pi  # opposed to the contact direction

                                acceleration_x += contact_force_amplitude * math.cos(contact_force_orientation)
                                acceleration_y += contact_force_amplitude * math.sin(contact_force_orientation)

                # WALL BOUNCING
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

                    if is_touching_edge_x and self.wall_contact_force_coefficient > 0:
                        # you can rarely have contact with two walls at the same time
                        contact_force_amplitude_x = self.wall_contact_force_coefficient * (
                            agent.radius - agent.loc_x if agent.loc_x < agent.radius
                            else agent.loc_x - self.stage_size + agent.radius
                        )
                        acceleration_x += sign(self.stage_size / 2 - agent.loc_x) * contact_force_amplitude_x

                    if is_touching_edge_y and self.wall_contact_force_coefficient > 0:
                        contact_force_amplitude_y = self.wall_contact_force_coefficient * (
                            agent.radius - agent.loc_y if agent.loc_y < agent.radius
                            else agent.loc_y - self.stage_size + agent.radius
                        )

                        acceleration_y += sign(self.stage_size / 2 - agent.loc_y) * contact_force_amplitude_y

                # # UPDATE ACCELERATION/SPEED/POSITION
                # Compute the amplitude and turn in polar coordinate
                acceleration_amplitude = math.sqrt(acceleration_x ** 2 + acceleration_y ** 2) / (agent.radius ** 3 * self.agent_density)
                acceleration_orientation = math.atan2(acceleration_y, acceleration_x)

                # Compute the speed using projection
                agent.speed_x += acceleration_amplitude * math.cos(acceleration_orientation)
                agent.speed_y += acceleration_amplitude * math.sin(acceleration_orientation)

                # Note : agent.heading was updated right after getting the action list
                # Update the agent's location
                agent.loc_x += agent.speed_x
                agent.loc_y += agent.speed_y

                # periodic boundary
                if self.periodical_boundary is True:
                    agent.loc_x = agent.loc_x % self.stage_size
                    agent.loc_y = agent.loc_y % self.stage_size


    def _get_reward(self, action_list):
        # Initialize rewards
        reward_list = {agent.agent_id: 0 for agent in self.agents}

        for agent in self.agents:
            if agent.still_in_game:
                # AGENT EATEN OR NOT
                if agent.agent_type == 0:  # 0 for prey, 1 for predator
                    reward_list[agent.agent_id] += self.surviving_reward_for_prey

                    for other_agent in self.agents:
                        # We compare distance with predators
                        if other_agent.agent_type == 1:
                            dist = ComputeDistance(agent, other_agent)
                            eating_distance = other_agent.radius + agent.radius
                            if dist < eating_distance:
                                # The prey is eaten
                                reward_list[other_agent.agent_id] += self.eating_reward_for_predator
                                reward_list[agent.agent_id] += self.death_penalty_for_prey
                                self.num_preys -= 1
                                agent.still_in_game = 0
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
        # True at the end
        done_for_all = self.timestep >= self.episode_length or self.num_preys == 0
        dones = {agent.agent_id: done_for_all for agent in self.agents}
        # True when agent are eaten or if agent id is over the num_agent (but under the max_num_agent)
        truncateds = {agent.agent_id: agent.still_in_game == 0 for agent in self.agents}

        dones['__all__'] = done_for_all
        truncateds['__all__'] = done_for_all

        return dones, truncateds
