from gym.spaces import Box
import numpy as np
import copy
import numpy as np
from gym import spaces
from gym.utils import seeding
import math

_OBSERVATIONS = "observations"
_ACTIONS = "sampled_actions"
_REWARDS = "rewards"
_LOC_X = "loc_x"
_LOC_Y = "loc_y"
_SP = "speed"
_ORI = "orientation"
_ACC = "acceleration"
_SIG = "still_in_the_game"
_DONE = "done"


def sign(x):
    if x == 0:
        return 0
    elif x > 0:
        return 1
    else:
        return -1


def ComputeDistance(agent_id1, agent_id2):
    return math.sqrt(
        ((agent_id1.loc_x - agent_id2.loc_x) ** 2)
        + ((agent_id1.loc_y - agent_id2.loc_y) ** 2)
    )

def ComputeAngle(agent_id1, agent_id2):
    return math.degrees(
        math.atan2(
            agent_id1.loc_y - agent_id2.loc_y,
            agent_id1.loc_x - agent_id2.loc_x
        ) - agent_id1.orientation
    )


class Agent:
    _id = 0

    def __init__(self, size=None, agent_type=None,
                 loc_x=None, loc_y=None, orientation=None,
                 speed_x=None, speed_y=None,
                 acceleration_amplitude=None, acceleration_orientation=None, still_in_game=True):
        self.id = Agent._id
        self.agent_type = agent_type
        self.size = size
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.orientation = orientation
        self.acceleration_amplitude = acceleration_amplitude
        self.acceleration_orientation = acceleration_orientation
        self.still_in_game = still_in_game


class CustomEnvironment(object):
    """
    task description:
    all agents try to meet at the same target_position of a 2D-plane
    """

    def __init__(self,
                 num_preys=50,
                 num_predators=1,
                 stage_size=100.0,
                 episode_length=240,
                 preparation_length=120,
                 # Physics
                 draging_force_coefficient=0,
                 contact_force_coefficient=0,
                 wall_contact_force_coefficient=0,
                 prey_size=0.2,
                 predator_size=0.2,
                 min_speed=0.2,
                 max_speed=0.5,
                 max_acceleration=0.5,
                 min_acceleration=-0.5,
                 max_turn=np.pi / 4,
                 min_turn=-np.pi / 4,
                 num_acceleration_levels=5,
                 num_turn_levels=5,
                 starving_penalty_for_predator=-1.0,
                 eating_reward_for_predator=1.0,
                 surviving_reward_for_prey=1.0,
                 death_penalty_for_prey=-1.0,
                 edge_hit_penalty=-0.1,
                 end_of_game_penalty=-10,
                 end_of_game_reward=10,
                 use_energy_cost=True,
                 use_full_observation=True,
                 max_seeing_angle=None,
                 max_seeing_distance=None,
                 num_other_agents_observed=None,
                 use_time_in_observation=True,
                 use_polar_coordinate=False,
                 seed=None,
                 ):
        self.float_dtype = np.float32
        self.int_dtype = np.int32
        # small number to prevent indeterminate cases
        self.eps = self.float_dtype(1e-10)

        # Seeding
        self.np_random = np.random
        if seed is not None:
            self.np_random, seed = seeding.np_random(seed)

        # ENVIRONMENT
        # Length and preparation
        assert episode_length > 0
        self.episode_length = episode_length
        self.preparation_length = preparation_length

        # Square 2D grid
        assert stage_size > 0
        self.stage_size = self.float_dtype(stage_size)
        self.grid_diagonal = self.stage_size * np.sqrt(2)

        # AGENTS
        assert num_preys > 0
        assert num_predators > 0
        self.num_preys = num_preys
        self.num_predators = num_predators
        self.num_agents = self.num_preys + self.num_predators

        assert 0 <= prey_size <= 1
        assert 0 <= predator_size <= 1
        self.prey_size = prey_size
        self.predator_size = predator_size

        # Initialize agent objects
        self.agents = []
        for i in range(self.num_agents):
            if i < self.num_preys:
                agent_type = 'prey'
                size = self.prey_size
            else:
                agent_type = 'predator'
                size = self.predator_size
            self.agents.append(Agent(agent_type=agent_type, size=size))

        # PHYSICS
        # Eating distance
        # Distance margin between agents for eating
        # If a predator is closer than this to a prey,
        # the predator eats the prey
        self.eating_distance = prey_size + predator_size

        # Set the max speed level
        self.max_speed = self.float_dtype(max_speed)
        self.min_speed = self.float_dtype(min_speed)

        self.draging_force_coefficient = draging_force_coefficient
        self.contact_force_coefficient = contact_force_coefficient
        self.wall_contact_force_coefficient = wall_contact_force_coefficient

        # ACTION SPACE
        # The num_acceleration and num_turn levels refer to the number of
        # uniformly-spaced levels between (min_acceleration and max_acceleration)
        # and (min_turn and max_turn), respectively.
        assert num_acceleration_levels >= 0
        assert num_turn_levels >= 0
        self.num_acceleration_levels = num_acceleration_levels
        self.num_turn_levels = num_turn_levels
        self.max_acceleration = self.float_dtype(max_acceleration)
        self.min_acceleration = self.float_dtype(min_acceleration)

        self.max_turn = self.float_dtype(max_turn)
        self.min_turn = self.float_dtype(min_turn)

        # Acceleration actions
        self.acceleration_actions = np.linspace(
            self.min_acceleration, self.max_acceleration, self.num_acceleration_levels
        )
        # Add action 0 - this will be the no-op, or 0 acceleration
        self.acceleration_actions = np.insert(self.acceleration_actions, 0, 0).astype(
            self.float_dtype
        )

        # Turn actions
        self.turn_actions = np.linspace(
            self.min_turn, self.max_turn, self.num_turn_levels
        )
        # Add action 0 - this will be the no-op, or 0 turn
        self.turn_actions = np.insert(self.turn_actions, 0, 0).astype(self.float_dtype)

        # These will be set during reset (see below)
        self.timestep = None

        self.action_space = {
            agent_id: spaces.MultiDiscrete(
                (len(self.acceleration_actions), len(self.turn_actions))
            )
            for agent_id in range(self.num_agents)
        }

        # OBSERVATION SPACE
        if sum(var for var in [use_full_observation, num_other_agents_observed is not None,
                               (max_seeing_angle is not None and max_seeing_distance is not None)]) != 1:
            raise ValueError("Only one of use_full_observation, num_other_agents_observed, and max_seeing_angle should be set.")

        self.observation_space = None  # Note: this will be set via the env_wrapper
        self.use_full_observation = use_full_observation
        self.num_other_agents_observed = self.num_agents if num_other_agents_observed is None else num_other_agents_observed
        self.max_seeing_angle = stage_size / np.sqrt(2) if max_seeing_angle is None else max_seeing_angle
        self.max_seeing_distance = np.pi if max_seeing_distance is None else max_seeing_distance

        self.use_time_in_observation = use_time_in_observation
        self.use_polar_coordinate = use_polar_coordinate

        # Used in generate_observation()
        self.init_obs = None  # Will be set later in generate_observation()

        # REWARDS
        self.starving_penalty_for_predator = starving_penalty_for_predator
        self.eating_reward_for_predator = eating_reward_for_predator
        self.surviving_reward_for_prey = surviving_reward_for_prey
        self.death_penalty_for_prey = death_penalty_for_prey
        self.edge_hit_penalty = edge_hit_penalty
        self.end_of_game_penalty = end_of_game_penalty
        self.end_of_game_reward = end_of_game_reward
        self.use_energy_cost = use_energy_cost

    name = "CustomEnv"

    def _generate_observation(self, agent):
        """
        Generate and return the observations for every agent.
        """
        # initialize obs as an empty list of correct size
        obs = np.zeros(5*self.num_agents, dtype=self.float_dtype)

        # Generate observation for each agent
        obs[0] = agent.speed_x / self.max_speed
        obs[1] = agent.speed_y / self.max_speed
        obs[2] = agent.acceleration_amplitude / self.max_acceleration
        obs[3] = agent.acceleration_orientation / np.pi

        obs[4] = agent.loc_x / self.stage_size
        obs[5] = agent.loc_y / self.stage_size
        obs[6] = agent.orientation / np.pi
        obs[7] = agent.size
        obs[8] = agent.still_in_game

        # Add the agent position and speed to the observation
        j = 9  # start adding at this index after adding the initial properties
        for i, other in enumerate(self.agents):
            if other is agent:
                continue
            obs[j] = (other.loc_x - agent.loc_x) / self.stage_size
            obs[j + 1] = (other.loc_y - agent.loc_y) / self.stage_size
            obs[j + 2] = (other.orientation - agent.orientation) / np.pi
            obs[j + 3] = other.size
            obs[j + 4] = other.still_in_game
            j += 5  # move to the next 5 indices for the next agent

        return obs

    def _get_observation_list(self):
        return [self._generate_observation(agent) for agent in self.agents]

    def _get_state(self):
        """
        concat all agent's observation to construct state info
        """
        return np.concatenate(self._generate_observation(), axis=0)  # state shape is (-1, )

    def reset(self):
        """
        Env reset(). when done is called
        """
        # Reset time to the beginning
        self.timestep = 0

        for i, agent in enumerate(self.agents):
            agent.loc_x = self.stage_size * self.np_random.rand()
            agent.loc_y = self.stage_size * self.np_random.rand()
            agent.speed = 0.0
            agent.orientation = self.np_random.rand() * 2 * np.pi
            agent.acceleration_amplitude = 0.0
            agent.acceleration_orientation = 0.0
            agent.still_in_game = True

        observation_list = self._get_observation_list()
        state = self._get_state()
        return observation_list, state

    def step(self, action_list):
        for _ in range(self.action_effective_step):
            self._simulate_one_step(action_list)

        observation_list = self._get_observation_list()
        state = self._get_state()
        reward_list, team_reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        return (observation_list, state), (reward_list, team_reward), done, info

    def render(self):
        # TODO
        raise NotImplementedError()

    def _simulate_one_step(self, action_list):
        self.timestep += 1

        kNumActions = 2

        assert 0 < self.timestep <= self.episode_length

        for agent in self.agents:
            # get the actions for this agent
            self_force_amplitude = agent.acceleration_action[agent.action_index[0]]
            self_force_orientation = agent.orientation + agent.turn_action[agent.action_index[1]]
            acceleration_x = self_force_amplitude * math.cos(self_force_orientation)
            acceleration_y = self_force_amplitude * math.sin(self_force_orientation)

            # set the energy cost penalty
            if self.use_energy_cost:
                agent.energy_cost_penalty = - (
                        self_force_amplitude / self.max_acceleration + abs(self_force_orientation) / self.max_turn) / 100

            # DRAGGING FORCE
            dragging_force_amplitude = agent.speed * self.draging_force_coefficient
            dragging_force_orientation = agent.orientation - math.pi  # opposed to the current speed
            acceleration_x += dragging_force_amplitude * math.cos(dragging_force_orientation)
            acceleration_y += dragging_force_amplitude * math.sin(dragging_force_orientation)

            # BUMP INTO OTHER AGENTS
            # contact_force
            if self.contact_force_coefficient > 0:
                for other_agent in self.agents:
                    if agent.agent_type == other_agent.agent_type and agent.id != other_agent.id:
                        dist = ComputeDistance(
                            agent.loc_x,
                            agent.loc_y,
                            other_agent.loc_x,
                            other_agent.loc_y
                        )
                        if dist < other_agent.size + agent.radius:
                            contact_force_amplitude = self.contact_force_coefficient * (other_agent.radius + agent.radius - dist)
                            contact_force_orientation = ComputeAngle(agent.loc_x, agent.loc_y, agent.orientation, other_agent.loc_x,
                                                                     other_agent.loc_y) - math.pi  # opposed to the contact direction
                            acceleration_x += contact_force_amplitude * math.cos(contact_force_orientation)
                            acceleration_y += contact_force_amplitude * math.sin(contact_force_orientation)

            # WALL BOUNCING
            # Check if the agent has crossed the edge
            is_touching_edge = (
                    agent.loc_x < agent.radius
                    or agent.loc_x > self.stage_size - agent.radius
                    or agent.loc_y < agent.radius
                    or agent.loc_y > self.stage_size - agent.radius
            )
            if is_touching_edge and self.wall_contact_force_coefficient > 0:
                contact_force_amplitude_x = self.wall_contact_force_coefficient * min(-agent.loc_x + agent.radius,
                                                                                      agent.radius - self.stage_size + agent.loc_x)
                contact_force_amplitude_y = self.wall_contact_force_coefficient * min(-agent.loc_y + agent.radius,
                                                                                      agent.radius - self.stage_size + agent.loc_y)
                acceleration_x += sign(self.stage_size / 2 - agent.loc_x) * contact_force_amplitude_x
                acceleration_y += sign(self.stage_size / 2 - agent.loc_y) * contact_force_amplitude_y

            # UPDATE ACCELERATION/SPEED
            # Compute the amplitude and turn in polar coordinate
            agent.acceleration_amplitude = math.sqrt(acceleration_x ** 2 + acceleration_y ** 2)
            agent.acceleration_orientation = math.atan2(acceleration_y, acceleration_x)

            # Compute the speed using projection
            speed_x = agent.speed * math.cos(agent.orientation) + acceleration_x
            speed_y = agent.speed * math.sin(agent.orientation) + acceleration_y

            # Update the agent's acceleration and directions
            agent.acceleration = agent.acceleration_amplitude * agent.still_in_game
            agent.orientation = math.atan2(speed_y, speed_x) * agent.still_in_game
            agent.speed = math.sqrt(speed_x ** 2 + speed_y ** 2) * agent.still_in_game

            # speed clipping
            if agent.speed > self.max_speed:
                agent.speed = self.max_speed * agent.still_in_game

            # UPDATE POSITION
            # Update the agent's location
            agent.loc_x += agent.speed * math.cos(agent.orientation)
            agent.loc_y += agent.speed * math.sin(agent.orientation)


    def _get_reward(self):
        reward_list = [0] * self.num_agents

        for i, agent in enumerate(self.agents):
            # Initialize rewards
            reward_list[i] = 0

            if agent.still_in_game:
                is_prey = not agent.agent_type  # 0 for prey, 1 for predator

                if is_prey:
                    reward_list[i] += self.surviving_reward_for_prey
                    min_dist = self.stage_size * math.sqrt(2.0)

                    for j, other_agent in enumerate(self.agents):
                        # We compare distance with predators
                        is_predator = other_agent.agent_type == 1
                        if is_predator:
                            dist = ComputeDistance(agent, other_agent)
                            if dist < min_dist:
                                min_dist = dist
                                nearest_predator_id = j
                    if min_dist < self.eating_distance:
                        # The prey is eaten
                        reward_list[nearest_predator_id] += self.eating_reward_for_predator
                        reward_list[i] += self.death_penalty_for_prey
                        self.num_preys -= 1
                        agent.still_in_game[i] = 0
                else:  # is_predator
                    reward_list[i] += self.starving_penalty_for_predator

            # Add the edge hit penalty
            has_crossed_edge = (
                    agent.loc_x < agent.radius
                    or agent.loc_x > self.stage_size - agent.radius
                    or agent.loc_y < agent.radius
                    or agent.loc_y > self.stage_size - agent.radius
            )

            # EDGE CROSSING
            # Clip x and y if agent has crossed edge
            if has_crossed_edge:
                if agent.loc_x < 0:
                    agent.loc_x = 0.0
                elif agent.loc_x > self.stage_size:
                    agent.loc_x = self.stage_size

                if agent.loc_y < 0:
                    agent.loc_y = 0.0
                elif agent.loc_y > self.stage_size:
                    agent.loc_y = self.stage_size

                reward_list[i] += self.edge_hit_penalty

            # Add the energy efficiency penalty
            reward_list[i] += self.energy_cost_penalty[i] / 2

        if self.env_timestep == self.episode_length:
            self.done = 1

        # reward_list[-1] is the global reward (i.e., sum of individual rewards)
        return (reward_list, sum(reward_list))

    def _get_done(self):
        if self.timestep >= self.max_step_count:
            return True
        return False

    def _get_info(self):
        info = {
            'step_count': self.timestep,
            'max_step_count': self.max_step_count
        }
        return info


