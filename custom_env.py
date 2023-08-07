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
    return math.degrees(
        math.atan2(
            agent1.loc_y - agent2.loc_y,
            agent1.loc_x - agent2.loc_x
        ) - agent1.orientation
    )


class ParticuleAgent:
    def __init__(self, id=None, size=None, agent_type=None,
                 loc_x=None, loc_y=None, orientation=None,
                 speed_x=None, speed_y=None,
                 acceleration_amplitude=None, acceleration_orientation=None, still_in_game=True):
        self.agent_id = id
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

        self.episode_length = config.get('episode_length')
        assert self.episode_length > 0
        self.preparation_length = config.get('preparation_length')

        self.stage_size = config.get('stage_size')
        assert self.stage_size > 0
        self.grid_diagonal = self.stage_size * np.sqrt(2)

        self.num_preys = config.get('num_preys')
        self.num_predators = config.get('num_predators')
        assert self.num_preys > 0
        assert self.num_predators > 0
        self.num_agents = self.num_preys + self.num_predators

        self.prey_size = config.get('prey_size')
        self.predator_size = config.get('predator_size')
        assert 0 <= self.prey_size <= 1
        assert 0 <= self.predator_size <= 1

        self.agents = []
        for i in range(self.num_agents):
            if i < self.num_preys:
                agent_type = 0 # for preys
                size = self.prey_size
            else:
                agent_type = 1 # for predators
                size = self.predator_size
            self.agents.append(ParticuleAgent(id=i, agent_type=agent_type, size=size))
        self._agent_ids = {agent.agent_id for agent in self.agents}

        self.eating_distance = self.prey_size + self.predator_size

        self.max_speed = config.get('max_speed')
        self.min_speed = config.get('min_speed')

        self.dragging_force_coefficient = config.get('dragging_force_coefficient')
        self.contact_force_coefficient = config.get('contact_force_coefficient')
        self.wall_contact_force_coefficient = config.get('wall_contact_force_coefficient')

        self.num_acceleration_levels = config.get('num_acceleration_levels')
        self.num_turn_levels = config.get('num_turn_levels')
        self.max_acceleration = config.get('max_acceleration')
        self.min_acceleration = config.get('min_acceleration')

        self.max_turn = config.get('max_turn')
        self.min_turn = config.get('min_turn')

        assert self.num_acceleration_levels >= 0
        assert self.num_turn_levels >= 0

        self.acceleration_actions = np.linspace(
            self.min_acceleration, self.max_acceleration, self.num_acceleration_levels
        )
        self.acceleration_actions = np.insert(self.acceleration_actions, 0, 0).astype(self.float_dtype)

        self.turn_actions = np.linspace(
            self.min_turn, self.max_turn, self.num_turn_levels
        )
        self.turn_actions = np.insert(self.turn_actions, 0, 0).astype(self.float_dtype)

        self.timestep = None

        self.action_space = spaces.Dict({
            agent.agent_id: spaces.MultiDiscrete((len(self.acceleration_actions), len(self.turn_actions))) for agent in self.agents
        })

        use_full_observation = config.get('use_full_observation')
        num_other_agents_observed = config.get('num_other_agents_observed')
        max_seeing_angle = config.get('max_seeing_angle')
        max_seeing_distance = config.get('max_seeing_distance')

        if sum(var for var in [use_full_observation, num_other_agents_observed is not None,
                               (max_seeing_angle is not None and max_seeing_distance is not None)]) != 1:
            raise ValueError("Only one of use_full_observation, num_other_agents_observed, and max_seeing_angle should be set.")

        low = np.full((6 * self.num_agents + 4,), -1)
        high = np.full((6 * self.num_agents + 4,), 1)

        self.observation_space = spaces.Dict({
            agent.agent_id: spaces.Box(low=low, high=high, dtype=self.float_dtype) for agent in self.agents
        })

        self.use_full_observation = use_full_observation
        self.num_other_agents_observed = self.num_agents if num_other_agents_observed is None else num_other_agents_observed
        self.max_seeing_angle = self.stage_size / np.sqrt(2) if max_seeing_angle is None else max_seeing_angle
        self.max_seeing_distance = np.pi if max_seeing_distance is None else max_seeing_distance

        self.use_time_in_observation = config.get('use_time_in_observation')
        self.use_polar_coordinate = config.get('use_polar_coordinate')

        self.init_obs = None

        self.starving_penalty_for_predator = config.get('starving_penalty_for_predator')
        self.eating_reward_for_predator = config.get('eating_reward_for_predator')
        self.surviving_reward_for_prey = config.get('surviving_reward_for_prey')
        self.death_penalty_for_prey = config.get('death_penalty_for_prey')
        self.edge_hit_penalty = config.get('edge_hit_penalty')
        self.end_of_game_penalty = config.get('end_of_game_penalty')
        self.end_of_game_reward = config.get('end_of_game_reward')
        self.use_energy_cost = config.get('use_energy_cost')

    def _generate_observation(self, agent):
        """
        Generate and return the observations for every agent.
        """
        # initialize obs as an empty list of correct size
        obs = np.zeros(6 * self.num_agents + 4, dtype=self.float_dtype)

        # Generate observation for each agent
        obs[0] = agent.speed_x / self.max_speed
        obs[1] = agent.speed_y / self.max_speed
        obs[2] = agent.acceleration_amplitude / self.max_acceleration
        obs[3] = agent.acceleration_orientation / (2*np.pi)

        obs[4] = agent.loc_x / self.stage_size
        obs[5] = agent.loc_y / self.stage_size
        obs[6] = agent.orientation / (2*np.pi)
        obs[7] = agent.size
        obs[8] = agent.still_in_game
        obs[9] = agent.agent_type

        # Add the agent position and speed to the observation
        j = 10  # start adding at this index after adding the initial properties
        for other in self.agents:
            if other is agent:
                continue
            obs[j] = (other.loc_x - agent.loc_x) / self.stage_size
            obs[j + 1] = (other.loc_y - agent.loc_y) / self.stage_size
            obs[j + 2] = (other.orientation - agent.orientation) / (2*np.pi)
            obs[j + 3] = other.size
            obs[j + 4] = other.still_in_game
            obs[j + 5] = other.agent_type
            j += 6  # move to the next 6 indices for the next agent

        return obs

    def _get_observation_list(self):
        return {agent.agent_id: self._generate_observation(agent) for agent in self.agents}

    def reset(self, seed=None, options=None):
        """
        Env reset(). when done is called
        """
        # Reset time to the beginning
        self.timestep = 0

        for agent in self.agents:
            agent.loc_x = self.np_random.random()
            agent.loc_y = self.np_random.random()
            agent.speed_x = 0.0
            agent.speed_y = 0.0
            agent.orientation = self.np_random.random()
            agent.acceleration_amplitude = 0.0
            agent.acceleration_orientation = 0.0
            agent.still_in_game = 1 # True = 1 and False = 0

        observation_list = self._get_observation_list()
        return observation_list, {}

    def step(self, action_list):
        self.timestep += 1
        energy_cost_penalty = self._simulate_one_step(action_list)

        observation_list = self._get_observation_list()

        reward_list = self._get_reward(energy_cost_penalty)
        dones, truncateds = self._get_done()
        return (observation_list, 
                reward_list,             
                dones,
                truncateds, 
                {},
        )

    def render(self):
        # TODO
        raise NotImplementedError()

    def _simulate_one_step(self, action_list):
        energy_cost_penalty = []
        for agent in self.agents:
            # get the actions for this agent
            self_force_amplitude, self_force_orientation = action_list.get(agent.agent_id)
            self_force_orientation = agent.orientation + self_force_orientation
            acceleration_x = self_force_amplitude * math.cos(self_force_orientation)
            acceleration_y = self_force_amplitude * math.sin(self_force_orientation)

            # set the energy cost penalty
            if self.use_energy_cost:
                energy_cost_penalty.append( -(self_force_amplitude / self.max_acceleration + abs(self_force_orientation) / self.max_turn) / 100)

            # DRAGGING FORCE
            dragging_force_amplitude = math.sqrt(agent.speed_x**2 + agent.speed_y**2) * self.dragging_force_coefficient
            dragging_force_orientation = agent.orientation - math.pi  # opposed to the current speed
            acceleration_x += dragging_force_amplitude * math.cos(dragging_force_orientation)
            acceleration_y += dragging_force_amplitude * math.sin(dragging_force_orientation)

            # BUMP INTO OTHER AGENTS
            # contact_force
            if self.contact_force_coefficient > 0:
                for other_agent in self.agents:
                    if agent.agent_type == other_agent.agent_type and agent.agent_id != other_agent.agent_id:
                        dist = ComputeDistance(agent, other_agent)
                        if dist < other_agent.size + agent.size:
                            contact_force_amplitude = self.contact_force_coefficient * (other_agent.size + agent.size - dist)
                            contact_force_orientation = ComputeAngle(agent, other_agent) - math.pi  # opposed to the contact direction
                            acceleration_x += contact_force_amplitude * math.cos(contact_force_orientation)
                            acceleration_y += contact_force_amplitude * math.sin(contact_force_orientation)

            # WALL BOUNCING
            # Check if the agent has crossed the edge
            is_touching_edge = (
                    agent.loc_x < agent.size
                    or agent.loc_x > self.stage_size - agent.size
                    or agent.loc_y < agent.size
                    or agent.loc_y > self.stage_size - agent.size
            )
            if is_touching_edge and self.wall_contact_force_coefficient > 0:
                contact_force_amplitude_x = self.wall_contact_force_coefficient * min(-agent.loc_x + agent.size,
                                                                                      agent.size - self.stage_size + agent.loc_x)
                contact_force_amplitude_y = self.wall_contact_force_coefficient * min(-agent.loc_y + agent.size,
                                                                                      agent.size - self.stage_size + agent.loc_y)
                acceleration_x += sign(self.stage_size / 2 - agent.loc_x) * contact_force_amplitude_x
                acceleration_y += sign(self.stage_size / 2 - agent.loc_y) * contact_force_amplitude_y

            # UPDATE ACCELERATION/SPEED
            # Compute the amplitude and turn in polar coordinate
            agent.acceleration_amplitude = math.sqrt(acceleration_x ** 2 + acceleration_y ** 2)
            if agent.acceleration_amplitude > self.max_acceleration: 
                acceleration_x = acceleration_x * self.max_acceleration/agent.acceleration_amplitude
                acceleration_y = acceleration_y * self.max_acceleration/agent.acceleration_amplitude
                agent.acceleration_amplitude = self.max_acceleration
            agent.acceleration_orientation = math.atan2(acceleration_y, acceleration_x)

            # Compute the speed using projection
            speed_x = agent.speed_x + acceleration_x
            speed_y = agent.speed_y + acceleration_y

            # Update the agent's acceleration and directions
            agent.orientation = math.atan2(speed_y, speed_x) * agent.still_in_game
            agent.speed = math.sqrt(speed_x ** 2 + speed_y ** 2) * agent.still_in_game

            # speed clipping
            if agent.speed > self.max_speed:
                agent.speed_x = agent.speed_x * self.max_speed/agent.speed
                agent.speed_y = agent.speed_y * self.max_speed/agent.speed

            # UPDATE POSITION
            # Update the agent's location
            agent.loc_x += agent.speed_x
            agent.loc_y += agent.speed_y

        return energy_cost_penalty

    def _get_reward(self, energy_cost_penalty):
        # Initialize rewards
        reward_list = {agent.agent_id: 0 for agent in self.agents}

        for i, agent in enumerate(self.agents):
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
                        agent.still_in_game = 0
                else:  # is_predator
                    reward_list[i] += self.starving_penalty_for_predator

            # Add the edge hit penalty
            has_crossed_edge = (
                    agent.loc_x < agent.size
                    or agent.loc_x > self.stage_size - agent.size
                    or agent.loc_y < agent.size
                    or agent.loc_y > self.stage_size - agent.size
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
            reward_list[i] += energy_cost_penalty[i] / 2

        return reward_list

    def _get_done(self):
        # Assuming that all agents terminate at the same time.
        done_for_all = self.timestep >= self.episode_length

        dones = {agent_id: done_for_all for agent_id in self._agent_ids}
        dones['__all__'] = done_for_all
        truncateds = dones

        return dones, truncateds


if __name__ == "__main__":
    from config import run_config
    from ray import tune

    # From https://docs.ray.io/en/latest/tune/examples/includes/pb2_ppo_example.html
    # and https://github.com/ray-project/ray/issues/35923
    from ray.tune.schedulers import PopulationBasedTraining
    import random
        def explore(config):
        # Ensure we collect enough timesteps to do sgd.
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # Ensure we run at least one sgd iter.
        if config["lambda"] > 1:
            config["lambda"] = 1
        config["train_batch_size"] = int(config["train_batch_size"])
        return config
        
    pbt_scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=5,
        quantile_fraction=0.25,
        # Specifies the search space for these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.1, 0.5),
            "lr": lambda: random.uniform(1e-3, 1e-5),
            "train_batch_size": lambda: random.randint(1000, 60000),
        },
        custom_explore_fn=explore,
    
    ## and workarround (that doesn't work)


        
    env = CustomEnvironment(run_config["env"])

    
    tune.run(
        "PPO",
        stop={
            "timesteps_total": 500,
            "episode_reward_mean": 7.99,
        },
        config={
            "env": CustomEnvironment,
            "env_config": run_config["env"],
            "batch_mode": "complete_episodes",
            "num_workers": 0,
            "multiagent": {
                "policies": {
                    "prey": (None, env.observation_space,
                             env.action_space, {}),
                    "predator": (None, env.observation_space,
                                 env.action_space, {}),
                },
                "policy_mapping_fn": lambda x: "prey" if x <=env.num_prey else "predator",
            },
        },
        scheduler=pbt_scheduler)
