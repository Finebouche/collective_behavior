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

        # ENVIRONMENT
        self.episode_length = config.get('episode_length')
        assert self.episode_length > 0
        self.preparation_length = config.get('preparation_length')

        self.stage_size = config.get('stage_size')
        assert self.stage_size > 0
        self.grid_diagonal = self.stage_size * np.sqrt(2)

        # AGENTS
        self.ini_num_preys = config.get('num_preys')
        self.num_preys = self.ini_num_preys
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

        self.eating_distance = self.prey_size + self.predator_size
        #Used by RLlib
        self._agent_ids = {agent.agent_id for agent in self.agents}

        # PHYSICS
        self.max_speed = config.get('max_speed')
        self.min_speed = config.get('min_speed')

        self.dragging_force_coefficient = config.get('dragging_force_coefficient')
        self.contact_force_coefficient = config.get('contact_force_coefficient')
        self.wall_contact_force_coefficient = config.get('wall_contact_force_coefficient')

        self.max_acceleration = config.get('max_acceleration')
        self.min_acceleration = config.get('min_acceleration')

        self.max_turn = config.get('max_turn')
        self.min_turn = config.get('min_turn')

        self.timestep = None

        self.action_space = spaces.Dict({
            agent.agent_id: spaces.Box(low=np.array([self.min_acceleration, self.min_turn]), high=np.array([self.max_acceleration, self.max_turn]), shape=(2,), dtype=self.float_dtype) for agent in self.agents
        })

        # OBSERVATION
        use_full_observation = config.get('use_full_observation')
        num_other_agents_observed = config.get('num_other_agents_observed')
        max_seeing_angle = config.get('max_seeing_angle')
        max_seeing_distance = config.get('max_seeing_distance')

        if sum(var for var in [use_full_observation, num_other_agents_observed is not None,
                               (max_seeing_angle is not None and max_seeing_distance is not None)]) != 1:
            raise ValueError("Only one of use_full_observation, num_other_agents_observed, and max_seeing_angle should be set.")

        self.observation_space = spaces.Dict({
            agent.agent_id: spaces.Box(low=-1, high=1, shape=(5 * self.num_agents + 4,), dtype=self.float_dtype) for agent in self.agents
        })

        self.use_full_observation = use_full_observation
        self.num_other_agents_observed = self.num_agents if num_other_agents_observed is None else num_other_agents_observed
        self.max_seeing_angle = self.stage_size / np.sqrt(2) if max_seeing_angle is None else max_seeing_angle
        self.max_seeing_distance = np.pi if max_seeing_distance is None else max_seeing_distance
        
        self.use_polar_coordinate = config.get('use_polar_coordinate')

        # REWARDS
        self.starving_penalty_for_predator = config.get('starving_penalty_for_predator')
        self.eating_reward_for_predator = config.get('eating_reward_for_predator')
        self.surviving_reward_for_prey = config.get('surviving_reward_for_prey')
        self.death_penalty_for_prey = config.get('death_penalty_for_prey')
        self.edge_hit_penalty = config.get('edge_hit_penalty')
        self.use_energy_cost = config.get('use_energy_cost')

    def _generate_observation(self, agent):
        """
        Generate and return the observations for every agent.
        """
        # initialize obs as an empty list of correct size
        obs = np.zeros(5 * self.num_agents + 4, dtype=self.float_dtype)

        # Generate observation for each agent
        obs[0] = agent.speed_x / self.max_speed
        obs[1] = agent.speed_y / self.max_speed
        obs[2] = agent.acceleration_amplitude / self.max_acceleration
        obs[3] = agent.acceleration_orientation / (2*np.pi)

        obs[4] = agent.loc_x / (self.stage_size*np.sqrt(2))
        obs[5] = agent.loc_y / (self.stage_size*np.sqrt(2))
        obs[6] = agent.orientation / (2*np.pi)
        obs[7] = agent.size
        obs[8] = agent.agent_type

        # Add the agent position and speed to the observation
        j = 9  # start adding at this index after adding the initial properties
        for other in self.agents:
            if other is agent or other.still_in_game == 0:
                continue
            obs[j] = (other.loc_x - agent.loc_x) / (self.stage_size*np.sqrt(2))
            obs[j + 1] = (other.loc_y - agent.loc_y) / (self.stage_size*np.sqrt(2))
            obs[j + 2] = (other.orientation - agent.orientation) / (2*np.pi)
            obs[j + 3] = other.size
            obs[j + 4] = other.agent_type
            j += 5  # move to the next 6 indices for the next agent

        return obs

    def _get_observation_list(self):
        return {agent.agent_id: self._generate_observation(agent) for agent in self.agents if agent.still_in_game==1}

    def reset(self, seed=None, options=None):
        # Reset time to the beginning
        self.timestep = 0
        self.num_preys = self.ini_num_preys
        
        for agent in self.agents:
            agent.loc_x = self.np_random.random() * self.stage_size
            agent.loc_y = self.np_random.random() * self.stage_size
            agent.speed_x = 0.0
            agent.speed_y = 0.0
            agent.orientation = self.np_random.random() * 2 * np.pi
            agent.acceleration_amplitude = 0.0
            agent.acceleration_orientation = 0.0
            agent.still_in_game = 1 # True = 1 and False = 0

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
            if agent.still_in_game:
                # get the actions for this agent
                self_force_amplitude, self_force_orientation = action_list.get(agent.agent_id)
                
                self_force_orientation = agent.orientation + self_force_orientation
                acceleration_x = self_force_amplitude * math.cos(self_force_orientation)
                acceleration_y = self_force_amplitude * math.sin(self_force_orientation)
    
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
    
                # UPDATE ACCELERATION/SPEED/POSITION
                # Compute the amplitude and turn in polar coordinate
                agent.acceleration_amplitude = math.sqrt(acceleration_x ** 2 + acceleration_y ** 2)
                if agent.acceleration_amplitude > self.max_acceleration: 
                    acceleration_x = acceleration_x * self.max_acceleration/agent.acceleration_amplitude
                    acceleration_y = acceleration_y * self.max_acceleration/agent.acceleration_amplitude
                    agent.acceleration_amplitude = self.max_acceleration
                agent.acceleration_orientation = math.atan2(acceleration_y, acceleration_x)
    
                # Compute the speed using projection
                agent.speed_x += acceleration_x
                agent.speed_y += acceleration_y
    
    
                # speed clipping
                agent.speed = math.sqrt(agent.speed_x ** 2 + agent.speed_y ** 2) * agent.still_in_game
                if agent.speed > self.max_speed:
                    agent.speed_x = agent.speed_x * self.max_speed/agent.speed
                    agent.speed_y = agent.speed_y * self.max_speed/agent.speed

                # Update the agent's orientation
                agent.orientation = math.atan2(agent.speed_y , agent.speed_x) * agent.still_in_game
                
                # Update the agent's location
                agent.loc_x += agent.speed_x
                agent.loc_y += agent.speed_y

                # speed clipping
                agent.loc_x = np.clip(agent.loc_x, 0, self.stage_size)
                agent.loc_y = np.clip(agent.loc_y, 0, self.stage_size)

    def _get_reward(self, action_list):
        # Initialize rewards
        reward_list = {agent.agent_id: 0 for agent in self.agents}

        for agent in self.agents:
            if agent.still_in_game:
                # AGENT EATEN OR NOT
                if agent.agent_type == 0:# 0 for prey, 1 for predator
                    reward_list[agent.agent_id] += self.surviving_reward_for_prey

                    for other_agent in self.agents:
                        # We compare distance with predators
                        if other_agent.agent_type == 1:
                            dist = ComputeDistance(agent, other_agent)
                            if dist < other_agent.size + agent.size:
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
                if self.use_energy_cost:
                    self_force_amplitude, self_force_orientation = action_list.get(agent.agent_id)
    
                    energy_cost_penalty = -(
                        self_force_amplitude / self.max_acceleration 
                        + abs(self_force_orientation) / self.max_turn
                    ) / 100
                    reward_list[agent.agent_id] += energy_cost_penalty

        return reward_list

    def _get_done(self):
        # True at the end
        done_for_all = self.timestep >= self.episode_length or self.num_preys == 0
        dones = {agent.agent_id: done_for_all for agent in self.agents}
        # True when agent are eaten
        truncateds = {agent.agent_id: agent.still_in_game == 0 for agent in self.agents}
        
        dones['__all__'] = done_for_all
        truncateds['__all__'] = done_for_all

        return dones, truncateds


if __name__ == "__main__":
    import os
    import ray
    from ray import air, tune
    from ray.rllib.utils.test_utils import check_learning_achieved
    from ray.rllib.policy.policy import PolicySpec
    from ray.rllib.algorithms.ppo import PPOConfig
    
    from custom_env import CustomEnvironment
    from config import run_config
    
    class Args:
        def __init__(self):
            self.run = "PPO"
            self.framework = "torch" # "tf2" or "torch"
            self.stop_iters = 50
            self.stop_timesteps = 100000
            self.stop_reward = 0.1
            self.as_test = False
    
    args = Args()
    
    ray.init()
    env = CustomEnvironment(run_config["env"])
    
    config = (
        PPOConfig()
        .rollouts(rollout_fragment_length="auto", num_rollout_workers=0)
        .environment(CustomEnvironment, env_config=run_config["env"])
        .framework(args.framework)
        .training(num_sgd_iter=10, sgd_minibatch_size=256, train_batch_size=4000)
        .multi_agent(
            policies= {
                "prey": PolicySpec(
                    policy_class=None,  # infer automatically from Algorithm
                    observation_space=env.observation_space[0],  # if None infer automatically from env
                    action_space=env.action_space[0],  # if None infer automatically from env
                    config={"gamma": 0.85},  # use main config plus <- this override here
                ),
                "predator": PolicySpec(
                    policy_class=None,
                    observation_space=env.observation_space[0],
                    action_space=env.action_space[0],
                    config={"gamma": 0.85},
                ),
            },
            policy_mapping_fn = lambda id, *arg, **karg: "prey" if env.agents[id].agent_type == 0 else "predator",
            policies_to_train=["prey", "predator"]
        )
        .rl_module(_enable_rl_module_api=True)
        .training(_enable_learner_api=True)
        .resources(num_gpus=0)
    )
    
    
    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
    
    tuner = tune.Tuner(
        args.run,
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=3),
    )
    results = tuner.fit()
    
    if args.as_test:
        print("Checking if learning goals were achieved")
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()
