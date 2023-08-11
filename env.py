from gymnasium import spaces
import numpy as np
from ray.rllib.env import MultiAgentEnv

class ParticleEntity(object):    
    def __init__(self, id):
        self.id = str(id)
        self.position = None

class Environment(MultiAgentEnv):
    """
    task description:
    all agents try to meet at the same target_position of a 2D-plane
    """
    def __init__(self, config):
        self.agent_count = 2
        self.env_bound = 4
        self.env_dim = 2  # do not change this!
        self.action_effective_step = 1
        self.sparse_reward_flag = False
        self.agents = [ParticleEntity(id=id) for id in range(self.agent_count)]
        self.target_position = None
        self.step_count = 0
        self.max_step_count = 40

        #self.observation_space = spaces.Dict({agent.id : spaces.Box(low=-float('inf'), high=float('inf'), shape=(4,))  for agent in self.agents})

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(4,))

        #self.action_space = spaces.Dict({agent.id : spaces.Box(low=-1, high=1, shape=(2, ))  for agent in self.agents})
        self.action_space = spaces.Box(low=-1, high=1, shape=(2, ))
        self.is_discrete = False

    def reset(self, seed=None, options=None):
        for i, agent in enumerate(self.agents):
            agent.position = np.random.uniform(0, self.env_bound, self.env_dim)  # continue space
        self.target_position = np.random.uniform(0, self.env_bound, self.env_dim)

        observations = self._get_observation()
        return observations, {}

    def step(self, action_list):
        for _ in range(self.action_effective_step):
            self._simulate_one_step(action_list)

        observations = self._get_observation()
        rewards = self._get_reward()
        dones = self._get_done()
        return observations, rewards, dones, dones, {}
    
    def _get_observation(self):
        observation_dict = {}
        for agent in self.agents:
            relative_position = []
            for other in self.agents:
                if other is not agent:
                    relative_position.append(other.position - agent.position)
            observation = np.concatenate([agent.position] + relative_position)  # each observation has a shape of (-1, )
            
            observation_dict[agent.id] = observation * 1.0 / self.env_bound  # always normalize the observation
        return observation_dict
        
    def _simulate_one_step(self, action_list):
        self.step_count += 1
        for agent in self.agents:
            agent.position = agent.position + action_list[agent.id]
            agent.position = agent.position % self.env_bound  # keep the agents in the env_bound

    def _get_reward(self):
        reward_dict = {agent.id : 0 for agent in self.agents}
        if self.sparse_reward_flag:
            for agent in self.agents:
                if (agent.position != self.target_position).any():
                    reward_dict[agent.id] = 0
                else:
                    reward_dict[agent.id] = 1
        else:
            for agent in self.agents:
                distance = np.sqrt(np.sum(np.square(agent.position - self.target_position)))
                reward_dict[agent.id] = -distance  # negative distance

        return reward_dict

    def _get_done(self):
        done_for_all = self.step_count >= self.max_step_count

        dones = {agent.id: done_for_all for agent in self.agents}
        dones['__all__'] = done_for_all
        return dones



if __name__ == "__main__":
    import os
    
    from ray.rllib.algorithms.ppo import PPOConfig
    from env import Environment
    from ray.rllib.policy.policy import PolicySpec
    
    env = Environment({})
    
    
    config = (
        PPOConfig()
        .rollouts(rollout_fragment_length="auto", num_rollout_workers=0)
        .environment(Environment)
        .framework("torch")
        .multi_agent(
            policies= {
                "0": PolicySpec(
                    policy_class=None,  # infer automatically from Algorithm
                    observation_space=env.observation_space,  # infer automatically from env
                    action_space=env.action_space,  # infer automatically from env
                    config={"gamma": 0.85},  # use main config plus <- this override here
                ),
                "1": PolicySpec(
                    policy_class=None,  # infer automatically from Algorithm
                    observation_space=env.observation_space,  # infer automatically from env
                    action_space=env.action_space,  # infer automatically from env
                    config={"gamma": 0.85},  # use main config plus <- this override here
                ),
            },
            policy_mapping_fn = lambda agent_id, episode, worker: "0" if agent_id == "0" else "1",
        )
        .resources(num_gpus=0)
    )
    
    my_ma_algo = config.build()
    my_ma_algo.train()