import numpy as np
import math

from ray.rllib.env import MultiAgentEnv
from gymnasium import spaces
from gymnasium.utils import seeding
from ray.rllib.env.env_context import EnvContext

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from metrics import calculate_dos, calculate_doa


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

        self.use_vectorized = config.get('use_vectorized')
        self.dt = config.get('temporal_increment')

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
            agent.agent_id: spaces.Box(
                low=np.array([0, -self.max_turn]),
                high=np.array([
                    self.max_acceleration_prey if agent.agent_type == 0 else self.max_acceleration_predator,
                    self.max_turn]
                ),
                dtype=self.float_dtype,
                shape=(2,)
            ) for agent in self.agents
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
                low=-1, 
                high=1,
                dtype=self.float_dtype,
                shape=(self.observation_size,)
            ) for agent in self.agents
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

    def _get_observation_dict(self):
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
            if i < self.num_agents:
                agent.loc_x, agent.loc_y = loc_x[i], loc_y[i]
                agent.speed_x, agent.speed_y = 0.0, 0.0
                agent.heading = headings[i]
                agent.still_in_game = 1

        observation_list = self._get_observation_dict()
        return observation_list, {}

    def step(self, action_list):
        self.timestep += 1

        sub_steps_nb = int(round(1 / self.dt))
        all_eating_events = []
        for i in range(sub_steps_nb):
            action_list = action_list if i == 0 else None  # the agent use action once and then the physics do the rest
            if self.use_vectorized:
                self._simulate_one_vectorized_step(self.dt, action_list)
            else:
                eating_events = self._simulate_one_step(self.dt, action_list)
                # append the eating events to the list
                all_eating_events.extend(eating_events)

        reward_dict = self._get_reward(action_list, all_eating_events)
        observation_dict = self._get_observation_dict()
        terminated, truncated = self._get_done()

        # get array of locations for each agent
        if self.num_preys > 0:
            loc_x = [agent.loc_x for agent in self.agents if agent.still_in_game == 1 and agent.agent_type == 0]
            loc_y = [agent.loc_y for agent in self.agents if agent.still_in_game == 1 and agent.agent_type == 0]
            heading = [agent.heading for agent in self.agents if agent.still_in_game == 1 and agent.agent_type == 0]
            dos = calculate_dos(loc_x, loc_y) / (self.num_preys * self.grid_diagonal) 
            doa = calculate_doa(heading) / (self.num_preys * 2 * np.pi)
        else:
            dos = 0
            doa = 0
        infos = {"__common__": {"dos": dos, "doa": doa, "timestep": self.timestep}}

        return observation_dict, reward_dict, terminated, truncated, infos

    def render(self):
        raise NotImplementedError()

    def _simulate_one_step(self, dt, action_dict=None):

        # BUMP INTO OTHER AGENTS
        eating_events = []
        contact_force_dict = {agent.agent_id: np.array([0.0, 0.0]) for agent in self.agents}
        for a, agent_a in enumerate(self.agents):
            for b, agent_b in enumerate(self.agents):
                if b < a:  # Avoid double-checking and self-checking
                    continue
                if not agent_a.still_in_game or not agent_b.still_in_game:
                    continue

                delta_x = agent_a.loc_x - agent_b.loc_x
                delta_y = agent_a.loc_y - agent_b.loc_y
                dist = math.sqrt(delta_x ** 2 + delta_y ** 2)
                dist_min = agent_a.radius + agent_b.radius

                if dist < dist_min:  # There's a collision
                    if agent_a.agent_type != agent_b.agent_type:
                        prey_agent, predator_agent = (agent_a, agent_b) if agent_a.agent_type == 0 else (agent_b, agent_a)
                        eating_events.append({"predator_id": predator_agent.agent_id, "prey_id": prey_agent.agent_id})
                        if self.prey_consumed:
                            prey_agent.still_in_game = 0
                            self.num_preys -= 1
                            continue
                    if agent_a.agent_type != agent_b.agent_type and self.prey_consumed:
                        print("this should never happend, because of continue")
                    k = self.contact_margin  # This is defined in config
                    penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
                    force_magnitude = self.contact_force_coefficient * penetration  # This is defined in config

                    if dist == 0:  # To avoid division by zero
                        force_direction = np.random.rand(2)
                        force_direction /= np.linalg.norm(force_direction)  # Normalize
                    else:
                        force_direction = np.array([delta_x, delta_y]) / dist

                    force = force_magnitude * force_direction
                    contact_force_dict[agent_a.agent_id] += force
                    contact_force_dict[agent_b.agent_id] -= force  # Apply equal and opposite force

        for agent in self.agents:
            if agent.still_in_game == 1:
                # DRAGGING FORCE
                # Calculate the speed magnitude
                speed_magnitude = math.sqrt(agent.speed_x ** 2 + agent.speed_y ** 2)

                # Calculate the dragging force amplitude based on the chosen type of friction
                if self.friction_regime == "linear":
                    dragging_force_amplitude = speed_magnitude * self.dragging_force_coefficient
                elif self.friction_regime == "quadratic":
                    dragging_force_amplitude = speed_magnitude ** 2 * self.dragging_force_coefficient
                else:
                    dragging_force_amplitude = speed_magnitude ** 1.4 * self.dragging_force_coefficient

                # opposed to the speed direction of previous step
                dragging_force_orientation = math.atan2(agent.speed_y, agent.speed_x) - math.pi
                acceleration_x = dragging_force_amplitude * math.cos(dragging_force_orientation)
                acceleration_y = dragging_force_amplitude * math.sin(dragging_force_orientation)

                # ACCELERATION FORCE
                if action_dict is not None:
                    # get the actions for this agent
                    self_force_amplitude, self_force_orientation = action_dict.get(agent.agent_id)
                    agent.heading = (agent.heading + self_force_orientation) % (2 * np.pi)
                    acceleration_x += self_force_amplitude * math.cos(agent.heading)
                    acceleration_y += self_force_amplitude * math.sin(agent.heading)

                # CONTACT FORCE
                contact_force = contact_force_dict.get(agent.agent_id)
                acceleration_x += contact_force[0]
                acceleration_y += contact_force[1]

                # WALL BOUNCING
                # Check if the agent is touching the edge
                if self.periodical_boundary is False and self.wall_contact_force_coefficient > 0:
                    if agent.loc_x < agent.radius or agent.loc_x > self.stage_size - agent.radius:
                        # you can rarely have contact with two walls at the same time
                        contact_force_amplitude_x = self.wall_contact_force_coefficient * (
                            agent.radius - agent.loc_x if agent.loc_x < agent.radius
                            else agent.loc_x - self.stage_size + agent.radius
                        )
                        acceleration_x += sign(self.stage_size / 2 - agent.loc_x) * contact_force_amplitude_x

                    if agent.loc_y < agent.radius or agent.loc_y > self.stage_size - agent.radius:
                        contact_force_amplitude_y = self.wall_contact_force_coefficient * (
                            agent.radius - agent.loc_y if agent.loc_y < agent.radius
                            else agent.loc_y - self.stage_size + agent.radius
                        )
                        acceleration_y += sign(self.stage_size / 2 - agent.loc_y) * contact_force_amplitude_y

                # # UPDATE ACCELERATION/SPEED/POSITION
                # Update speed using acceleration
                agent.speed_x += acceleration_x * dt / (agent.radius ** 3 * self.agent_density)
                agent.speed_y += acceleration_y * dt / (agent.radius ** 3 * self.agent_density)

                # limit the speed
                max_speed = self.max_speed_prey if agent.agent_type == 0 else self.max_speed_predator
                current_speed = math.sqrt(agent.speed_x ** 2 + agent.speed_y ** 2)
                if current_speed > max_speed:
                    agent.speed_x *= max_speed / current_speed
                    agent.speed_y *= max_speed / current_speed

                # Note : agent.heading was updated right after getting the action list
                # Update the agent's location
                agent.loc_x += agent.speed_x * dt
                agent.loc_y += agent.speed_y * dt

                # limit the location to the stage size and set speed to 0 in the direction
                if agent.loc_x < agent.radius / 2:
                    agent.loc_x = agent.radius
                    agent.speed_x = 0
                elif agent.loc_x > self.stage_size - agent.radius / 2:
                    agent.loc_x = self.stage_size - agent.radius
                    agent.speed_x = 0
                if agent.loc_y < agent.radius / 2:
                    agent.loc_y = agent.radius
                    agent.speed_y = 0
                elif agent.loc_y > self.stage_size - agent.radius / 2:
                    agent.loc_y = self.stage_size - agent.radius
                    agent.speed_y = 0

                # periodic boundary
                if self.periodical_boundary:
                    agent.loc_x = agent.loc_x % self.stage_size
                    agent.loc_y = agent.loc_y % self.stage_size

        return eating_events

    def _simulate_one_vectorized_step(self, dt, action_dict=None):
        # Not worth it for small number of agents !
        raise NotImplementedError("See archive.py")

    def _get_reward(self, action_list, all_eating_events):
        # Initialize rewards
        reward_list = {agent.agent_id: 0 for agent in self.agents}

        predator_agents = [other_agent for other_agent in self.agents if other_agent.agent_type == 1]

        for event in all_eating_events:
            predator_id, prey_id = event["predator_id"], event["prey_id"]
            # Apply the eating reward for the predator and the death penalty for the prey
            reward_list[predator_id] += self.eating_reward_for_predator
            reward_list[prey_id] += self.death_penalty_for_prey
            # collective penalty for preys
            for agent in self.agents:
                if agent.still_in_game == 1:
                    if agent.agent_type == 0:  # Prey
                        reward_list[agent.agent_id] += self.collective_death_penalty_for_prey
                    elif agent.agent_type == 1:  # Predator
                        reward_list[agent.agent_id] += self.collective_eating_reward_for_predator

        for agent in self.agents:
            if agent.still_in_game:
                # AGENT EATEN OR NOT
                if agent.agent_type == 0:  # 0 for prey, is_prey
                    reward_list[agent.agent_id] += self.surviving_reward_for_prey
                else:  # is_predator
                    reward_list[agent.agent_id] += self.starving_penalty_for_predator

                # ENERGY EFFICIENCY
                # Add the energy efficiency penalty
                # set the energy cost penalty
                self_force_amplitude, self_force_orientation = action_list.get(agent.agent_id)

                energy_cost_penalty = -(abs(self_force_amplitude) + abs(self_force_orientation)) * self.energy_cost_penalty_coef
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
        # is True where the prey is eaten (agent.still_in_game == 0)
        # or when episode ends because all preys have been eaten (self.num_preys == 0)
        terminated = {agent.agent_id: self.num_preys == 0 or agent.still_in_game == 0 for agent in self.agents}
        terminated['__all__'] = self.num_preys == 0
        # Premature ending (because of time limit)
        truncated = {agent.agent_id: self.timestep >= self.episode_length for agent in self.agents}
        truncated['__all__'] = self.timestep >= self.episode_length

        return terminated, truncated


class MetricsCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        # Initialize sum of DoS for the episode
        episode.user_data['dos'] = []
        episode.user_data['doa'] = []

    def on_episode_step(self, worker, base_env, policies, episode, **kwargs):
        # Assuming you can extract loc_x and loc_y from the episode
        info = episode.last_info_for("__common__")
        episode.user_data["dos"].append(info["dos"])
        episode.user_data["doa"].append(info["doa"])

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        # Average DoS at the end of episode
        info = episode.last_info_for("__common__")
        average_dos = sum(episode.user_data['dos']) / info["timestep"]
        average_doa = sum(episode.user_data['doa']) / info["timestep"]
        episode.custom_metrics['dos'] = average_dos
        episode.custom_metrics['doa'] = average_doa

