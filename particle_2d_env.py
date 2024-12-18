import numpy as np

from ray.rllib.env import MultiAgentEnv
from gymnasium import spaces, vector
from gymnasium.utils import seeding
from ray.rllib.env.env_context import EnvContext

from metrics import calculate_dos, calculate_doa
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import wandb
import pygame


def sign(x):
    if x == 0:
        return 0
    elif x > 0:
        return 1
    else:
        return -1


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
                 still_in_game=1):
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
        self.step_per_time_increment = config.get('step_per_time_increment')
        assert isinstance(self.step_per_time_increment, int)
        self.dt = 1 / self.step_per_time_increment

        # ENVIRONMENT
        self.timestep = 0
        self.episode_length = config.get('episode_length')
        assert self.episode_length > 0
        # stage size in an interval or a fixed value
        stage_size = config.get('stage_size')
        self.stage_size, _ = random_int_in_interval(stage_size)
        assert self.stage_size > 1
        self.grid_diagonal = self.stage_size * np.sqrt(2, dtype=self.float_dtype)

        # PHYSICS
        self.inertia = config.get('inertia')
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

        self.prey_consumed = config.get('prey_consumed')

        # AGENTS (PREYS AND PREDATORS)
        # random number of preys
        self.ini_num_preys, self.max_num_preys = random_int_in_interval(config.get('num_preys'))
        self.num_preys = self.ini_num_preys
        # random number of predators
        self.num_predators, self.max_num_predators = random_int_in_interval(config.get('num_predators'))
        assert self.num_preys > 0
        assert self.num_predators >= 0

        self.prey_radius = config.get('prey_radius')
        self.predator_radius = config.get('predator_radius')
        assert 0 <= self.prey_radius <= 1
        assert 0 <= self.predator_radius <= 1

        self.agent_density = config.get('agent_density')
        assert 0 < self.agent_density

        self.particule_agents = []
        for i in range(self.max_num_agents):
            if i < self.num_agents:
                if i < self.num_preys:
                    agent_type, radius = 0, self.prey_radius  # for preys
                else:
                    agent_type, radius = 1, self.predator_radius  # for predators
                still_in_game = 1
            else:
                agent_type, radius, still_in_game = None, None, 0
            self.particule_agents.append(ParticuleAgent(id=i, agent_type=agent_type, radius=radius, still_in_game=still_in_game))


        self.agents = [agent.agent_id for agent in self.particule_agents if agent.still_in_game == 1]
        self.possible_agents = [agent.agent_id for agent in self.particule_agents]
        self._agent_ids = {agent.agent_id for agent in self.particule_agents}  # Used by RLlib

        # ACTIONS (ACCELERATION AND TURN)
        self.max_acceleration_prey = config.get('max_acceleration_prey')
        self.max_acceleration_predator = config.get('max_acceleration_predator')
        self.max_turn = config.get('max_turn')

        self.action_space = spaces.Dict({
            agent.agent_id: spaces.Box(
                low=np.array([0, -1]),  # this gets multiplied later
                high=np.array([1, -1]),  # this gets multiplied later
                dtype=self.float_dtype,
                shape=(2,)
            ) for agent in self.particule_agents
        })

        # OBSERVATION SETTINGS
        self.max_seeing_angle = config.get('max_seeing_angle')
        if not 0 < self.max_seeing_angle <= self.float_dtype(np.pi):
            self.max_seeing_angle = self.float_dtype(np.pi)

        self.max_seeing_distance = config.get('max_seeing_distance')
        if not 0 < self.max_seeing_distance <= self.grid_diagonal:
            self.max_seeing_distance = self.grid_diagonal

        self.num_other_agents_observed = config.get('num_other_agents_observed')
        if not 0 < self.num_other_agents_observed <= self.num_agents - 1 or self.num_other_agents_observed == "all":
            self.num_other_agents_observed = self.max_num_agents - 1

        self.use_polar_coordinate = config.get('use_polar_coordinate')
        self.use_speed_observation = config.get('use_speed_observation')

        # Number of observed properties
        self.self_observed_properties = 7  # 4 position, 1 heading, 2 speed
        self.num_observed_properties = 4  # relative heading and positions + type are always observed
        if self.use_speed_observation:  # speed is observed when use_speed_observation is True
            self.num_observed_properties += 2
        self.observation_size = self.self_observed_properties + self.num_observed_properties * self.num_other_agents_observed

        self.observation_space = spaces.Dict({
            agent.agent_id: spaces.Box(
                low=-np.inf,
                high=np.inf,
                dtype=self.float_dtype,
                shape=(self.observation_size,)
            ) for agent in self.particule_agents
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


    @property
    def num_agents(self):
        return self.num_preys + self.num_predators

    @property
    def max_num_agents(self):
        return self.max_num_preys + self.max_num_predators

    def compute_distance(self, agent1, agent2):
        if self.periodical_boundary:
            # Calculate the distance considering wrapping at the boundaries
            delta_x = abs(agent1.loc_x - agent2.loc_x)
            delta_y = abs(agent1.loc_y - agent2.loc_y)

            # Consider wrapping effect: if distance is greater than half the stage,
            # it's shorter to go the other way around
            delta_x = min(delta_x, self.stage_size - delta_x)
            delta_y = min(delta_y, self.stage_size - delta_y)
        else:
            # Standard Euclidean distance
            delta_x = agent1.loc_x - agent2.loc_x
            delta_y = agent1.loc_y - agent2.loc_y

        return np.sqrt(delta_x ** 2 + delta_y ** 2, dtype=self.float_dtype)

    def compute_angle(self, agent1, agent2):
        if self.periodical_boundary:
            # Compute the shortest path deltas considering wrapping at the boundaries
            delta_x = agent2.loc_x - agent1.loc_x
            delta_y = agent2.loc_y - agent1.loc_y

            # Adjust deltas for periodic boundary conditions
            if abs(delta_x) > self.stage_size / 2:
                if delta_x > 0:
                    delta_x -= self.stage_size
                else:
                    delta_x += self.stage_size

            if abs(delta_y) > self.stage_size / 2:
                if delta_y > 0:
                    delta_y -= self.stage_size
                else:
                    delta_y += self.stage_size
        else:
            # Compute the direct deltas without considering periodic boundaries
            delta_x = agent2.loc_x - agent1.loc_x
            delta_y = agent2.loc_y - agent1.loc_y

        # Compute the direction to the other agent
        direction = np.arctan2(delta_y, delta_x).astype(self.float_dtype) - agent1.heading

        # Normalize the direction to the range [-pi, pi]
        direction = (direction + self.float_dtype(np.pi)) % (2 * self.float_dtype(np.pi)) - self.float_dtype(np.pi)
        return direction

    def _observation_pos(self, agent, other_agent):
        if not self.use_polar_coordinate:
            return (other_agent.loc_x - agent.loc_x), (
                    other_agent.loc_y - agent.loc_y)
        else:
            dist = self.compute_distance(agent, other_agent)
            direction = self.compute_angle(agent, other_agent)
            return dist, direction

    def _generate_observation(self, agent):
        # initialize obs as an empty list of correct size
        obs = np.zeros(self.observation_size, dtype=self.float_dtype)

        # SELF OBSERVATION
        # Distance to walls
        obs[0] = agent.loc_x
        obs[1] = agent.loc_y
        obs[2] = self.stage_size - agent.loc_x
        obs[3] = self.stage_size - agent.loc_y
        # modulo 2pi to avoid large values
        obs[4] = agent.heading % (2 * self.float_dtype(np.pi))
        # speed
        if not self.use_polar_coordinate:
            obs[5] = agent.speed_x
            obs[6] = agent.speed_y
        else:
            obs[5] = np.sqrt(agent.speed_x ** 2 + agent.speed_y ** 2, dtype=self.float_dtype)
            obs[6] = np.arctan2(agent.speed_y, agent.speed_x).astype(self.float_dtype) % (2 * self.float_dtype(np.pi))

        # OTHER AGENTS
        # Remove the agent itself and the agents that are not in the game
        other_agents = [
            other_agent for other_agent in self.particule_agents
            if other_agent is not agent and other_agent.still_in_game == 1
               and abs(self.compute_angle(agent, other_agent)) < self.max_seeing_angle
               and self.compute_distance(agent, other_agent) < self.max_seeing_distance
        ]

        other_agents = sorted(other_agents, key=lambda other_agent: self.compute_distance(agent, other_agent))
        # keep only the closest agents
        other_agents = other_agents[:self.num_other_agents_observed]

        for j, other in enumerate(other_agents):
            # count the number of already observed properties
            base_index = 5 + j * self.num_observed_properties
            obs[base_index], obs[base_index + 1] = self._observation_pos(agent, other)  # relative position
            obs[base_index + 2] = ((other.heading - agent.heading) % (2 * self.float_dtype(np.pi)) - self.float_dtype(np.pi))  # relative heading
            if self.use_speed_observation:
                # add speed normalized by max speed for prey or predator

                if not self.use_polar_coordinate:
                    obs[base_index + 3] = (other.speed_x - agent.speed_x)
                    obs[base_index + 4] = (other.speed_y - agent.speed_y)
                else:
                    obs[base_index + 3] = np.sqrt(
                        (other.speed_x - agent.speed_x) ** 2 + (other.speed_y - agent.speed_y) ** 2,
                        dtype=self.float_dtype
                    )
                    obs[base_index + 4] = np.arctan2(other.speed_y - agent.speed_y,
                                                     other.speed_x - agent.speed_x
                                                     ).astype(self.float_dtype) - agent.heading

            obs[base_index + (self.num_observed_properties - 1)] = other.agent_type

        return obs

    def _get_observation_dict(self):
        return {agent.agent_id: self._generate_observation(agent) for agent in self.particule_agents if agent.still_in_game == 1}

    def reset(self, seed=None, options=None):
        # Reset time to the beginning
        self.timestep = 0
        self.num_preys = self.ini_num_preys

        # Vectorized operations for random values
        # len(self.particule_agents) can be bigger than self.num_agents
        random_values = self.np_random.random(size=(len(self.particule_agents), 3))
        loc_x = random_values[:, 0] * self.stage_size
        loc_y = random_values[:, 1] * self.stage_size
        headings = random_values[:, 2] * 2 * self.float_dtype(np.pi)

        # Assigning the vectorized values to agents
        for i, agent in enumerate(self.particule_agents):
            if i < self.num_agents:
                agent.loc_x, agent.loc_y = loc_x[i], loc_y[i]
                agent.speed_x, agent.speed_y = 0.0, 0.0
                agent.heading = headings[i]
                agent.still_in_game = 1

        observation_list = self._get_observation_dict()
        return observation_list, {}

    def step(self, action_list):
        self.timestep += 1

        all_eating_events = []
        for i in range(self.step_per_time_increment):
            action_list = action_list if i == 0 else None  # the agent use action once and then the physics do the rest
            if self.use_vectorized:
                eating_events = self._simulate_one_vectorized_step(self.dt, action_list)
            else:
                eating_events = self._simulate_one_step(self.dt, action_list)
                # append the eating events to the list
            all_eating_events.extend(eating_events)

        reward_dict = self._get_reward(action_list, all_eating_events)
        observation_dict = self._get_observation_dict()
        terminated, truncated = self._get_done()

        # get array of locations for each agent
        if self.num_preys > 0:
            loc_x = [agent.loc_x for agent in self.particule_agents if agent.still_in_game == 1 and agent.agent_type == 0]
            loc_y = [agent.loc_y for agent in self.particule_agents if agent.still_in_game == 1 and agent.agent_type == 0]
            heading = [agent.heading for agent in self.particule_agents if agent.still_in_game == 1 and agent.agent_type == 0]
            dos = calculate_dos(loc_x, loc_y) / (self.num_preys * self.grid_diagonal)
            doa = calculate_doa(heading) / (self.num_preys * 2 * self.float_dtype(np.pi))
        else:
            dos = 0
            doa = 0

        infos = {"__common__": {"dos": dos, "doa": doa, "timestep": self.timestep}}

        return observation_dict, reward_dict, terminated, truncated, infos

    def _simulate_one_step(self, dt, action_dict=None):

        # BUMP INTO OTHER AGENTS
        eating_events = []
        contact_force_dict = {agent.agent_id: np.array([0.0, 0.0]) for agent in self.particule_agents}
        for agent_a in self.particule_agents:
            for agent_b in self.particule_agents:
                if agent_a.still_in_game == 0 or agent_b.still_in_game == 0:
                    continue
                if agent_a.agent_id < agent_b.agent_id:  # Avoid double-checking and self-checking
                    continue

                delta_x = agent_a.loc_x - agent_b.loc_x
                delta_y = agent_a.loc_y - agent_b.loc_y
                dist = np.sqrt(delta_x ** 2 + delta_y ** 2, dtype=self.float_dtype)
                dist_min = agent_a.radius + agent_b.radius

                if dist < dist_min:  # There's a collision
                    if agent_a.agent_type != agent_b.agent_type:
                        prey_agent, predator_agent = (agent_a, agent_b) if agent_a.agent_type == 0 else (
                            agent_b, agent_a)
                        eating_events.append({"predator_id": predator_agent.agent_id, "prey_id": prey_agent.agent_id})

                        # No bouncing if the prey is already consumed therefore we skip the contact force
                        if self.prey_consumed:
                            continue

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

        for agent in self.particule_agents:
            if agent.still_in_game == 1:
                # DRAGGING FORCE
                # Calculate the speed magnitude
                speed_magnitude = np.sqrt(agent.speed_x ** 2 + agent.speed_y ** 2, dtype=self.float_dtype)

                # Calculate the dragging force amplitude based on the chosen type of friction
                if self.friction_regime == "linear":
                    dragging_force_amplitude = speed_magnitude * self.dragging_force_coefficient
                elif self.friction_regime == "quadratic":
                    dragging_force_amplitude = speed_magnitude ** 2 * self.dragging_force_coefficient
                else:
                    dragging_force_amplitude = speed_magnitude ** 1.4 * self.dragging_force_coefficient

                # opposed to the speed direction of previous step
                dragging_force_orientation = np.arctan2(agent.speed_y, agent.speed_x).astype(self.float_dtype) - self.float_dtype(np.pi)
                acceleration_x = dragging_force_amplitude * np.cos(dragging_force_orientation)
                acceleration_y = dragging_force_amplitude * np.sin(dragging_force_orientation)

                # ACCELERATION FORCE
                if action_dict is not None:
                    # get the actions for this agent
                    self_force_amplitude, self_force_orientation = action_dict.get(agent.agent_id)
                    agent.heading = (agent.heading + self_force_orientation * self.max_turn) % (2 * self.float_dtype(np.pi))
                    self_force_amplitude *= self.max_acceleration_prey if agent.agent_type == 0 else self.max_acceleration_predator
                    acceleration_x += self_force_amplitude * np.cos(agent.heading)
                    acceleration_y += self_force_amplitude * np.sin(agent.heading)

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

                if self.inertia:
                    acceleration_x /= (agent.radius ** 3 * self.agent_density)
                    acceleration_y /= (agent.radius ** 3 * self.agent_density)

                # # UPDATE ACCELERATION/SPEED/POSITION
                # Update speed using acceleration
                agent.speed_x += acceleration_x * dt
                agent.speed_y += acceleration_y * dt

                # Apply the speed limit
                max_speed = self.max_speed_prey if agent.agent_type == 0 else self.max_speed_predator
                current_speed = np.sqrt(agent.speed_x ** 2 + agent.speed_y ** 2, dtype=self.float_dtype)
                if max_speed is not None and current_speed > max_speed:
                    agent.speed_x *= max_speed / current_speed
                    agent.speed_y *= max_speed / current_speed

                # Note : agent.heading was updated right after getting the action list
                # Update the agent's location
                agent.loc_x += agent.speed_x * dt
                agent.loc_y += agent.speed_y * dt

                # periodic boundary
                if self.periodical_boundary:
                    agent.loc_x = agent.loc_x % self.stage_size
                    agent.loc_y = agent.loc_y % self.stage_size
                else:
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

        return eating_events

    def _simulate_one_vectorized_step(self, dt, action_dict=None):
        # Not worth it for small number of agents !
        raise NotImplementedError("See archive.py")

    def _get_reward(self, action_list, all_eating_events):
        # Initialize rewards
        reward_dict = {agent.agent_id: 0 for agent in self.particule_agents if agent.still_in_game == 1}

        for event in all_eating_events:
            predator_id, prey_id = event["predator_id"], event["prey_id"]
            # Apply the eating reward for the predator and the death penalty for the prey
            reward_dict[predator_id] += self.eating_reward_for_predator
            reward_dict[prey_id] += self.death_penalty_for_prey
            self.num_preys -= 1
            # find the prey in particule_agents and set still_in_game to 0
            for agent in self.particule_agents:
                if agent.agent_id == prey_id:
                    agent.still_in_game = 0

            # collective penalty for preys
            for agent in self.particule_agents:
                if agent.agent_type == 0 and agent.still_in_game == 1:  # Prey
                    reward_dict[agent.agent_id] += self.collective_death_penalty_for_prey
                elif agent.agent_type == 1:  # Predator
                    reward_dict[agent.agent_id] += self.collective_eating_reward_for_predator

        for agent in self.particule_agents:
            if agent.still_in_game:
                if agent.agent_type == 0:  # 0 for prey, is_prey
                    reward_dict[agent.agent_id] += self.surviving_reward_for_prey
                else:  # is_predator
                    reward_dict[agent.agent_id] += self.starving_penalty_for_predator

                # ENERGY EFFICIENCY
                # Add the energy efficiency penalty
                # set the energy cost penalty
                if action_list is not None:
                    self_force_amplitude, self_force_orientation = action_list.get(agent.agent_id)

                    energy_cost_penalty = -(
                            abs(self_force_amplitude) + abs(self_force_orientation)
                    ) * self.energy_cost_penalty_coef
                    reward_dict[agent.agent_id] += energy_cost_penalty

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
                        reward_dict[agent.agent_id] += self.edge_hit_penalty

        return reward_dict

    def _get_done(self):
        # Natural ending
        # is True where the prey is eaten (agent.still_in_game == 0)
        # or when episode ends because all preys have been eaten (self.num_preys == 0)
        terminated = {
            agent.agent_id: self.num_preys == 0 or self.timestep >= self.episode_length or agent.still_in_game == 0 for
            agent in self.particule_agents}
        terminated['__all__'] = self.num_preys == 0 or self.timestep >= self.episode_length
        # Premature ending (because of time limit)
        truncated = {agent.agent_id: self.timestep >= self.episode_length for agent in self.particule_agents}
        truncated['__all__'] = self.timestep >= self.episode_length

        return terminated, truncated

    def render(self, render_mode="rgb_array"):
        if render_mode == "rgb_array":
            predator_color = pygame.Color("#C843C3")
            prey_color = pygame.Color("#245EB6")
            fig_size = 512

            canvas = pygame.Surface((fig_size, fig_size))
            canvas.fill((255, 255, 255))

            # Calculating the pixel square size
            pix_square_size = fig_size / self.stage_size

            # Drawing the agents
            for agent in self.particule_agents:
                if agent.still_in_game == 1:
                    agent_pos = (agent.loc_x + 0.5) * pix_square_size, (agent.loc_y + 0.5) * pix_square_size
                    agent_size = agent.radius * pix_square_size
                    if agent.agent_type == 0:
                        pygame.draw.circle(canvas, prey_color, agent_pos, agent_size)
                    else:
                        pygame.draw.circle(canvas, predator_color, agent_pos, agent_size)

            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

class MetricsCallbacks(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_episode_start(self, *, episode, **kwargs):
        # Initialize sum of DoS for the episode
        episode.user_data['dos'] = []
        episode.user_data['doa'] = []

    #  episode, env_runner, metrics_logger, env, env_index, rl_module, worker,base_env, policies,
    # worker,base_env, policies,
    def on_episode_step(self, *, episode, **kwargs):
        # Assuming you can extract loc_x and loc_y from the episode
        info = episode.last_info_for("__common__")
        episode.user_data["dos"].append(info["dos"])
        episode.user_data["doa"].append(info["doa"])

    def on_episode_end(self, *, episode, **kwargs):
        # Average DoS at the end of episode
        info = episode.last_info_for("__common__")
        average_dos = sum(episode.user_data['dos']) / info["timestep"]
        average_doa = sum(episode.user_data['doa']) / info["timestep"]
        episode.custom_metrics['dos'] = average_dos
        episode.custom_metrics['doa'] = average_doa

class RenderingCallbacks(DefaultCallbacks):
    # Based on example from https://github.com/ray-project/ray/blob/master/rllib/examples/envs/env_rendering_and_recording.py
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_episode_and_return = (None, float("-inf"))

    def on_episode_step(self, *, episode, env, **kwargs):
        if isinstance(env.unwrapped, vector.VectorEnv):
            frame = env.envs[0].render()
        else:
            frame = env.render()
        episode.add_temporary_timestep_data("render_images", frame)

    def on_episode_end(self, *, episode, **kwargs):
        episode_return = episode.get_return()
        # Better than the best Episode thus far?
        if episode_return > self.best_episode_and_return[1]:
            # Pull all images from the temp. data of the episode.
            images = episode.get_temporary_timestep_data("render_images")
            # `images` is now a list of 3D ndarrays

            video = np.transpose(np.array(images), (0, 3, 1, 2))
            # save the 4D numpy array as a gif
            # imageio.mimsave("video.gif", np.array(images), fps=10)
            if episode_return > self.best_episode_and_return[1]:
                self.best_episode_and_return = (wandb.Video(video, fps=30, format="gif"), episode_return)

    def on_sample_end(self, *, metrics_logger, **kwargs) -> None:
        """Logs the best cideo to this EnvRunner's MetricsLogger."""
        # Best video.
        if self.best_episode_and_return[0] is not None:
            metrics_logger.log_value(
                "episode_videos_best",
                self.best_episode_and_return[0],
                # Do not reduce the videos (across the various parallel EnvRunners).
                # This would not make sense (mean over the pixels?). Instead, we want to
                # log all best videos of all EnvRunners per iteration.
                reduce=None,
                # B/c we do NOT reduce over the video data (mean/min/max), we need to
                # make sure the list of videos in our MetricsLogger does not grow
                # infinitely and gets cleared after each `reduce()` operation, meaning
                # every time, the EnvRunner is asked to send its logged metrics.
                clear_on_reduce=True,
            )
            self.best_episode_and_return = (None, float("-inf"))

