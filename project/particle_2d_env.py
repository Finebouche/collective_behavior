import numpy as np

from ray.rllib.env import MultiAgentEnv
from gymnasium import spaces, vector
from gymnasium.utils import seeding
from ray.rllib.env.env_context import EnvContext

from metrics import calculate_dos, calculate_doa
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.callbacks.callbacks import RLlibCallback

import wandb
import cv2

def color_from_hex(hex_str):
    """
    Convenience: Convert a color hex string like '#C843C3' to an RGB NumPy array.
    """
    hex_str = hex_str.lstrip('#')
    return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=np.uint8)

def sign(x):
    if x == 0:
        return 0
    elif x > 0:
        return 1
    else:
        return -1

def random_int_in_interval(variable):
    """
    function that check if variable is an array, then choose a random int in the interval
    return the number and the max_value
    """
    if isinstance(variable, list):
        return np.random.randint(variable[0], variable[1]), variable[1]
    else:
        return variable, variable

class BaseEntity:
    """
    Minimal base class holding positional information, a radius, and a type.
    Both agents and food will inherit from this class.
    """
    def __init__(self, entity_id=None, radius=None, entity_type=None, loc_x=None, loc_y=None):
        self.entity_id = entity_id
        self.entity_type = entity_type    # e.g. 0 = prey, 1 = predator, 2 = food
        self.radius = radius
        self.loc_x = loc_x
        self.loc_y = loc_y

class ParticuleAgent(BaseEntity):
    """
    Agents (prey or predator). Inherits from BaseEntity, adding
    speeds, heading, and a flag whether still in game.
    """
    def __init__(self, id=None, radius=None, agent_type=None,
                 loc_x=None, loc_y=None, heading=None,
                 speed_x=None, speed_y=None, still_in_game=1):
        super().__init__(entity_id=id, radius=radius, entity_type=agent_type, loc_x=loc_x, loc_y=loc_y)
        self.heading = heading
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.still_in_game = still_in_game

class Food(BaseEntity):
    """
    A Food entity. Also inherits from BaseEntity.
    A Food item can be inactive when eaten;
    it re-spawns based on the patch logic.
    """
    def __init__(self, entity_id=None, patch_id=None, radius=None, food_type=2, loc_x=None, loc_y=None, active=True):
        super().__init__(entity_id=entity_id, radius=radius, entity_type=2, loc_x=loc_x, loc_y=loc_y)
        self.patch_id = patch_id
        self.active = active

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

        #######################################
        # ENVIRONMENT
        #######################################
        self.timestep = 0
        self.episode_length = config.get('episode_length')
        assert self.episode_length > 0
        # stage size in an interval or a fixed value
        stage_size = config.get('stage_size')
        self.stage_size, _ = random_int_in_interval(stage_size)
        assert self.stage_size > 1
        self.grid_diagonal = self.stage_size * np.sqrt(2, dtype=self.float_dtype)

        #######################################
        # PHYSICS
        #######################################
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

        #######################################
        # AGENTS (PREYS AND PREDATORS)
        #######################################
        # random number of preys
        self.ini_num_preys, self.max_num_preys = random_int_in_interval(config.get('num_preys'))
        self.num_preys = self.ini_num_preys
        # random number of predators
        self.num_predators, self.max_num_predators = random_int_in_interval(config.get('num_predators'))
        assert self.num_preys > 0
        assert self.num_predators >= 0

        self.prey_radius = config.get('prey_radius')
        self.predator_radius = config.get('predator_radius')
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

        self.agents = [agent.entity_id for agent in self.particule_agents if agent.still_in_game == 1]
        self.possible_agents = [agent.entity_id for agent in self.particule_agents]
        self._agent_ids = {agent.entity_id for agent in self.particule_agents}  # Used by RLlib

        #######################################
        # FOOD / FOOD PATCHES
        #######################################
        self.num_food_patch = config.get('num_food_patch', 0)
        if self.num_food_patch > 0:
            self.food_patch_radius = config.get('food_patch_radius')  # the patch area
            self.food_radius = config.get('food_radius')              # each food piece radius
            self.food_patch_regen_time = config.get('food_patch_regen_time')
            self.max_number_of_food = config.get('max_number_of_food')
            self.food_reward = config.get('food_reward')

        # initialize the food patches
        self.food_patches = []
        # The active food items in the env:
        self.foods = []


        #######################################
        # ACTIONS (ACCELERATION AND TURN)
        #######################################
        self.max_acceleration_prey = config.get('max_acceleration_prey')
        self.max_acceleration_predator = config.get('max_acceleration_predator')
        self.max_turn = config.get('max_turn')

        self.action_space = spaces.Dict({
            agent.entity_id: spaces.Box(
                low=np.array([0, -1], dtype=self.float_dtype),  # this gets multiplied later
                high=np.array([1, 1], dtype=self.float_dtype),  # this gets multiplied later
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
        # - Self observation has 7 slots: 4 (distance to walls) + 1 heading + 2 speed
        self.self_observed_properties = 7
        # - Each "other agent" observed has (relative pos) + relative heading + optional speed + type
        self.num_observed_properties = 4 if not self.use_speed_observation else 6

        # For nearest food, we'll add 3 more dimensions if there's any food patch:
        #   either (dx, dy, type) or (dist, angle, type)
        self.food_observation_size = 3 if self.num_food_patch > 0 else 0

        self.observation_size = (
            self.self_observed_properties
            + self.num_observed_properties * self.num_other_agents_observed
            + self.food_observation_size
        )

        self.observation_space = spaces.Dict({
            agent.entity_id: spaces.Box(
                low=np.full(self.observation_size, -np.inf, dtype=self.float_dtype),
                high=np.full(self.observation_size,  np.inf, dtype=self.float_dtype),
                dtype=self.float_dtype,
                shape=(self.observation_size,)
            ) for agent in self.particule_agents
        })

        #######################################
        # REWARDS
        #######################################
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

    def compute_distance(self, agent, other_entity):
        # agent is a ParticuleAgent; other_entity could be another agent or food.
        if self.periodical_boundary:
            # Calculate the distance considering wrapping at the boundaries
            delta_x = abs(agent.loc_x - other_entity.loc_x)
            delta_y = abs(agent.loc_y - other_entity.loc_y)

            delta_x = min(delta_x, self.stage_size - delta_x)
            delta_y = min(delta_y, self.stage_size - delta_y)
        else:
            delta_x = agent.loc_x - other_entity.loc_x
            delta_y = agent.loc_y - other_entity.loc_y

        return np.sqrt(delta_x ** 2 + delta_y ** 2, dtype=self.float_dtype)

    def compute_angle(self, agent, other_entity):
        # agent is a ParticuleAgent; other_entity could be another agent or food.
        if self.periodical_boundary:
            delta_x = other_entity.loc_x - agent.loc_x
            delta_y = other_entity.loc_y - agent.loc_y
            if abs(delta_x) > self.stage_size / 2:
                delta_x -= self.stage_size * sign(delta_x)
            if abs(delta_y) > self.stage_size / 2:
                delta_y -= self.stage_size * sign(delta_y)
        else:
            delta_x = other_entity.loc_x - agent.loc_x
            delta_y = other_entity.loc_y - agent.loc_y

        direction = np.arctan2(delta_y, delta_x).astype(self.float_dtype) - agent.heading
        # Normalize to [-pi, pi]
        direction = (direction + self.float_dtype(np.pi)) % (2 * self.float_dtype(np.pi)) - self.float_dtype(np.pi)
        return direction

    def _observation_pos(self, agent, other_entity):
        """
        Return either Cartesian or polar coordinates from agent to `other_entity`.
        """
        if not self.use_polar_coordinate:
            return (other_entity.loc_x - agent.loc_x), (
                    other_entity.loc_y - agent.loc_y)
        else:
            dist = self.compute_distance(agent, other_entity)
            direction = self.compute_angle(agent, other_entity)
            return dist, direction

    def _generate_observation(self, agent):
        # initialize obs as an empty list of correct size
        obs = np.zeros(self.observation_size, dtype=self.float_dtype)

        # 1) SELF OBSERVATION (7 slots)
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

        # 2) OTHER AGENTS OBSERVATION
        # Remove the agent itself and the agents that are not in the game
        other_agents = [
            other_agent for other_agent in self.particule_agents
            if other_agent is not agent and other_agent.still_in_game == 1
               and abs(self.compute_angle(agent, other_agent)) < self.max_seeing_angle
               and self.compute_distance(agent, other_agent) < self.max_seeing_distance
        ]
        # Sort by distance
        other_agents = sorted(other_agents, key=lambda other_agent: self.compute_distance(agent, other_agent))
        # keep only the closest agents
        other_agents = other_agents[:self.num_other_agents_observed]

        for j, other in enumerate(other_agents):
            # count the number of already observed properties
            base_index = self.self_observed_properties + j * self.num_observed_properties
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

            obs[base_index + (self.num_observed_properties - 1)] = other.entity_type

        # 3) NEAREST FOOD (if any) -> last 3 slots
        if self.num_food_patch > 0:
            base_index = self.self_observed_properties + self.num_other_agents_observed * self.num_observed_properties
            active_foods = [f for f in self.foods if f.active]
            if len(active_foods) > 0:
                nearest_food = min(active_foods, key=lambda f: self.compute_distance(agent, f))
                fx_or_dist, fy_or_angle = self._observation_pos(agent, nearest_food)
                obs[base_index] = fx_or_dist
                obs[base_index + 1] = fy_or_angle
                obs[base_index + 2] = nearest_food.entity_type  # e.g. 2 for food
            else:
                # No active food: just fill with 0
                pass

        return obs.astype(self.float_dtype, copy=False)

    def _get_observation_dict(self):
        return {agent.entity_id: self._generate_observation(agent) for agent in self.particule_agents if agent.still_in_game == 1}

    def generate_food(self, patch_cx, patch_cy, patch_id):
        r = self.np_random.uniform(0, self.food_patch_radius)
        theta = self.np_random.uniform(0, 2 * np.pi)
        fx = patch_cx + r * np.cos(theta)
        fy = patch_cy + r * np.sin(theta)
        # make sure it is inside the stage, make a distinction if borders are periodic or not
        if self.periodical_boundary:
            fx = fx % self.stage_size
            fy = fy % self.stage_size
        else:
            fx = max(self.food_radius, min(fx, self.stage_size - self.food_radius))
            fy = max(self.food_radius, min(fy, self.stage_size - self.food_radius))

        new_food = Food(
            entity_id=len(self.foods),
            patch_id=patch_id,
            radius=self.food_radius,
            loc_x=fx,
            loc_y=fy,
            active=True
        )

        return new_food

    def reset(self, seed=None, options=None):
        # Reset time to the beginning
        self.timestep = 0
        self.num_preys = self.ini_num_preys

        # Vectorized operations for random values
        # len(self.particule_agents) can be bigger than self.num_agents
        random_values = self.np_random.random(size=(len(self.particule_agents), 3)).astype(self.float_dtype)
        loc_x = random_values[:, 0] * self.stage_size
        loc_y = random_values[:, 1] * self.stage_size
        headings = random_values[:, 2] * 2 * self.float_dtype(np.pi)

        # Assigning the vectorized values to agents
        for i, _ in enumerate(self.particule_agents):
            if i < self.num_agents:
                self.particule_agents[i].loc_x, self.particule_agents[i].loc_y = loc_x[i], loc_y[i]
                self.particule_agents[i].speed_x, self.particule_agents[i].speed_y = 0.0, 0.0
                self.particule_agents[i].heading = headings[i]
                self.particule_agents[i].still_in_game = 1


        # """
        # Initialize food patches and food
        # """
        self.food_patches = []
        for i in range(self.num_food_patch):
            patch_id = i
            cx = self.np_random.uniform(0, self.stage_size)
            cy = self.np_random.uniform(0, self.stage_size)
            self.food_patches.append({"patch_id": patch_id, "cx": cx, "cy": cy, "next_spawn_time": 0})

        # initial spawn
        self.foods = []
        for patch in self.food_patches:
            initial_count = self.np_random.randint(0, self.max_number_of_food + 1)
            for _ in range(initial_count):
                new_food = self.generate_food(patch["cx"], patch["cy"], patch["patch_id"])
                self.foods.append(new_food)

        observation_list = self._get_observation_dict()
        return observation_list, {}

    def step(self, action_list):

        all_eating_events = []
        for i in range(self.step_per_time_increment):
            action_list = action_list if i == 0 else None  # the agent use action once and then the physics do the rest
            if self.use_vectorized:
                eating_events = self._simulate_one_vectorized_step(self.dt, action_list)
            else:
                eating_events = self._simulate_one_step(self.dt, action_list)
                # append the eating events to the list
            all_eating_events.extend(eating_events)

        # Possibly spawn more food if the time interval has elapsed
        for patch in self.food_patches:
            # check the time
            if self.timestep >= patch["next_spawn_time"]:
                # check if there's already active food in that patch
                patch_food = [f for f in self.foods if f.active and f.patch_id == patch["patch_id"]]
                if len(patch_food) < self.max_number_of_food:
                    new_food = self.generate_food(patch["cx"], patch["cy"], patch["patch_id"])
                    self.foods.append(new_food)
                # schedule next spawn
                patch["next_spawn_time"] = self.timestep + (self.food_patch_regen_time or 9999999)


        reward_dict = self._get_reward(action_list, all_eating_events)
        observation_dict = self._get_observation_dict()
        terminated, truncated = self._get_done()
        self.timestep += 1

        infos = {}

        return observation_dict, reward_dict, terminated, truncated, infos

    def _simulate_one_step(self, dt, action_dict=None):
        """
        A single sub-step of physics, including collisions among agents,
        collisions with food, bouncing from walls, etc.
        """
        eating_events = []

        #######################################################################
        # 1) Agent-Agent collisions (predator eats prey).
        #######################################################################        eating_events = []
        contact_force_dict = {agent.entity_id: np.array([0.0, 0.0]) for agent in self.particule_agents}
        for agent_a in self.particule_agents:
            for agent_b in self.particule_agents:
                if agent_a.still_in_game == 0 or agent_b.still_in_game == 0:
                    continue
                if agent_a.entity_id < agent_b.entity_id:  # Avoid double-checking and self-checking
                    continue

                delta_x = agent_a.loc_x - agent_b.loc_x
                delta_y = agent_a.loc_y - agent_b.loc_y
                dist = np.sqrt(delta_x ** 2 + delta_y ** 2, dtype=self.float_dtype)
                dist_min = agent_a.radius + agent_b.radius

                if dist < dist_min:  # There's a collision
                    if agent_a.entity_type != agent_b.entity_type:
                        prey_agent, predator_agent = (agent_a, agent_b) if agent_a.entity_type == 0 else (
                            agent_b, agent_a)
                        eating_events.append({"predator_id": predator_agent.entity_id, "prey_id": prey_agent.entity_id})

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
                    contact_force_dict[agent_a.entity_id] += force
                    contact_force_dict[agent_b.entity_id] -= force  # Apply equal and opposite force

        #######################################################################
        # 2) Agent-Food collisions
        #######################################################################
        food_eating_events = []
        for food_item in self.foods:
            if not food_item.active:
                continue
            for agent in self.particule_agents:
                if agent.still_in_game == 0 or agent.entity_type == 1: # Only preys can eat
                    continue
                dist = self.compute_distance(agent, food_item)
                if dist < (agent.radius + food_item.radius):
                    # Agent picks up/eats this food
                    food_eating_events.append(
                        {"food_id": food_item.entity_id, "agent_id": agent.entity_id}
                    )
                    food_item.active = False
                    # Once eaten by one agent, break so it's not double-eaten
                    break

        if len(food_eating_events) > 0:
            eating_events.extend(food_eating_events)

        #######################################################################
        # 3) Update each agent's speed and position (simple physics)
        #######################################################################
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
                    self_force_amplitude, self_force_orientation = action_dict.get(agent.entity_id)
                    agent.heading = (agent.heading + self_force_orientation * self.max_turn) % (2 * self.float_dtype(np.pi))
                    self_force_amplitude *= self.max_acceleration_prey if agent.entity_type == 0 else self.max_acceleration_predator
                    acceleration_x += self_force_amplitude * np.cos(agent.heading)
                    acceleration_y += self_force_amplitude * np.sin(agent.heading)

                # CONTACT FORCE
                contact_force = contact_force_dict.get(agent.entity_id)
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
                max_speed = self.max_speed_prey if agent.entity_type == 0 else self.max_speed_predator
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
        reward_dict = {agent.entity_id: 0 for agent in self.particule_agents if agent.still_in_game == 1}

        # 1) Agent-vs-agent eating events (predation)
        for event in all_eating_events:
            if "predator_id" in event and "prey_id" in event:
                predator_id, prey_id = event["predator_id"], event["prey_id"]
                # Apply the eating reward for the predator and the death penalty for the prey
                reward_dict[predator_id] += self.eating_reward_for_predator
                reward_dict[prey_id] += self.death_penalty_for_prey
                # find the prey in particule_agents and set still_in_game to 0
                if self.prey_consumed:
                    self.num_preys -= 1
                    for agent in self.particule_agents:
                        if agent.entity_id == prey_id:
                            agent.still_in_game = 0

                # collective penalty for preys
                for agent in self.particule_agents:
                    if agent.entity_type == 0 and agent.still_in_game == 1:  # Prey
                        reward_dict[agent.entity_id] += self.collective_death_penalty_for_prey
                    elif agent.entity_type == 1:  # Predator
                        reward_dict[agent.entity_id] += self.collective_eating_reward_for_predator

        # 2) Agent-vs-food events
        for event in all_eating_events:
            if "food_id" in event and "agent_id" in event:
                agent_id = event["agent_id"]
                if agent_id in reward_dict:
                    reward_dict[agent_id] += self.food_reward
                # The food is marked inactive in _simulate_one_step,
                # so no further changes needed here.

        # 3) Survival / starve penalty / energy cost / edge hits
        for agent in self.particule_agents:
            if agent.still_in_game:
                if agent.entity_type == 0:  # 0 for prey, is_prey
                    reward_dict[agent.entity_id] += self.surviving_reward_for_prey
                else:  # is_predator
                    reward_dict[agent.entity_id] += self.starving_penalty_for_predator

                # ENERGY EFFICIENCY
                # Add the energy efficiency penalty
                # set the energy cost penalty
                if action_list is not None:
                    self_force_amplitude, self_force_orientation = action_list.get(agent.entity_id)

                    energy_cost_penalty = -(
                            abs(self_force_amplitude) + abs(self_force_orientation)
                    ) * self.energy_cost_penalty_coef
                    reward_dict[agent.entity_id] += energy_cost_penalty

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
                        reward_dict[agent.entity_id] += self.edge_hit_penalty

        return reward_dict

    def _get_done(self):
        # Natural ending
        # is True where the prey is eaten (agent.still_in_game == 0)
        # or when episode ends because all preys have been eaten (self.num_preys == 0)
        terminated = {
            agent.entity_id: self.num_preys == 0 or self.timestep >= self.episode_length or agent.still_in_game == 0 for
            agent in self.particule_agents}
        terminated['__all__'] = self.num_preys == 0 or self.timestep >= self.episode_length
        # Premature ending (because of time limit)
        truncated = {agent.entity_id: self.timestep >= self.episode_length for agent in self.particule_agents}
        truncated['__all__'] = self.timestep >= self.episode_length

        return terminated, truncated

    def render(self, render_mode: str = "rgb_array"):
        if render_mode != "rgb_array":
            return

        predator_color = (195, 67, 200)
        prey_color = (44, 60, 182)
        food_color = (60, 180, 75)
        patch_zone_color = (200, 255, 190)

        fig_size = 512
        pix_square_size = fig_size / self.stage_size

        # Create a blank white canvas
        canvas = np.ones((fig_size, fig_size, 3), dtype=np.uint8) * 255

        # Draw food patches
        for patch in self.food_patches:
            center = (
                int(patch["cx"] * pix_square_size),
                int(patch["cy"] * pix_square_size),
            )
            radius = int(self.food_patch_radius * pix_square_size)
            cv2.circle(canvas, center, radius, patch_zone_color, thickness=-1)

        # Draw agents
        for agent in self.particule_agents:
            if agent.still_in_game == 1:
                center = (
                    int(agent.loc_x * pix_square_size),
                    int(agent.loc_y * pix_square_size),
                )
                radius = int(agent.radius * pix_square_size)
                color = prey_color if agent.entity_type == 0 else predator_color

                # thickness = -1 means filled circle
                cv2.circle(canvas, center, radius, color, thickness=-1)

        # Draw food
        for f in self.foods:
            if f.active:
                center = (
                    int(f.loc_x * pix_square_size),
                    int(f.loc_y * pix_square_size),
                )
                radius = int(f.radius * pix_square_size)
                cv2.circle(canvas, center, radius, food_color, thickness=-1)

        return canvas

class MetricsCallbacks(RLlibCallback):
    # Based on example from https://github.com/ray-project/ray/blob/master/rllib/examples/envs/env_rendering_and_recording.py
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_episode_and_return = (None, float("-inf"))
        self.sample_step = 0

    def on_episode_step(self, *, episode, env, **kwargs):
        ## Metrics
        loc_x = [agent.loc_x for agent in env.envs[0].unwrapped.particule_agents if agent.still_in_game == 1 and agent.entity_type == 0]
        loc_y = [agent.loc_y for agent in env.envs[0].unwrapped.particule_agents if agent.still_in_game == 1 and agent.entity_type == 0]
        heading = [agent.heading for agent in env.envs[0].unwrapped.particule_agents if agent.still_in_game == 1 and agent.entity_type == 0]

        if env.envs[0].unwrapped.num_preys > 0:
            dos = calculate_dos(loc_x, loc_y) / (env.envs[0].unwrapped.num_preys * env.envs[0].unwrapped.grid_diagonal)
            doa = calculate_doa(heading) / (env.envs[0].unwrapped.num_preys * 2 * env.envs[0].unwrapped.float_dtype(np.pi))

            episode.add_temporary_timestep_data("dos", dos)
            episode.add_temporary_timestep_data("doa", doa)
            episode.add_temporary_timestep_data("time_step", 1)

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):

        ## Metrics
        # Average DoS at the end of episode
        average_dos = sum(episode.get_temporary_timestep_data("dos")) / sum(episode.get_temporary_timestep_data("time_step"))
        average_doa = sum(episode.get_temporary_timestep_data("doa")) / sum(episode.get_temporary_timestep_data("time_step"))

        metrics_logger.log_value("mean_dos", average_dos, reduce="mean")
        metrics_logger.log_value("max_dos", average_dos, reduce="max")
        metrics_logger.log_value("mean_doa", average_doa, reduce="mean")
        metrics_logger.log_value("max_doa", average_doa, reduce="max")


class RenderingCallback (RLlibCallback):
    # Based on example from https://github.com/ray-project/ray/blob/master/rllib/examples/envs/env_rendering_and_recording.py
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_episode_and_return = (None, float("-inf"))
        self.sample_step = 0

    def on_episode_step(self, *, episode, env, **kwargs):
        # Rendering
        if self.sample_step % 10 == 0:
            frame = env.envs[0].unwrapped.render()

            episode.add_temporary_timestep_data("render_images", frame)

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        # Rendering
        episode_return = episode.get_return()
        if episode_return > self.best_episode_and_return[1] and self.sample_step % 10 == 0:
            # Pull all images from the temp. data of the episode.
            images = episode.get_temporary_timestep_data("render_images")
            # `images` is now a list of 3D ndarrays
            # For WandB videos, we need to put channels first.
            video = np.transpose(np.array(images), (0, 3, 1, 2))
            # save the 4D numpy array as a gif
            # imageio.mimsave("video.gif", np.array(images), fps=10)
            if episode_return > self.best_episode_and_return[1]:
                # maybe video = video = np.expand_dims(video, axis=0) is better
                self.best_episode_and_return = (wandb.Video(video, fps=30, format="gif") , episode_return)

    def on_sample_end(self, *, metrics_logger, **kwargs) -> None:
        """Logs the best video to this EnvRunner's MetricsLogger."""
        # Best video.
        if self.best_episode_and_return[0] is not None and self.sample_step % 10 == 0:
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

        self.sample_step += 1

