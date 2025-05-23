import numpy as np

env_config = dict(
    # Computational
    use_vectorized=False,   # mostly useless for low number of fish
    step_per_time_increment = 2, # int ≥ 1
    # General
    num_preys=30,
    num_predators=4,
    stage_size=10,
    episode_length=700,
    # Physics
    inertia=True,
    dragging_force_coefficient=2,
    contact_force_coefficient=5,
    contact_margin=0.2,
    friction_regime="quadratic",            # linear=1, quadratic=2 or intermediate=1.4
    periodical_boundary=False,              # If False, the wall is solid
    wall_contact_force_coefficient=5,       # Only used when periodical_boundary=False, else ignored
    prey_consumed=False,                    # If True, prey are removed when eaten
    # Agents
    prey_radius=0.1,
    predator_radius=0.4,
    agent_density=1000,  # density of the agents to calculate the mass (should be arround 1000 if radius about 0.1)
    max_speed_prey=None,
    max_speed_predator=None,
    # Food
    num_food_patch=2,
    food_patch_radius=4,
    max_number_of_food=8,
    food_patch_regen_time=10,
    food_radius=0.1,
    food_reward=0.01,
    # Action
    max_acceleration_prey=0.2,
    max_acceleration_predator=2.0,
    max_turn=np.pi / 4,  # pi radians
    # Rewards
    # reward must be positive, penalty must be negative
    starving_penalty_for_predator=-0,
    eating_reward_for_predator=1.0,
    collective_eating_reward_for_predator=0,
    surviving_reward_for_prey=0,
    death_penalty_for_prey=-1.0,
    collective_death_penalty_for_prey=-0.001,
    edge_hit_penalty=0,
    energy_cost_penalty_coef=0.001, # positive < 1 (this is for a ratio)
    # Observations
    max_seeing_angle=3*np.pi/4,  # Put None if not used (between 0 and pi)
    max_seeing_distance=20,  # Put None if not used
    num_other_agents_observed=10,  # Put "all" if not used
    use_polar_coordinate=True,
    use_speed_observation=False,
    seed=None,
)
