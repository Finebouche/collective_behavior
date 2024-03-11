import numpy as np

run_config = dict(
    # Environment settings
    env=dict(
        # Computational
        use_vectorized=False,   # mostly useless for low number of fish
        temporal_increment = 1, # default should be 1
        # General
        num_preys=[10, 20],
        num_predators=[2, 4],
        stage_size=25,
        episode_length=500,
        prey_consumed=False,
        # Physics
        dragging_force_coefficient=2,
        contact_force_coefficient=25,
        contact_margin=0.5,
        friction_regime="linear",            # linear, quadratic or intermediate
        periodical_boundary=False,           # If False, the wall is solid
        wall_contact_force_coefficient=3,    # Only used when periodical_boundary=False, else ignored
        prey_radius=0.1,
        predator_radius=0.2,
        agent_density=2000,  # density of the agents to calculate the mass (should be > 1000 if radius about 0.1)
        max_speed_prey=0.3,
        max_speed_predator=0.5,
        # Action
        max_acceleration_prey=1,
        max_acceleration_predator=2,
        max_turn=np.pi / 4,  # pi radians
        # Rewards
        # reward must be positive, penalty must be negative
        starving_penalty_for_predator=-0,
        eating_reward_for_predator=1.0,
        collective_eating_reward_for_predator=0,
        surviving_reward_for_prey=0,
        death_penalty_for_prey=-1.0,
        collective_death_penalty_for_prey=-0,
        edge_hit_penalty=-0.001,
        energy_cost_penalty_coef=0.00001, # positive (this is for a ratio)
        # Observations
        max_seeing_angle=3*np.pi/4,  # Put None if not used (between 0 and pi)
        max_seeing_distance=20,  # Put None if not used
        sort_by_distance=True,
        num_other_agents_observed=6,  # Put "all" if not used
        use_polar_coordinate=True,
        use_speed_observation=False,
        seed=None,
    ),
)
