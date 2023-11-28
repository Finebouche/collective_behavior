import numpy as np

run_config = dict(
    # Environment settings
    env=dict(
        num_preys=[20, 60],
        num_predators=[2, 5],
        stage_size=[30, 50],
        episode_length=300,
        preparation_length=120,
        # Physics
        dragging_force_coefficient=1,
        contact_force_coefficient=0.2,
        wall_contact_force_coefficient=0.4,
        prey_radius=0.1,
        predator_radius=0.2,
        max_speed=1,
        # Action parameters
        max_acceleration=1,
        min_acceleration=0,
        max_turn=np.pi / 2,  # pi radians
        # Reward parameters
        # reward must be positive, penalty must be negative
        starving_penalty_for_predator=-0,
        eating_reward_for_predator=10.0,
        surviving_reward_for_prey=0,
        death_penalty_for_prey=-10.0,
        edge_hit_penalty=-0,
        energy_cost_penalty_coef=0.001,
        
        # Observation parameters
        max_seeing_angle=3*np.pi/4,  # Put None if not used (between 0 and pi)
        max_seeing_distance=20,  # Put None if not used
        num_other_agents_observed=6,  # Put "all" if not used
        use_polar_coordinate=True,
        seed=None,
    ),
)
