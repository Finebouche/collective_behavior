import numpy as np

run_config = dict(
    # Environment settings
    env=dict(
        num_preys=20,
        num_predators=3,
        stage_size=10,
        episode_length=300,
        preparation_length=120,
        # Physics
        dragging_force_coefficient=2,
        contact_force_coefficient=0.8,
        periodical_boundary=False,  # If False, the wall is solid
        wall_contact_force_coefficient=0.8,  # Only used when periodical_boundary=False, else ignored
        prey_radius=0.1,
        predator_radius=0.2,
        agent_density=2000,  # density of the agents to calculate the mass (should be > 1000 if radius < 0.1)
        max_speed=1,
        # Action parameters
        max_acceleration=1,
        max_turn=np.pi / 2,  # pi radians
        # Reward parameters
        # reward must be positive, penalty must be negative
        starving_penalty_for_predator=-0,
        eating_reward_for_predator=1.0,
        surviving_reward_for_prey=0,
        death_penalty_for_prey=-1.0,
        edge_hit_penalty=-0.1,
        energy_cost_penalty_coef=0.0001,
        
        # Observation parameters
        max_seeing_angle=3*np.pi/4,  # Put None if not used (between 0 and pi)
        max_seeing_distance=20,  # Put None if not used
        num_other_agents_observed=6,  # Put "all" if not used
        use_polar_coordinate=True,
        seed=None,
    ),
)
