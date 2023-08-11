import numpy as np

run_config = dict(
    # Environment settings
    env=dict(
        num_preys=15,
        num_predators=2,
        stage_size=30,
        episode_length=1000,
        preparation_length=120,
        # Physics
        dragging_force_coefficient=0.5,
        contact_force_coefficient=0.2,
        wall_contact_force_coefficient=0.2,
        prey_size=0.1,
        predator_size=0.2,
        min_speed=0,
        max_speed=100,
        # Action parameters
        max_acceleration=0.1,
        min_acceleration=0,
        max_turn=np.pi / 2,  # pi radians
        min_turn=- np.pi / 2,  # pi radians
        # Reward parameters
        # reward must be positive, penalty must be negative
        starving_penalty_for_predator=--0,
        eating_reward_for_predator=1.0,
        surviving_reward_for_prey=0,
        death_penalty_for_prey=-10.0,
        edge_hit_penalty=-0,
        use_energy_cost=True,
        
        # Observation parameters
        use_full_observation=False,  # Put False if not used
        max_seeing_angle=None,  # Put None if not used
        max_seeing_distance=None,  # Put None if not used
        num_other_agents_observed=8,  # Put None if not used
        use_polar_coordinate=True,
        seed=None,
    ),
    # Policy network settings
    policies=dict(  # list all the policies below
        prey=dict(
            to_train=True,  # flag indicating whether the model needs to be trained
            algorithm="PPO",  # algorithm used to train the policy
            gamma=0.98,  # discount rate gamms
            lr=0.001,  # learning rate
            vf_loss_coeff=1,  # loss coefficient for the value function loss
            entropy_coeff=[[0, 0.5], [2000000, 0.05]],  # entropy coefficient (can be a list of lists)
            model=dict(  # policy model settings
                type="prey_policy",
                fc_dims=[64, 64, 64],  # dimension(s) of the fully connected layers as a list
                model_ckpt_filepath="",  # filepath (used to restore a previously saved model)
            ),
        ),
        predator=dict(
            to_train=True,
            algorithm="PPO",
            gamma=0.98,
            lr=0.001,
            vf_loss_coeff=1,
            entropy_coeff=[[0, 0.5], [2000000, 0.05]],  # entropy coefficient (can be a list of lists)
            model=dict(
                type="predator_policy",
                fc_dims=[64, 64, 64],
                model_ckpt_filepath="",
            )
        )
    ),
)
