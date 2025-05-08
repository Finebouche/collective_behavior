import os
import ray

import psutil
from ray.tune.registry import get_trainable_cls
from ray.rllib.policy.policy import PolicySpec

from particle_2d_env import Particle2dEnvironment, RenderingCallback, MetricsCallbacks
from config import env_config

from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

# Can be modified
os.environ["RAY_DEDUP_LOGS"] = "0"

print("Ray version :", ray.__version__)
print("Number of CPUs: ", psutil.cpu_count())

num_cpus = 6
num_gpus = 1

if psutil.cpu_count() is not None:
    assert num_cpus <= psutil.cpu_count()
else:
    raise ValueError("Unable to determine the number of CPUs available.")

ALGO = "PPO" # or "PPO", "DQN", "SAC", "APPO"
FRAMEWORK= "torch"
env = Particle2dEnvironment(env_config)

ppo_config = (
    PPOConfig()
    .environment(Particle2dEnvironment, env_config=env_config, disable_env_checking=True)
    .framework(FRAMEWORK,)
    .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True,)
    .callbacks(callbacks_class=[RenderingCallback, MetricsCallbacks])
    .rl_module(
        model_config={
            "fcnet_hiddens" : [128, 128, 128],
            "use_attention" : True,
        }
    )
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
        policy_mapping_fn = lambda id, *arg, **karg: "prey" if env.particule_agents[id].entity_type == 0 else "predator",
        policies_to_train=["prey", "predator"],
        count_steps_by="agent_steps",
    )
    .training(
        num_epochs=10,
        train_batch_size_per_learner=512,
    )
    # what will learn the model
    .learners(
        num_learners=1,          # or >2
        num_cpus_per_learner=2,  # <- default 1
        num_gpus_per_learner=0,  # <- default 0
    )
    # what will run the training
    .resources(
        num_cpus_for_main_process=1  # <- default  1
    )
    # what will run the environment
    .env_runners(
        num_env_runners=1,
        num_envs_per_env_runner=1,
        num_cpus_per_env_runner=2,
        rollout_fragment_length="auto",
        batch_mode="complete_episodes",# truncate_episodes or complete_episodes
    )
    .checkpointing(export_native_model_files=True)
)

from ray.air.integrations.wandb import WandbLoggerCallback

# Read the API key from the file to use Wanddb
with open('wandb_api_key.txt', 'r') as file:
    api_key = file.read().strip()

from ray import tune

ray.init(
    num_cpus=num_cpus,
    num_gpus=num_gpus,
    ignore_reinit_error=True,
)

############################################
# Where to save
############################################
# absolute path + ray_results directory
storage_path = os.getcwd() + "/ray_results"
checkpoint_folder = None  # is something like "PPO_2024-12-19_01-09-51"

############################################
# Config
############################################
import numpy as np
config_dict = ppo_config.to_dict()

# 🔽 Put any Tune search-space objects **exactly** where the real value lives
config_dict["env_config"].update({
    "dragging_force_coefficient": tune.grid_search([1.0, 2.0, 3.0]),
    "max_acceleration_prey":      tune.grid_search([0.1, 0.2, 0.3]),
    "max_turn":                   tune.grid_search([np.pi / 6, np.pi / 4]),
})
# You can mix grid / choice / loguniform:
#config_dict["training"]["entropy_coeff"] = tune.loguniform(1e-4, 3e-3)

############################################
# Build the Tuner
############################################
tune_callbacks = [
    WandbLoggerCallback(
        project="marl-rllib",
        group=None,
        api_key=api_key,
        log_config=True,
        upload_checkpoints=False
    ),
]

tuner = tune.Tuner(
    trainable=ALGO,  # Defined before
    param_space=config_dict,  # Defined before
    tune_config=tune.TuneConfig(
        metric="schooling_score",
        mode="max",
    ),
    run_config=tune.RunConfig(
        storage_path=storage_path,
        stop={"training_iteration": 1500},
        verbose=3,
        callbacks=tune_callbacks,
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_at_end=True,
            checkpoint_frequency=100,
        ),
    ),
)

# Run the experiment
results = tuner.fit()

ray.shutdown()
