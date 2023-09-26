# Reinforcement Learning with Ray RLlib

This repository contains the implementation of a multi-agent reinforcement learning (RL) scenario using Ray's RLlib. It uses Proximal Policy Optimization (PPO) as the RL algorithm and enables training and experimentation with different configurations.

## Dependencies
This project depends on the Ray library for distributed computing, along with its RLlib for reinforcement learning.

## Setup

### Creating a Conda Environment
To install the dependencies, you should create a Conda environment. If you haven’t installed Conda, please follow the [official installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Once you have Conda installed, you can create an environment using the following command:

```sh
conda create --name <env_name>
```

Replace ```<env_name>``` with the name you wish to give to the environment.
Activating the Conda Environment

To activate the created Conda environment, use the command below:

```sh
conda activate <env_name>
```

Replace ```<env_name>``` with the name of your environment.

### Installing Dependencies

To install the required dependencies, you can use the requirements.txt file. Run the following command:

```sh
pip install -r requirements.txt
```

## Code Overview

The provided code is structured as follows:

    custom_env.py includes the implementation of the custom environment.
    config.py contains various configurations for running the experiment.
    The main script initializes Ray, configures and runs the PPO algorithm, and handles multi-agent scenarios.

### Args Class

This class initializes the parameters used for running the RL algorithm:

    run: Specifies the RL algorithm to be used. Here, it's PPO.
    framework: Indicates the deep learning framework to be used, either "tf2" or "torch". Here, it's "torch".
    stop_iters, stop_timesteps, and stop_reward: Stop the training iterations, time steps, and reward, respectively.

### Custom Environment and Configuration

The custom_env.py file should define the custom environment class CustomEnvironment and config.py should contain the run_config configuration dict which will be used when initializing the environment instance.
### Multi-Agent Configuration

The script supports multi-agent configurations, specifically "prey" and "predator". The policy_mapping_fn determines the policy for each agent based on the agent_type.
### Resource Allocation

The resources used by Ray, like GPUs, CPUs per worker, etc., are configured in the resources section of the config object.

#### Note

Ensure that your machine has the required resources, like GPUs and CPUs, as configured in the script.

### Training and Rollouts

The training parameters like the number of SGD iterations, minibatch size, etc., are configured, as are the rollout parameters like fragment length, batch mode, and the number of rollout workers.
Checkpointing

The model’s native files are exported during checkpointing.

## Integration with Weights & Biases

The script also includes integration with Weights & Biases (W&B) for experiment logging through WandbLoggerCallback.