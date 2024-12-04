<p align="center">
    <img src="collective_behavior.png" align="center" width="70%">
</p>
<p align="center"><h1 align="center">EMERGENT COLLECTIVE BEHAVIOR</h1></p>
<p align="center">
	<em>Exploring the Emergence of Confusion in Collective behavior through Reinforcement Learning Simulation</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/Finebouche/collective_behavior" alt="license">
	<img src="https://img.shields.io/github/last-commit/Finebouche/collective_behavior" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Finebouche/collective_behavior?style=default&color=0080ff" alt="repo-top-language">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

## 🔗 Table of Contents

- [📍 Overview](#-overview)
- [🚀 Setup](#-getting-started)
  - [☑️ Prerequisites](#-prerequisites)
  - [⚙️ Installation](#-installation)
- [🎗 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

This project delves into the fascinating dynamics of animal collective behavior, exploring how confusion and order emerge in multi-agent systems. Using Reinforcement Learning (RL) powered by Ray RLlib, the simulation provides a framework for studying realistic interactions between agents in a 2D environment. With a focus on physics realism and behavioral analysis, the project enables researchers to simulate and study behaviors such as predator-prey dynamics, flocking, or swarming, bridging the gap between biological studies and AI-driven insights.

---
*	Switchable Optimization Algorithms: Effortlessly toggle between various RL optimization algorithms to test and analyze behaviors.
*	Realistic Physics: Incorporates particle-based physics to model agent movement and interactions accurately.
*	Behavioral Insights: Focus on metrics and visualizations that reveal emergent patterns and their implications in collective dynamics.

---
## 🛠 Key Features

|    |      Feature      | Summary                                                                                                                                                                                                                                                                                  |
|:---|:-----------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🧪 | **Architecture**  | Leverages Ray RLlib for multi-agent reinforcement learning simulations.Interactive 2D environment for physics-based agent interactions, implemented in particle_2d_env.py.Highly configurable scenarios via config.py for flexible experimentation.                                                                                                                                                                                                                                                                                         |
| 🩻 | **Visualization** | Real-time visualizations using Pygame, making simulations engaging and accessible.Seamless tracking and analysis through Weights & Biases integration.Interactive data exploration with JupyterLab notebooks.                    |
| 🧩 |  **Modularity**   | Encapsulated behaviors in ParticleAgent class for extensibility.Independent modules for metrics, configuration, and visualization to maintain clean architecture.Easy-to-adapt structure for new scenarios or agent types. |
| ⚡️ |  **Performance**  | Optimized multi-agent simulations with vectorized operations for scalable performance.                                                                                                                                                                                          |
---


## Setup

### ☑️ Prerequisites

Before getting started with collective_behavior, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Conda


### ⚙️ Installation

Install collective_behavior using one of the following methods:

**Build from source:**

1. Clone the collective_behavior repository:
```sh
❯ git clone https://github.com/Finebouche/collective_behavior
```

2. Navigate to the project directory:
```sh
❯ cd collective_behavior
```

3. Install the project dependencies:


**Using `conda`** &nbsp; [<img align="center" src="https://img.shields.io/badge/conda-342B029.svg?style={badge_style}&logo=anaconda&logoColor=white" />](https://docs.conda.io/)

```sh
❯ conda env create -f environment.yml
```

4. Register the environment as jupyter kernel:
```sh
❯ python -m ipykernel install --user --name collective_behavior
```

5. Specify you wandb API key in  wandb_api_key.txt file

6. Activate the conda environment:
```sh
❯ conda activate collective_behavior
```
7. Start JupyterLab:
```sh
❯ jupyter lab
```

---


## 🎗 License

This project is protected under the [MIT License ](https://choosealicense.com/licenses/mit/) License.

---

## 🙌 Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---
