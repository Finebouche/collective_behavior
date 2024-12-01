<p align="center">
    <img src="collective_behavior.png" align="center" width="30%">
</p>
<p align="center"><h1 align="center">COLLECTIVE_BEHAVIOR</h1></p>
<p align="center">
	<em>Unleashing Dynamics in Multi-Agent Simulations</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/Finebouche/collective_behavior?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Finebouche/collective_behavior?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Finebouche/collective_behavior?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Finebouche/collective_behavior?style=default&color=0080ff" alt="repo-language-count">
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
  - [🤖 Usage](#🤖-usage)
  - [🧪 Testing](#🧪-testing)
- [📌 Project Roadmap](#-project-roadmap)
- [🎗 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

The collectivebehavior project addresses the challenge of understanding and simulating multi-agent dynamics, such as predator-prey interactions. It offers customizable simulations, detailed metrics, and engaging visualizations to analyze agent behavior and interactions. Ideal for researchers and developers in AI and behavioral sciences, it supports experimentation and performance tracking, enhancing insights into collective dynamics.

---

|      |     Feature      | Summary       |
| :--- |:----------------:| :---          |
| ⚙️  | **Architecture** | <ul><li>Utilizes a multi-agent simulation framework leveraging `<Ray RLlib>` for reinforcement learning.</li><li>Incorporates a 2D particle-based environment for agent interactions, defined in [`particle_2d_env.py`](#).</li><li>Supports customizable and repeatable simulation scenarios through configuration settings in [`config.py`](#).</li></ul> |
| 🔌 |  **Monitoring**  | <ul><li>Seamless integration with `<Weights & Biases>` for experiment tracking and visualization.</li><li>Utilizes `<JupyterLab>` for interactive development and analysis.</li><li>Supports visualization tools like `<Pygame>` for rendering simulations.</li></ul> |
| 🧩 |  **Modularity**  | <ul><li>Encapsulates agent behaviors and properties in the `ParticuleAgent` class.</li><li>Modular design allows for easy extension and modification of simulation parameters.</li><li>Separate modules for metrics, configuration, and visualization enhance maintainability.</li></ul> |
| ⚡️  | **Performance**  | <ul><li>Efficient simulation of multi-agent environments using vectorized operations.</li><li>Leverages `<Ray>` for distributed computing and scalability.</li><li>Optimized for performance with tools like `<NumPy>` and `<PyTorch>`.</li></ul> |
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
## 📌 Project Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

## 🎗 License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## 🙌 Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---
