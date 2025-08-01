
---

# RPS: Information Elicitation with Reinforcement Prompt Selection

## Overview

This repository provides a reinforcement learning (RL) framework for dynamic prompt selection in legal dialogue scenarios. The system is designed to train and evaluate RL agents (e.g., DQN-based) in simulated multi-turn interactions between a language model and a user, aiming to improve prompt policy adaptability using user feedback and strategic questioning.

## Repository Structure

* **`prompt/`**
  Contains predefined dialogue prompts and templates used to simulate legal consultation cases.

* **`RPS/`**
  Includes the core implementation of the DQN algorithm and its training routines.

* **`agent.py`**
  Defines the tools need to use.

* **`conversation.py`**
  Implements the dialogue simulation environment, modeling the interaction between the agent (lawyer) and user (defendant).

* **`DB.py`**
  Manages data structures for storing and retrieving case-specific information and user responses.

* **`embedding_evaluate.py`**
  Provides tools for evaluating dialogue outcomes based on embedding-based similarity or alignment metrics.

* **`plot_variance.py`**
  Visualization utilities for plotting experimental metrics such as reward variance and confidence intervals.

* **`requirements.txt`**
  Lists all Python dependencies required to run the project.


## Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage
### 1. Running Single Prompt Dialogue Simulation
Use this mode to simulate a dialogue with a manually specified or static prompt template.
1. **Simulating Conversations**:

   ```bash
   python conversation.py
   ```

2. **Evaluating Embedding Metrics**:

   ```bash
   python embedding_evaluate.py
   ```

3. **Plotting Result Variance**:

   ```bash
   python plot_variance.py
   ```

### 2. Running Reinforcement Learning for Prompt Selection
Use this mode to train and evaluate an RL agent (e.g., DQN) that dynamically selects prompts based on dialogue history and feedback.
1. **Train RPS mode**:

   ```bash
   python RPS/train.py
   ```

## License

MIT License 

