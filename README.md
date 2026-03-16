# Deep Q-Network (DQN) Reinforcement Learning Agent for CartPole

A practical implementation of a **Deep Q-Network (DQN)** trained to solve the **CartPole-v1** control problem using **PyTorch** and **Gymnasium-compatible APIs**.

This project demonstrates how **value-based reinforcement learning** can be applied to learn optimal policies directly from environment interaction using neural networks.

The implementation is designed to be **clean, minimal, and extensible**, making it a useful starting point for experimenting with reinforcement learning algorithms.

---

# Project Motivation

Reinforcement learning enables agents to learn optimal decision-making policies through interaction with dynamic environments. One of the foundational algorithms in deep reinforcement learning is the **Deep Q-Network (DQN)**, which approximates the action-value function using a neural network.

This project demonstrates:

- how experience replay stabilizes learning
- how target networks prevent divergence
- how epsilon-greedy exploration balances exploration and exploitation

The **CartPole control task** serves as a classic benchmark for validating reinforcement learning algorithms and understanding their behavior.

---

# Key Features

- Implementation of **Deep Q-Network (DQN)** from scratch
- **Experience Replay Buffer** for stable off-policy learning
- **Target Network Synchronization** for stable Q-value estimation
- **Huber Loss (`smooth_l1_loss`)** for robust training
- **Epsilon-Greedy Exploration Strategy**
- **Gymnasium-compatible API handling**
- Automatic **model checkpoint saving**

---

# Tech Stack

- **Python**
- **PyTorch**
- **OpenAI Gym / Gymnasium**
- **NumPy**
- **Reinforcement Learning (Value-Based Methods)**

---

# Repository Structure

```
Deep-Q-Network-Reinforcement-Learning-Agent-for-CartPole
│
├── dqn.py
│   Main training script implementing the DQN algorithm
│
├── dqn_cartpole_model.pth
│   Saved trained model weights
│
├── requirements.txt
│   Python dependencies
│
├── LICENSE
│
└── README.md
```

---

# Deep Q-Network Overview

The DQN algorithm approximates the optimal action-value function:

Q(s,a) ≈ E[Rₜ | sₜ = s, aₜ = a]

The neural network predicts Q-values for each possible action.

During training:

1. The agent observes state **s**
2. Chooses action **a** using epsilon-greedy exploration
3. Receives reward **r** and next state **s'**
4. Stores transition in the **replay buffer**
5. Samples mini-batches from memory
6. Updates Q-network using gradient descent

The training target follows the Bellman equation:

y = r + γ maxₐ' Q_target(s', a')

A separate **target network** is periodically updated to stabilize learning.

---

# Neural Network Architecture

The Q-network is implemented as a simple **Multi-Layer Perceptron (MLP)**.

Input:
- 4-dimensional state vector (CartPole environment)

Hidden Layers:

- Fully Connected (128 units) + ReLU
- Fully Connected (128 units) + ReLU

Output:

- 2 Q-values corresponding to the possible actions:
  - push cart left
  - push cart right

---

# Hyperparameters

| Parameter | Value |
|----------|-------|
| Learning Rate | 0.0005 |
| Discount Factor (γ) | 0.98 |
| Replay Buffer Size | 50,000 |
| Batch Size | 32 |
| Training Episodes | 10,000 |
| Training Updates | 10 per optimization phase |

Rewards are scaled before being stored in memory to improve training stability.

---

# Example Training Output

During training, logs are printed every 20 episodes.

Example:

```
Episode 20 | Avg Score: 9.7 | Replay Buffer: 398 | Epsilon: 7.8%
Episode 40 | Avg Score: 9.8 | Replay Buffer: 791 | Epsilon: 7.6%
Episode 60 | Avg Score: 9.9 | Replay Buffer: 1183 | Epsilon: 7.4%
```

This indicates the agent progressively learning to balance the pole.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Pavankumarmanagoli/Deep-Q-Network-Reinforcement-Learning-Agent-for-CartPole.git
cd Deep-Q-Network-Reinforcement-Learning-Agent-for-CartPole
```

---

## Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

If using Gymnasium:

```bash
pip install gymnasium[classic-control]
```

---

# Training the Agent

Run the training script:

```bash
python dqn.py
```

During training the script will:

- interact with the environment
- store experiences in replay memory
- update the Q-network
- periodically synchronize the target network

After training completes, the model checkpoint will be saved as:

```
dqn_cartpole_model.pth
```

---

# Results

After training, the agent successfully learns to balance the pole for extended durations.

Typical performance:

- Average reward approaching **200 (environment limit)**
- Stable control policy learned through Q-value approximation

---

# Possible Improvements

Several extensions can significantly improve performance:

- **Double DQN**
- **Dueling Network Architecture**
- **Prioritized Experience Replay**
- **Soft Target Updates**
- **TensorBoard Logging**
- **Evaluation Script with Visualization**
- **Multi-Environment Training**

These extensions are natural next steps for improving the agent.

---

# Troubleshooting

## ModuleNotFoundError for gym

Install Gymnasium classic control environments:

```bash
pip install gymnasium[classic-control]
```

---

## Slow training

CartPole is lightweight but performance may vary depending on CPU speed and Python environment.

Ensure PyTorch is installed with optimized BLAS libraries.

---

# License

This project is licensed under the terms described in the **LICENSE** file.

---

# Acknowledgments

- PyTorch for neural network tooling
- OpenAI Gym / Gymnasium ecosystem for RL environments
- Reinforcement learning research community for foundational DQN work
