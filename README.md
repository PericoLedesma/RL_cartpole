# Reinforcement Learning Algorithms on CartPole Environment(WIP)

## Introduction

This project implements three fundamental reinforcement learning algorithms on the OpenAI Gym's CartPole environment:

- **Deep Q-Network (DQN)**
- **Policy Gradient Descent**
- **Actor-Critic**

The goal is to compare the performance and learning behaviors of these algorithms in a controlled setting.

## Algorithms Implemented

### Deep Q-Network (DQN)

DQN is a value-based reinforcement learning algorithm that approximates the Q-value function using a neural network. It utilizes experience replay and target networks to stabilize training and overcome the instability caused by function approximation in Q-learning.

### Policy Gradient Descent

Policy Gradient methods directly adjust the policy parameters by maximizing the expected reward. This is achieved by computing the gradient of the policy's performance with 
respect to its parameters and updating the policy in the direction that increases expected rewards.  
Results:  
	-> model_16 : {'n_games': 1000, 'mean_score': 55.76}  
	-> model_32 : {'n_games': 1000, 'mean_score': 212.6}  
	-> model_64 : {'n_games': 1000, 'mean_score': 459.48}  
	-> model_128 : {'n_games': 1000, 'mean_score': 410.2}  

### Actor-Critic

The Actor-Critic method combines both policy-based and value-based methods. The actor learns the policy function, while the critic estimates the value function. This combination allows for efficient learning by reducing variance and bias in policy updates.
Note: experimenting with memory replay  

Results(with and without replay memory)  
	-> model_64_replayTrue : {'n_games': 1000, 'mean_score': 32.51}  
	-> model_64_replayFalse : {'n_games': 1000, 'mean_score': 493.76}  
	-> model_128_replayTrue : {'n_games': 1000, 'mean_score': 29.04}  
	-> model_128_replayFalse : {'n_games': 1000, 'mean_score': 472.05}  
	-> model_256_replayTrue : {'n_games': 1000, 'mean_score': 63.85}  
	-> model_256_replayFalse : {'n_games': 1000, 'mean_score': 481.56}  
	-> model_64_128_replayTrue : {'n_games': 1000, 'mean_score': 181.89}  
	-> model_64_128_replayFalse : {'n_games': 1000, 'mean_score': 485.38}  


## ToDo
    PG: rewards = (rewards - rewards_mean) / rewards_std

## Environment

[CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) is a classic control problem where the agent's goal is to keep a pole balanced upright on a cart by applying forces to move the cart left or right.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/PericoLedesma/RL_carpole.git
   cd cartpole-rl-algorithms
   ```

2. **Run any of the algorithms in the algorithm directory.**

   ```bash
    python main.py
    ```

Notes:
- Delete model files in the model directory to train a new model.

