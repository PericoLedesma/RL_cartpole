import matplotlib.pyplot as plt
import math


# Linear epsilon decay function
def linear_epsilon_decay(episode, epsilon_max=1.0, epsilon_min=0.1, n_episodes=1000):
    """
    Linearly decay epsilon from epsilon_max to epsilon_min over n_episodes.

    Parameters:
    - episode (int): The current episode number.
    - epsilon_max (float): The starting epsilon value (maximum).
    - epsilon_min (float): The minimum epsilon value after decay.
    - n_episodes (int): The total number of episodes over which epsilon should decay.

    Returns:
    - float: The decayed epsilon value for the current episode.
    """
    decay_rate = (epsilon_max - epsilon_min) / n_episodes
    epsilon = epsilon_max - episode * decay_rate
    return max(epsilon, epsilon_min)


# Exponential epsilon decay function
def exponential_epsilon_decay(episode, epsilon_max=1.0, epsilon_min=0.1, decay_rate=0.005):
    """
    Exponentially decay epsilon from epsilon_max to epsilon_min.

    Parameters:
    - episode (int): The current episode number.
    - epsilon_max (float): The starting epsilon value (maximum).
    - epsilon_min (float): The minimum epsilon value after decay.
    - decay_rate (float): The rate of decay. Higher values result in faster decay.

    Returns:
    - float: The decayed epsilon value for the current episode.
    """
    epsilon = epsilon_min + (epsilon_max - epsilon_min) * math.exp(-decay_rate * episode)
    return max(epsilon, epsilon_min)


# Parameters
n_episodes = 2000  # Total number of episodes
epsilon_max = 1.0
epsilon_min = 0.1

# Exponential decay rate (you can adjust this for faster/slower decay)
exp_decay_rate = 0.005

# Calculate epsilon values over episodes for both decay strategies
epsilon_values_linear = [linear_epsilon_decay(episode, epsilon_max, epsilon_min, n_episodes) for episode in range(n_episodes)]
epsilon_values_exponential = [exponential_epsilon_decay(episode, epsilon_max, epsilon_min, exp_decay_rate) for episode in range(n_episodes)]

# Plot the evolution of epsilon for both decay methods
plt.figure(figsize=(10, 6))

# Plot linear decay
plt.plot(epsilon_values_linear, label='Linear Decay', color='blue')

# Plot exponential decay
plt.plot(epsilon_values_exponential, label='Exponential Decay', color='red')

plt.title('Epsilon Evolution: Linear vs Exponential Decay')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.legend()

# Show the plot
plt.show()