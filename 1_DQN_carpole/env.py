import gym
import numpy as np
import time

from utils import *


class EnvironmentClass:
    def __init__(self, env_id, render_mode, max_ep_steps):
        print(f'\n ****** Creating Environment {env_id} ... ******')
        self.env = gym.make(env_id,
                            render_mode=render_mode)
        print('\tEnvironment Created. Action space: ', self.env.action_space, ' | Observation space: ', self.env.observation_space)

    def run_env(self, agent, n_episodes, batch_size, max_ep_steps, save_model, plot, save_plot):
        print('\n', '=' * 60, '\n', ' ' * 10, f'RUN {agent.agent_name} FOR {n_episodes} EPISODES')
        start_time = time.perf_counter()

        reward_history, epsilon_history = [], []
        try:
            for i in range(n_episodes):
                done, truncated = False, False
                score = 0
                state, _ = self.env.reset()

                while not done:
                    action = agent.get_action(state)
                    state_, reward, done, truncated, info = self.env.step(action)

                    # Short memory, just this step
                    agent.learning(state, action, reward, state_, done)

                    # Store in replay memory
                    # print('Storing in replay memory ... Number of memories ', len(agent.memory) + 1)
                    agent.memory.append((state, action, reward, state_, done))

                    state = state_
                    score += reward
                    if score > max_ep_steps:
                        break

                # At the end of each episode
                agent.n_games += 1
                agent.linear_epsilon_decay(i, n_episodes)
                reward_history.append(score)
                epsilon_history.append(agent.epsilon)

                message = f"[Episode {i + 1}] Total Reward/steps = {score}| Average score = {np.mean(reward_history[-100:]):.2f}| Epsilon = {agent.epsilon:.2f}"
                print('\r', message, end='')

                # Long memory, replay memory
                if len(agent.memory) > batch_size:
                    agent.memory_replay(batch_size)

                # Update target network
                # Optionally update target network periodically
                # if episode % 10 == 0:
                agent.Q_target.load_state_dict(agent.Q_model.state_dict())


        except KeyboardInterrupt:
            print('\n*Training Interrupted', '\n')

        finally:
            elapsed_time = time.perf_counter() - start_time
            print(f'\nTraining Completed, executed in {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds', '\n', '-' * 60)
        if save_model:
            agent.save()
            agent.store_agent_parameters(np.mean(reward_history[-500:]))
        print('=' * 60)
        # ------------------ PLOTTING RESULTS ------------------ #

        plot_rewards(reward_history, epsilon_history, max_ep_steps, agent, plot, save_plot)

    def close(self):
        self.env.close()
        print('\n ****** Environment Closed ******')
