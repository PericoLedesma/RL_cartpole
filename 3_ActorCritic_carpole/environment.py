# Libraries
import time
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from IPython.display import clear_output

# Files
from utils import *

class EnvironmentClass:
    def __init__(self, env_id, render_mode, max_ep_steps) -> None:
        print(f'\n ****** Creating Environment {env_id} ... ******')
        self.env = gym.make(env_id,
                            max_episode_steps=max_ep_steps,
                            render_mode=render_mode)
        print('\tEnvironment Created. Action space: ', self.env.action_space, ' | Observation space: ', self.env.observation_space)

    def run_env(self, agent, n_episodes, batch_size, max_ep_steps, mean_batch, plot_eps_inf_every, plot, save_plot) -> None:
        print('\n', '=' * 60, '\n', ' ' * 10, f'RUN {agent.agent_name} FOR {n_episodes} EPISODES')
        start_time = time.perf_counter()

        reward_history, state_evolution = [], []

        try:
            for eps in range(n_episodes):
                eps_data = [] # for storing data from the episode
                score = 0
                truncated, terminated = False, False

                state, _ = self.env.reset()

                while not truncated and not terminated:
                    action = agent.get_action(state) # index of the action

                    state_, reward, terminated, truncated, info = self.env.step(action) # todo change reward to be positive

                    state_evolution.append((eps, state))
                    eps_data.append((eps, state, action, reward))

                    # Next steps loop
                    state = state_
                    score += reward


                # At the end of each episode
                agent.n_games += 1
                reward_history.append(score)

                if eps % plot_eps_inf_every == 0:
                    print(f"[Episode {eps}] Total Reward = {score:.2f}| Mean score = {np.mean(reward_history[-mean_batch:]):.2f}", end="\r")

                # Policy descent
                agent.policy_update(eps_data)

                # Store in replay memory
                # print('Eps', eps, ' |eps_data len ', len(eps_data))
                agent.memory.append((eps, eps_data))

                # Long memory, replay memory. If we have more than X memories
                if agent.replay: # Just for test, but it makes the agent play worse
                    if len(agent.memory) > batch_size:
                        agent.memory_replay(batch_size)


        except KeyboardInterrupt:
            print('\n*Training Interrupted')

        finally:
            elapsed_time = time.perf_counter() - start_time
            print(f'\nTraining Completed, executed in {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds', '\n', '-' * 60)
        agent.save()
        agent.store_agent_parameters(np.mean(reward_history[-mean_batch:]))
        print('=' * 60)
        # ------------------ PLOTTING RESULTS ------------------ #

        plot_rewards(reward_history, max_ep_steps, agent, plot, save_plot)

    def close(self):
        self.env.close()
        print('\n\n ****** Environment Closed ******\n\n')
