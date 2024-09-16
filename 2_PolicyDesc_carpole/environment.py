# Libraries
import time
import gymnasium as gym

# Files
from utils import *

class EnvironmentClass:
    def __init__(self, env_id, render_mode, max_ep_steps) -> None:
        print(f'\n ****** Creating Environment {env_id} ... ******')
        self.env = gym.make(env_id,
                            max_episode_steps=max_ep_steps,
                            render_mode=render_mode)  # 'human ' or 'rgb_array', 'ansi'
        print('\tEnvironment Created. Action space: ', self.env.action_space, ' | Observation space: ', self.env.observation_space)

    def run_env(self, agent, n_episodes, max_ep_steps, mean_batch, plot_eps_inf_every, plot, save_plot) -> None:
        print('\n', '=' * 60, '\n', ' ' * 10, f'RUN {agent.agent_name} FOR {n_episodes} EPISODES')
        start_time = time.perf_counter()

        reward_history, state_evolution = [], []

        try:
            for eps in range(n_episodes):
                eps_data = []
                score = 0
                truncated, terminated = False, False

                state, _ = self.env.reset()

                while not truncated and not terminated:
                    action = agent.get_action(state) # index of the action=0 or 1= action

                    state_, reward, terminated, truncated, info = self.env.step(action)

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
                agent.policy_update2(eps_data)



        except KeyboardInterrupt:
            print('\n', '*Training Interrupted', '\n')

        finally:
            elapsed_time = time.perf_counter() - start_time
            print('\n', '=' * 60, '\n', f'Training Completed, executed in {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds', '\n', '-' * 60)

        agent.save()
        agent.store_agent_parameters(np.mean(reward_history[-mean_batch:]))
        print('=' * 60)
        # ------------------ PLOTTING RESULTS ------------------ #
        plot_rewards(reward_history, max_ep_steps, agent, plot, save_plot)

    def close(self):
        self.env.close()
        print('\n\n ****** Environment Closed ******\n\n')
