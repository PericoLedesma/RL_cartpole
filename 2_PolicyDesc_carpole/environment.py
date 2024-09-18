# Libraries
import time
import gymnasium as gym
import json

# Files
from utils import *

METADATA_FILE = 'data/agent_PG_metadata'
class EnvironmentClass:
    def __init__(self, env_id, render_mode, max_ep_steps) -> None:
        print(f'\n ****** Creating Environment {env_id} ... ******')
        self.env = gym.make(env_id,
                            max_episode_steps=max_ep_steps,
                            render_mode=render_mode)  # 'human ' or 'rgb_array', 'ansi'
        print('\tEnvironment Created. Action space: ', self.env.action_space, ' | Observation space: ', self.env.observation_space)
        self.print_metadata()

    def run_env(self, agent, n_episodes, max_ep_steps, batch_size, plot_eps_inf_every, plot, save_plot, note) -> None:
        print('\n', '=' * 60, '\n', ' ' * 10, f'RUN {agent.agent_name} FOR {n_episodes} EPOCHS\n')
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
                    print(f"[Epoch {eps}] Reward = {score:.d}| avg_score_100={np.mean(reward_history[-100:]):.2f}| avg_score_50={np.mean(reward_history[-50:]):.2f}")

                # Policy descent
                agent.policy_update(eps_data)
                # agent.policy_update2(eps_data)



        except KeyboardInterrupt:
            print('\n', '*Training Interrupted', '\n')

        finally:
            elapsed_time = time.perf_counter() - start_time
            print('\n', '=' * 60, '\n', f'Training Completed, executed in {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds', '\n', '-' * 60)

        # agent.save()
        agent.store_agent_parameters(np.mean(reward_history[-500:]), np.mean(reward_history[-100:]), note)
        print('=' * 60)
        # ------------------ PLOTTING RESULTS ------------------ #
        plot_rewards(reward_history, max_ep_steps, agent, plot, save_plot)

    def print_metadata(self):
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)

            print('\n====>Models metadata... ')
            for model, date in metadata.items():
                print(f"\t\t-->{model} : ")
                for date, data in date.items():
                    print(f"\t\t\t]{date}] : {data}")
        else:
            print(f"No metadata of the agent found.")
    def close(self):
        self.env.close()
        print('\n\n ****** Environment Closed ******\n\n')

        self.print_metadata()
