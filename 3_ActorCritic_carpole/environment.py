# Libraries
import time
import gymnasium as gym
import json

# Files
from utils import *

METADATA_FILE = 'data/agent_AC_metadata'
class EnvironmentClass:
    def __init__(self, env_id, render_mode, max_ep_steps) -> None:
        print(f'\n ****** Creating Environment {env_id} ... ******')
        self.env = gym.make(env_id,
                            max_episode_steps=max_ep_steps,
                            render_mode=render_mode)
        # print('\tEnvironment Created. Action space: ', self.env.action_space, ' | Observation space: ', self.env.observation_space)
        self.print_metadata()

    def run_env(self, agent, n_episodes, batch_size, max_ep_steps, plot_eps_inf_every, plot, save_plot, note) -> None:
        print('\n', '=' * 60, '\n', ' ' * 10, f'RUN {agent.agent_name} FOR {n_episodes} EPOCHS\n')
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
                    print(f"[Epoch {eps}] Reward = {score:.0f}| avg_score_100={np.mean(reward_history[-100:]):.2f}| avg_score_50={np.mean(reward_history[-50:]):.2f}")

                # Policy descent
                agent.policy_update(eps_data)

                # Store in replay memory
                # print('Eps', eps, ' |eps_data len ', len(eps_data))
                agent.memory.append((eps, eps_data))

                # Long memory, replay memory. If we have more than X memories
                if agent.replay and len(agent.memory) > batch_size * 4: # Just for test, but it makes the agent play worse
                    if eps % 10 == 0:
                        agent.ActorNet_target.load_state_dict(agent.ActorNet.state_dict())
                        agent.CriticNet_target.load_state_dict(agent.CriticNet.state_dict())
                    if len(agent.memory) > batch_size:
                        agent.memory_replay(batch_size)


        except KeyboardInterrupt:
            print('\n*Training Interrupted')

        finally:
            elapsed_time = time.perf_counter() - start_time
            print(f'\nTraining Completed, executed in {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.0f} seconds', '\n', '-' * 60)
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
    def close(self):
        self.env.close()
        print('\n\n ****** Environment Closed ******\n\n')
        self.print_metadata()
