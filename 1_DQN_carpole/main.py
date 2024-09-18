from agent import DQN_Agent
from env import EnvironmentClass

# -----------------------------
'''
Notes: 
action_space = Discrete(2)  # Example: action = 0 (left), action = 1 (right)
Reward = steps, binary, where the agent receives +1 for every step it balances the pole.

# To change thresold: **/anaconda3/envs/general/lib/python3.11/site-packages/gym/envs/classic_control/cartpole.py
'''

MAX_EPISODE_STEPS = 500  # Default should be 500


def main():
    # ------------------ ENVIRONMENT  ------------------ #
    env_class = EnvironmentClass(env_id='CartPole-v1',
                                 render_mode='rgb_array',  # 'human ' or 'rgb_array'
                                 max_ep_steps=MAX_EPISODE_STEPS)

    # ------------------ AGENTS  ------------------ #
    agents = {}
    for layers in [[32], [64], [128]]:
        agents[f"model_{layers}"] = DQN_Agent(env_class=env_class,
                                              lr=0.001,
                                              gamma=0.99,
                                              epsilon_max=1,
                                              epsilon_decay=0.005,
                                              epsilon_min=0.01,
                                              hidden_layers=layers)


    # ------------------ TRAINING  ------------------ #
    for agent in agents.values():
        env_class.run_env(agent,
                          n_episodes=7500,
                          batch_size=64,
                          max_ep_steps=MAX_EPISODE_STEPS,
                          save_model=True,
                          plot=False,
                          save_plot=True,
                          note='First test')


    # # ------------------  TEST  ------------------ #
    # print('\n', '=' * 60, '\n', ' ' * 25, 'TESTING ...', '\n', '=' * 60, )
    # for agent in agents.values():
    #     agent.epsilon_max = 0.01
    #     env_class.run_env(agent,
    #                       n_episodes=10,
    #                       batch_size=64,
    #                       max_ep_steps=MAX_EPISODE_STEPS,
    #                       save_model=False,
    #                       plot=False,
    #                       save_plot=False)

    # ------------------ THE END  ------------------ #
    env_class.close()


# -----------------------------------------------------------
if __name__ == '__main__':
    main()
