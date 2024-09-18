from agent import GD_Agent
from environment import EnvironmentClass

# -----------------------------
'''
Notes: 
action_space = Discrete(2)  # Example: action = 0 (left), action = 1 (right)
Reward = steps, binary, where the agent receives +1 for every step it balances the pole.
'''


MAX_EPISODE_STEPS = 500  # Default should be 500


def main():
    # ------------------ ENVIRONMENT  ------------------ #
    env_class = EnvironmentClass(env_id='CartPole-v1',
                                 render_mode='rgb_array',  # 'human ' or 'rgb_array'
                                 max_ep_steps=MAX_EPISODE_STEPS)

    # ------------------ AGENTS  ------------------ #
    agents = {}
    for hidden_layers in [[16],[32],[64], [128]]: # , [128, 128], [256]
        agents[f"model_{hidden_layers}"] = GD_Agent(env_class=env_class,
                                                    lr=0.001,
                                                    gamma=0.99,
                                                    hidden_layers=hidden_layers)

    # ------------------ TRAINING  ------------------ #
    for agent in agents.values():
        env_class.run_env(agent,
                          n_episodes=1000,
                          max_ep_steps=MAX_EPISODE_STEPS,
                          batch_size=8,
                          plot_eps_inf_every=10,
                          plot=False,
                          save_plot=True,
                          note='First test')

    # ------------------ THE END  ------------------ #
    env_class.close()


# -----------------------------------------------------------
if __name__ == '__main__':
    main()
