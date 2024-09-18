from agent import AC_Agent
from environment import EnvironmentClass

# -----------------------------
'''
Notes: 
action_space = Discrete(2)  # Example: action = 0 (left), action = 1 (right)
Reward = steps, binary, where the agent receives +1 for every step it balances the pole.

To change thresold: **/anaconda3/envs/general/lib/python3.11/site-packages/gym/envs/classic_control/cartpole.py


'''

# TODO ANGLE THERSHOLD

MAX_EPISODE_STEPS = 500  # Default should be 500


def main():
    # ------------------ ENVIRONMENT  ------------------ #
    env_class = EnvironmentClass(env_id='CartPole-v1',
                                 render_mode='rgb_array',  # 'human ' or 'rgb_arraY'
                                 max_ep_steps=MAX_EPISODE_STEPS)

    # ------------------ AGENTS  ------------------ #
    agents = {}
    for layers in [[64], [128],[256],[64,128]]:  # , [128, 128], [256]
        for replay in [True, False]:
            key_model = f"model_{layers}_replay{replay}"
            agents[key_model] = AC_Agent(env_class=env_class,
                                         lr=0.001,
                                         gamma=0.99,
                                         hidden_layers=layers,
                                         replay=replay)

    # ------------------ TRAINING  ------------------ #
    for agent in agents.values():
        env_class.run_env(agent,
                          n_episodes=400,
                          batch_size=8,
                          max_ep_steps=MAX_EPISODE_STEPS,
                          plot_eps_inf_every=10,
                          plot=False,
                          save_plot=True,
                          note='Whats new?')

    # ------------------ THE END  ------------------ #
    env_class.close()


# -----------------------------------------------------------
if __name__ == '__main__':
    main()
