import numpy as np
import json
import torch as T  # PyTorch library for ML and DL
import torch.nn.functional as F  # PyTorch's functional module
from torch.distributions import Categorical


from model import Network
from utils import *

# -----------------------------
MAX_MEMORY = 100_000

METADATA_FILE = 'data/agent_PG_metadata'


class GD_Agent:
    def __init__(self, lr, gamma, env_class, hidden_layers):
        self.agent_name = f'model'
        for hidden_layer in hidden_layers:
            self.agent_name = self.agent_name + f'_{hidden_layer}'

        self.action_space = [i for i in range(env_class.env.action_space.n)]

        print(f'\n ****** Creating Agent {self.agent_name}... ******')
        print(f'\tInput/Observation = {env_class.env.observation_space.shape} | Output/action = {env_class.env.action_space.n}')

        self.hidden_layers = hidden_layers

        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor
        self.n_games = 0

        self.PolicyPi = Network(env_class.env.observation_space.shape[0],
                                hidden_layers,
                                len(self.action_space),
                                self.lr)

        self.action_space = T.tensor(self.action_space, dtype=T.float).to(self.PolicyPi.device)

    def get_action(self, observation: np.ndarray) -> int:
        with T.no_grad():
            state = T.tensor(np.array(observation), dtype=T.float).to(self.PolicyPi.device)
            logits = self.PolicyPi.forward(state)
            categorical_dist = Categorical(logits=logits)
            return categorical_dist.sample().item() # index of the action= 0 or 1= action


    def reward_to_go(self, rews):
        n = len(rews)
        rtgs = T.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
        return rtgs

    def policy_update(self, eps_data: list) -> None:
        eps, eps_states, eps_actions, eps_rewards = zip(*eps_data)
        eps_states = T.tensor(np.array(eps_states), dtype=T.float).to(self.PolicyPi.device)
        eps_actions = T.tensor(np.array(eps_actions), dtype=T.int).to(self.PolicyPi.device)
        eps_rewards = T.tensor(np.array(eps_rewards), dtype=T.float).to(self.PolicyPi.device)

        cum_reward = self.reward_to_go(eps_rewards)

        logits = self.PolicyPi(eps_states)
        prob_distri = Categorical(logits=logits)
        log_prob = prob_distri.log_prob(eps_actions)

        batch_loss = -(log_prob * cum_reward).mean()

        self.PolicyPi.optimizer.zero_grad()
        batch_loss.backward()
        self.PolicyPi.optimizer.step()


    def store_agent_parameters(self, mean_score_500, mean_score_100, note):
        # print(f"Storing agent {self.agent_name} metadata ... ==> N_games= {self.n_games} | Mean_score={self.init_mean_score:.2f} > {mean_score}")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)

            if self.agent_name not in metadata.keys():
                metadata[self.agent_name] = {}

            metadata[self.agent_name][current_time] = {}

            metadata[self.agent_name][current_time]['n_games'] = self.n_games
            metadata[self.agent_name][current_time]['mean_score_500'] = mean_score_500
            metadata[self.agent_name][current_time]['mean_score_100'] = mean_score_100
            metadata[self.agent_name][current_time]['note'] = note

        else:
            directory = os.path.dirname(METADATA_FILE)
            if not os.path.exists(directory):
                os.makedirs(directory)

            metadata = {self.agent_name: {
                current_time: {
                    'n_games': self.n_games,
                    'mean_score_500': mean_score_500,
                    'mean_score_100': mean_score_100,
                    'note': note}
            }}

        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f'*** Model {self.agent_name} metadata saved | mean_score_500: {mean_score_500} | mean_score_100: {mean_score_100} ')


    def print_metadata(self):
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)

            print('\tModels metadata... ')
            for model, date in metadata.items():
                print(f"\t\t-->{model} : ")
                for date, data in date.items():
                    print(f"\t\t\t]{date}] : {data}")


    def save(self):
        print(f'*** Saving model {self.agent_name} parameters ...', end=" ")
        self.PolicyPi.save()


    def policy_update2(self, eps_data: list) -> None:
        # eps_data: np.ndarray with all the data of the episode
        # print(' ---------- Learning... -------')

        eps, eps_states, eps_actions, eps_rewards = zip(*eps_data)
        eps_states = T.tensor(np.array(eps_states), dtype=T.float).to(self.PolicyPi.device)
        eps_actions = T.tensor(np.array(eps_actions), dtype=T.int).to(self.PolicyPi.device)
        eps_rewards = T.tensor(np.array(eps_rewards), dtype=T.float).to(self.PolicyPi.device)

        # print('eps_states', eps_states.shape, '| eps_actions', eps_actions.shape, '| eps_rewards', eps_rewards.shape)

        ep_ret, ep_len = sum(eps_rewards), len(eps_rewards)
        # print(f"Episode {eps_rewards} | ep_ret={ep_ret} | ep_len={ep_len}")

        # # REWARDS OF EACH STEP
        cum_rewards = T.zeros_like(eps_rewards)
        reward_len = len(eps_rewards)
        for j in reversed(range(reward_len)):
            cum_rewards[j] = eps_rewards[j] + (cum_rewards[j + 1] * self.gamma if j + 1 < reward_len else 0)

        # Raw values for each action in s state
        logits = self.PolicyPi(eps_states)

        # Calculate negative log probability (-log P) as loss.
        # Cross-entropy loss is -log P in categorical distribution.
        log_probs = -F.cross_entropy(logits, eps_actions, reduction="none")
        loss = (-log_probs * cum_rewards).sum()

        # loss = T.mean(-log_probs * cum_rewards)

        self.PolicyPi.optimizer.zero_grad()
        loss.backward()
        self.PolicyPi.optimizer.step()
