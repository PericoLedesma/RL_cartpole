import numpy as np
import random
import json
from collections import deque
import torch as T  # PyTorch library for ML and DL
import torch.nn.functional as F  # PyTorch's functional module
from torch.distributions import Categorical

from model import *
from utils import *

# -----------------------------
MAX_MEMORY = 100_000
METADATA_FILE = 'data/agent_AC_metadata'


class AC_Agent:
    def __init__(self, lr, gamma, env_class, hidden_layers, replay):
        # MODEL NAME
        self.agent_name = f'model'
        for hidden_layer in hidden_layers:
            self.agent_name = self.agent_name + f'_{hidden_layer}'
        self.agent_name += f'_replay{replay}'

        self.action_space = [i for i in range(env_class.env.action_space.n)]

        print(f'\n ****** Creating Agent {self.agent_name}... ******')
        print(f'\tInput/Observation = {env_class.env.observation_space.shape} | Output/action = {env_class.env.action_space.n}')

        self.hidden_layers = hidden_layers

        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor

        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.replay = replay

        self.n_games = 0
        self.gpu_use()

        # Actor network (Policy)
        self.ActorNet = ActorNetwork('Actor',
                                     env_class.env.observation_space.shape[0],
                                     hidden_layers,
                                     len(self.action_space),
                                     self.lr).to(self.device)

        self.ActorNet_target = ActorNetwork('Actor_target',
                                            env_class.env.observation_space.shape[0],
                                            hidden_layers,
                                            len(self.action_space),
                                            self.lr).to(self.device)

        self.ActorNet_target.load_state_dict(self.ActorNet.state_dict())

        # Critic network (Value)
        self.CriticNet = CriticNetwork('Critic',
                                       env_class.env.observation_space.shape[0],
                                       hidden_layers,
                                       1,
                                       self.lr).to(self.device)
        self.CriticNet_target = CriticNetwork('Critic_target',
                                              env_class.env.observation_space.shape[0],
                                              hidden_layers,
                                              1,
                                              self.lr).to(self.device)
        self.CriticNet_target.load_state_dict(self.CriticNet.state_dict())

        self.action_space = T.tensor(self.action_space, dtype=T.float).to(self.device)

    def get_action(self, observation: np.ndarray) -> int:
        with T.no_grad():
            state = T.tensor(np.array(observation), dtype=T.float).to(self.device)
            logits = self.ActorNet.forward(state)
            categorical_dist = Categorical(logits=logits)
            return categorical_dist.sample().item()  # return index of the action

    def memory_replay(self, batch_size):
        # print(f' ---------- Memory Replay... batch size={batch_size} -------')
        # memory array: [(eps, eps_data), (eps, eps_data), ...]
        batches: list = random.sample(self.memory, batch_size)  # list of tuples
        eps, eps_data = zip(*batches)  # : tuple of lists

        for count, data in enumerate(eps_data):
            print(f'Memory replay {count}', end='\r')
            self.policy_update(data)
        print(f'Memory replay batch {batch_size}.')

    def policy_update(self, eps_data: list) -> None:
        # eps_data: np.ndarray with all the data of the episode
        # print(' ---------- Learning... -------')

        eps, eps_states, eps_actions, eps_rewards = zip(*eps_data)
        eps_states = T.tensor(np.array(eps_states), dtype=T.float).to(self.device)
        eps_actions = T.tensor(np.array(eps_actions), dtype=T.int).to(self.device)
        eps_rewards = T.tensor(np.array(eps_rewards), dtype=T.float).to(self.device)

        # print('eps_states', eps_states.shape, '| eps_actions', eps_actions.shape, '| eps_rewards', eps_rewards.shape)

        # # REWARDS OF EACH STEP
        cum_rewards = T.zeros_like(eps_rewards)
        reward_len = len(eps_rewards)
        for j in reversed(range(reward_len)):
            cum_rewards[j] = eps_rewards[j] + (cum_rewards[j + 1] * self.gamma if j + 1 < reward_len else 0)

        # CRITIC - Optimize value loss (Critic)
        self.CriticNet.optimizer.zero_grad()
        # cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)

        values = self.CriticNet(eps_states)
        values = values.squeeze(dim=1)

        vf_loss = F.mse_loss(values, cum_rewards, reduction="none")
        vf_loss.sum().backward()
        self.CriticNet.optimizer.step()

        # ACTOR - Optimize policy loss (Actor)
        with T.no_grad():
            values = self.CriticNet(eps_states)
        self.ActorNet.optimizer.zero_grad()
        advantages = cum_rewards - values
        logits = self.ActorNet(eps_states)
        log_probs = -F.cross_entropy(logits, eps_actions, reduction="none")
        pi_loss = -log_probs * advantages
        pi_loss.sum().backward()
        self.ActorNet.optimizer.step()

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
                    'note': 'NOTE'}
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
        else:
            print(f"No metadata of the agent found.")

    def gpu_use(self):
        if not T.backends.mps.is_available():
            print("\tCHECK: CPU training")
            self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else:
            self.device = T.device("mps")


    # def save(self):
    #     print(f'*** Saving Models {self.agent_name} parameters ...')
    #     self.ActorNet.save(self.replay)
    #     self.CriticNet.save(self.replay)

