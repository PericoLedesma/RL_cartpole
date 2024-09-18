import os
import torch as T
import torch.nn as nn


class CriticNetwork(nn.Module):
    def __init__(self, name, input_size, hidden_layers, output_size, lr):
        super(CriticNetwork, self).__init__()
        print(f'\t*Creating {name} Network... {input_size}x{hidden_layers}x{output_size}')
        self.type = name  # Actor or Critic

        layers = []
        if len(hidden_layers) == 1:
            layers.append(nn.Sequential(nn.Linear(input_size, hidden_layers[0]),
                                        nn.ReLU()))
            layers.append(nn.Sequential(nn.Linear(hidden_layers[0], output_size)))
        else:
            layers.append(nn.Sequential(nn.Linear(input_size, hidden_layers[0]),
                                        nn.ReLU()))
            layers.append(nn.Sequential(nn.Linear(hidden_layers[0], hidden_layers[1]),
                                        nn.ReLU()))
            layers.append(nn.Sequential(nn.Linear(hidden_layers[1], output_size)))  # logits

        self.layers = nn.ModuleList(layers)
        self.loss = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state) -> T.tensor:  # logits
        input_weights = state
        for layer in self.layers[:-1]:
            input_weights = layer(input_weights)

        return self.layers[-1](input_weights)  # logits


class ActorNetwork(nn.Module):
    def __init__(self, name, input_size, hidden_layers, output_size, lr):
        super(ActorNetwork, self).__init__()
        print(f'\t*Creating {name} Network... {input_size}x{hidden_layers}x{output_size}')
        self.type = name

        layers = []
        if len(hidden_layers) == 1:
            layers.append(nn.Sequential(nn.Linear(input_size, hidden_layers[0]),
                                        nn.ReLU()))
            layers.append(nn.Sequential(nn.Linear(hidden_layers[0], output_size)))
        else:
            layers.append(nn.Sequential(nn.Linear(input_size, hidden_layers[0]),
                                        nn.ReLU()))
            layers.append(nn.Sequential(nn.Linear(hidden_layers[0], hidden_layers[1]),
                                        nn.ReLU()))
            layers.append(nn.Sequential(nn.Linear(hidden_layers[1], output_size),
                                        nn.Tanh()))  # logits

        self.layers = nn.ModuleList(layers)
        self.loss = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state) -> T.tensor:  # logits
        input_weights = state
        for layer in self.layers[:-1]:
            input_weights = layer(input_weights)

        return self.layers[-1](input_weights)  # logits
