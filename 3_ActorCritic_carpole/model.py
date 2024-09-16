import os
import torch as T
import torch.nn as nn

MODEL_FILE = 'AC_cartpole_weights'


class Network(nn.Module):
    def __init__(self, name, input_size, hidden_layers, output_size, lr, replay):
        """
        Args:
            input_size (int): The size of the input features.
            hidden_layers (list of int): A list where each element represents the number of units in a hidden layer.
            output_size (int): The size of the output layer.
        """
        super(Network, self).__init__()
        print(f'\t*Creating {name} Network... {input_size}x{hidden_layers}x{output_size}')

        self.type = name # Actor or Critic

        if len(hidden_layers) == 1:
            self.hidden_layers = str(hidden_layers[0])
        else:
            self.hidden_layers = "_".join(map(str, hidden_layers))

        layers = []

        layers.append(nn.Linear(input_size, hidden_layers[0]))
        if len(hidden_layers) > 1:
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        # Register all layers using nn.ModuleList
        self.layers = nn.ModuleList(layers)

        self.loss = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)

        self.load_model(replay)

    def forward(self, state) -> T.tensor:  # logits
        input_weights = state
        for layer in self.layers[:-1]:
            input_weights = T.relu(layer(input_weights))

        return self.layers[-1](input_weights)  # logits

    def save(self, replay):
        file = f"{MODEL_FILE}_{self.type}_{self.hidden_layers}_replay_{replay}.pth"
        if not os.path.exists('model'):
            os.makedirs('model')
        filepath = os.path.join('model', file)
        T.save(self.state_dict(), filepath)
        print(f'\tModel {self.type} saved as {file} in model folder. ')

    def load_model(self, replay):
        file = f"{MODEL_FILE}_{self.type}_{self.hidden_layers}_replay_{replay}.pth"
        print(f'\t\tLoading agent model... searching for {file} ', end=" ")
        filepath = os.path.join('model', file)
        if os.path.exists(filepath):
            self.load_state_dict(T.load(filepath))
            print("Weights loaded successfully!.")
        else:
            print(f"No model saved, the weights file does not exist. No weights loaded.")
