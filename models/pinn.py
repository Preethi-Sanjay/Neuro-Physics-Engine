import torch.nn as nn


class PINN(nn.Module):

    def __init__(self,
                 input_dim=2,
                 hidden_dim=64,
                 num_hidden_layers=4,
                 output_dim=1):

        super().__init__()

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)