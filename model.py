import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """

        """
        super(Actor, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.model = nn.Sequential(
            nn.BatchNorm1d(state_size),
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.BatchNorm1d(fc1_units),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.BatchNorm1d(fc2_units),
            nn.Linear(fc2_units, action_size),
            nn.Tanh()
        )

        self.model.apply(self.init_weights)

        print("Initilized Actor with state size {} and action size {}".format(
            state_size, action_size))

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.1)

    def forward(self, state):
        """
        Build an actor (policy) network that maps states -> actions
        """
        return self.model(state)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """
        """
        super(Critic, self).__init__()

        self.model_input = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.BatchNorm1d(fc1_units),
        )

        self.model_output = nn.Sequential(
            nn.Linear(fc1_units + action_size, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1)
        )

        self.model_input.apply(self.init_weights)
        self.model_output.apply(self.init_weights)

        print("Initilized Critic with state size {} and action size {}".format(
            state_size, action_size))

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat([self.model_input(state), action], dim=1)
        return self.model_output(x)