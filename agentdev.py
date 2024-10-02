import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a flexible DNN architecture (this will act as a policy network)
class FlexibleDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers=2):
        super(FlexibleDNN, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.input_size = input_size
        self.output_size = output_size

        # Define hidden layers and output layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        for _ in range(1, layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return torch.softmax(self.output_layer(x), dim=-1)

# Lower-level DNN (EEG signal processing)
class LowerDNN(FlexibleDNN):
    def __init__(self, input_size, hidden_size, output_size):
        super(LowerDNN, self).__init__(input_size, hidden_size, output_size)

    def forward(self, x):
        return super().forward(x)

# Upper-level DNN as a Reinforcement Learning agent
class UpperRLHF(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pre_trained_models):
        super(UpperRLHF, self).__init__()
        self.policy_network = FlexibleDNN(input_size, hidden_size, output_size)
        self.pre_trained_models = pre_trained_models

    def forward(self, x):
        # Generate action probabilities (which pretrained models to trigger)
        action_probs = self.policy_network(x)
        return action_probs

    def select_action(self, x):
        # Sample an action based on the action probabilities
        action_probs = self.forward(x)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        return action, action_distribution.log_prob(action)

    def trigger_pretrained_model(self, action, latent_representation):
        # Use the chosen action to select a pretrained model
        return self.pre_trained_models[action](latent_representation)

# Human feedback (reward) is integrated via reinforcement learning updates
class RLHFTrainer:
    def __init__(self, model, optimizer, gamma=0.99):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.rewards = []
        self.log_probs = []

    def store_outcome(self, log_prob, reward):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def compute_returns(self):
        # Compute the discounted cumulative rewards (returns)
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        return (returns - returns.mean()) / (returns.std)
