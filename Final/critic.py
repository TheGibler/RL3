import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np


# actor is a taken from reinforce.py

class Critic(nn.Module):
    """
    The critic class of the actor critic. Used for estimating a value function to judge the actor.
    The methods perform the following tasks:
    'init' creates the architecture,
    'forwards' passes a state forward, receiving an evaluation,
    'squared loss' calculates the loss of a given episode, given the calculated cumulative rewards of each state.
    """

    def __init__(self, input_dim, learning_rate, hidden_nodes=64):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, 1)
        )
        self.optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def forward(self, state):
        value = self.critic(state)
        return value

    def squared_loss(self, states, Q_sa):
        V = self.forward(states)
        critic_loss = torch.nn.functional.mse_loss(V.float(), Q_sa.float())
        return critic_loss

    def train(self, states, Q_sa):
        critic_loss = self.squared_loss(torch.from_numpy(np.array(states)), Q_sa)

        # apply gradients
        self.optimizer.zero_grad()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
        critic_loss.backward()
        self.optimizer.step()
