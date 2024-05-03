import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim


class PolicyNetwork(nn.Module):
    """
    The reinforce class. Used for deciding on what action to take.
    The methods perform the following tasks:
    'init' creates the architecture,
    'forwards' passes a state forward, recieving an action,
    'act' calls forward and calculates the action log prob and entropy.
    """

    def __init__(self, in_dims, out_dims, learning_rate, hidden_size=64):
        super().__init__()
        # create network architecture
        self.fc1 = nn.Linear(in_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_dims)
        # initialize weights from a uniform distribution
        nn.init.uniform_(self.fc1.weight)
        nn.init.uniform_(self.fc2.weight)
        # set optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state).cpu()
        model = Categorical(probs)
        action = model.sample()

        return action.item(), model.log_prob(action), model.entropy()

    def train(self, saved_log_probs, returns, entropies, eta_entropy):
        # sum^{T-1}_{t=0} R_t * log \pi_\theta
        policy_loss = []

        for log_prob, R, entropy in zip(saved_log_probs, returns, entropies):
            # vanilla policy loss
            loss = -log_prob * R
            # entropy regularization
            loss -= entropy * eta_entropy
            policy_loss.append(loss)
        policy_loss = torch.stack(policy_loss)
        policy_loss = torch.sum(policy_loss, dim=0)

        # apply gradients
        self.optimizer.zero_grad()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
        policy_loss.backward()
        self.optimizer.step()
