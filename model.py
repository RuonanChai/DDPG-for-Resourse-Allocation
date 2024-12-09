import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions, action_bound):
        super(PolicyNet, self).__init__()
        self.action_bound = action_bound
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.tanh(x) * self.action_bound  # [-action_bound, action_bound]
        return x

class QValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states + n_actions, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x