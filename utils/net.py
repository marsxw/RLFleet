# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init_(m):
    '''Initialize Policy weights'''
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class SingleQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)
        self.to(device)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(sa))
        x1 = F.relu(self.linear2(x1))
        return self.linear3(x1)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        # Q1
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # Q2
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)
        self.to(device)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(sa))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(sa))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(sa))
        x1 = F.relu(self.linear2(x1))
        return self.linear3(x1)

    def Q2(self, state, action):
        sa = torch.cat([state, action], 1)
        x2 = F.relu(self.linear4(sa))
        x2 = F.relu(self.linear5(x2))
        return self.linear6(x2)


class ActionNorml():
    def __init__(self, action_space=None):
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)


class DeterministicPolicy(nn.Module, ActionNorml):
    '''TD3'''

    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, Normalized=False):
        nn.Module.__init__(self)
        ActionNorml.__init__(self, action_space)

        self.aa = torch.tensor(0.)

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)
        self.Normalized = Normalized

        self.to(device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean(x)

        if self.Normalized:
            '''TD3+'''
            K = mean.size()[1]  # action dims
            G = torch.norm(mean, p=1, dim=1).view(-1, 1)  # l1 norm
            G = G/K
            ones = torch.ones(G.size()).to(G.device)
            mean = mean/torch.where(G >= 1, G, ones)

        return torch.tanh(mean) * self.action_scale + self.action_bias


class GaussianPolicy(nn.Module, ActionNorml):
    '''SAC'''

    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        nn.Module.__init__(self)
        ActionNorml.__init__(self, action_space)

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_aciton = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_aciton

 