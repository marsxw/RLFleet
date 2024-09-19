# %%
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.tool import soft_update, hard_update
from utils.net import QNetwork, GaussianPolicy, device
# %%

class SAC(object):
    def __init__(self, num_inputs, action_space, hidden_size, gamma, tau,  lr, alpha, target_update_interval):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size) 
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size) 
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space) 
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, replay_buffer, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = replay_buffer.sample(batch_size)

        ####Q updata #####
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        #### Policy updata ####
        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Minimum Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        #### alpha updata####
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone()  # For TensorboardX logs

        #### soft update ####
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


# qf1_loss = F.mse_loss(qf1, next_q_value) + (.1 * qf1).mean()
# qf2_loss = F.mse_loss(qf2, next_q_value) + (.1 * qf2).mean()

# q1 = torch.max(qf1, torch.pow(qf1 - next_q_value, 2) + next_q_value) + torch.max(qf1 - next_q_value, self.gamma*torch.pow(next_q_value, 2))
# q2 = torch.max(qf2, torch.pow(qf2 - next_q_value, 2) + next_q_value) + torch.max(qf2 - next_q_value, self.gamma*torch.pow(next_q_value, 2))
# qf1_loss = q1.mean()
# qf2_loss = q2.mean()
