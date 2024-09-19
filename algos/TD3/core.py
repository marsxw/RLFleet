# %%
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.tool import soft_update, hard_update
from utils.net import QNetwork, DeterministicPolicy,device



class TD3(object):
    def __init__(self, num_inputs, action_space, hidden_size, gamma, tau,  lr,  target_update_interval, sigma=0.1, policy_noise_sigma=0.2):
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.delay_freq = 2
        self.gamma = gamma

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size)
        hard_update(self.critic_target, self.critic)

        self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.policy_target = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space)
        hard_update(self.policy_target, self.policy)

        self.max_action = self.policy.action_scale

        self.expl_noise = sigma * self.max_action
        self.policy_noise = policy_noise_sigma*self.max_action
        self.noise_clip = 0.5*self.max_action

    def select_action(self, state, evaluate=False):

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.policy(state)

        if evaluate is False:
            action = action +  torch.randn_like(action) * self.expl_noise
            action = action.clamp(-self.max_action, self.max_action)

        return action.detach().cpu().numpy()[0]

    def update_parameters(self, replay_buffer, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = replay_buffer.sample(batch_size)
        ####Q updata #####
        with torch.no_grad():
            next_state_action = self.policy(next_state_batch)
            noise = torch.randn_like(next_state_action) * self.policy_noise
            smoothed_target_a = next_state_action + noise.clamp(-self.noise_clip, self.noise_clip)
            smoothed_target_a = smoothed_target_a.clamp(-self.max_action, self.max_action)

            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, smoothed_target_a)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        #### Policy updata ####
        if updates % self.delay_freq == 0:
            qf1_pi = self.critic.Q1(state_batch, self.policy(state_batch))
            policy_loss = (- qf1_pi).mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            if updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)
                soft_update(self.policy_target, self.policy, self.tau)

            return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), qf1.mean().item(), qf2.mean().item()
        else:
            return qf1_loss.item(), qf2_loss.item(), 0, qf1.mean().item(), qf2.mean().item()
