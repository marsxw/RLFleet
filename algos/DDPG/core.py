import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.tool import soft_update, hard_update
from utils.net import QNetwork, DeterministicPolicy, device


class DDPG(object):
    def __init__(self, num_inputs, action_space, hidden_size, gamma, tau, lr, sigma=0.1):
        self.tau = tau
        self.gamma = gamma
        self.max_action = action_space.high[0]

        # Critic网络，只需要一个
        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        # 目标Critic网络
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size)
        hard_update(self.critic_target, self.critic)

        # 策略网络
        self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        # 目标策略网络
        self.policy_target = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space)
        hard_update(self.policy_target, self.policy)

        # 探索噪声
        self.expl_noise = sigma * self.max_action

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.policy(state)

        if not evaluate:
            action += torch.randn_like(action) * self.expl_noise
            action = action.clamp(-self.max_action, self.max_action)

        return action.detach().cpu().numpy()[0]

    def update_parameters(self, replay_buffer, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = replay_buffer.sample(batch_size)

        #### Critic更新 ####
        with torch.no_grad():
            next_action = self.policy_target(next_state_batch)
            next_q_value = reward_batch + mask_batch * self.gamma * self.critic_target(next_state_batch, next_action)

        q_value = self.critic(state_batch, action_batch)
        q_loss = F.mse_loss(q_value, next_q_value)

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        #### 策略更新 ####
        policy_loss = -self.critic.Q1(state_batch, self.policy(state_batch)).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.policy_target, self.policy, self.tau)

        return q_loss.item(), policy_loss.item(), q_value.mean().item()
