import torch
import torch.optim as optim
import numpy as np
from model import PolicyNet, QValueNet
from replay_buffer import ReplayBuffer

class DDPG:
    def __init__(self, n_states, n_hiddens, n_actions, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device, buffer_size):
        self.actor = PolicyNet(n_states, n_hiddens, n_actions, action_bound).to(device)
        self.critic = QValueNet(n_states, n_hiddens, n_actions).to(device)
        self.target_actor = PolicyNet(n_states, n_hiddens, n_actions, action_bound).to(device)
        self.target_critic = QValueNet(n_states, n_hiddens, n_actions).to(device)

        # 目标网络初始化
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 经验回放池
        self.memory = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau
        self.device = device

        # 初始化损失记录列表
        self.actor_losses = []  # 用于记录 Actor 损失
        self.critic_losses = []  # 用于记录 Critic 损失

    def update(self, batch_size):

        # 如果经验池中的样本不足，跳过更新
        if len(self.memory) < batch_size:
            print(f"Not enough samples in memory. Current memory size: {len(self.memory)}")
            return  # 跳过此次更新

        # 从经验池中采样
        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))

        states = torch.tensor(np.array(batch[0]), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(batch[1]), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(batch[2]), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(batch[4]), dtype=torch.float).view(-1, 1).to(self.device)

        # 计算 Q 值目标
        next_actions = self.target_actor(next_states)
        next_q_values = self.target_critic(next_states, next_actions)
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        # 更新 Critic 网络
        q_values = self.critic(states, actions)
        critic_loss = torch.mean((q_values - q_targets) ** 2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新 Actor 网络
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 记录损失
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())

        # 软更新目标网络
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

    def soft_update(self, source, target):
        """软更新目标网络的参数"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state):
        """通过策略网络选择动作，加入噪声进行探索"""
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor).cpu().detach().numpy()[0]
        action += self.sigma * np.random.randn(*action.shape)  # 加入探索噪声
        return np.clip(action, -1, 1)

