# ddpg.py
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from logging import DEBUG, INFO
from ml.utils.logger import log
class Actor(nn.Module):
    """生成客户端权重分布的Actor网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  #action dim  是客户端数量
            nn.Softmax(dim=-1)  # 归一化
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """评估状态动作价值的Critic网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU()
        )
        self.output_net = nn.Linear(hidden_dim * 2, 1)

    def forward(self, state, action):
        """评估状态动作价值"""
        # 处理状态特征
        state_features = self.state_net(state)  # 输出形状: [batch_size, hidden_dim]
        action_features = self.action_net(action)  # 输出形状: [batch_size, hidden_dim]

        # # 打印调试信息
        # log(INFO, f"State features shape: {state_features.shape}")
        # log(INFO, f"Action features shape: {action_features.shape}")

        # 拼接状态和动作特征
        combined_features = torch.cat([state_features, action_features], dim=1)  # 拼接后形状: [batch_size, hidden_dim * 2]

        # 打印调试信息
        # log(INFO, f"Combined features shape: {combined_features.shape}")

        # 输出最终的 Q 值
        return self.output_net(combined_features)  # 输出形状: [batch_size, 1]

class DDPG:
    """DDPG算法实现"""
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = deque(maxlen=1000)
        self.device = device
        self.tau = 0.005
        self.gamma = 0.99

    def get_action(self, state, exploration_noise=0.1):
        """生成带探索噪声的动作"""
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state)  # 输入形状: (state_dim,)
        if np.random.rand() < exploration_noise:
            action = torch.softmax(action + torch.randn_like(action)*0.1, dim=-1)
        return action.detach().cpu().numpy()   # 输出形状: (action_dim,)  输出的是一个维度为客户端数量，然后加起来为1 的权重数组

    def update(self, batch_size=5):
        """更新网络参数"""
        if len(self.replay_buffer) < batch_size:
            return

        # 从经验池采样
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)  # 原始形状: [batch_size, state_dim]
        actions = torch.FloatTensor(np.array(actions)).to(self.device)  # 原始形状: [batch_size, action_dim]
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # 形状: [batch_size, 1]
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)  # 原始形状: [batch_size, state_dim]

        # log(INFO, f"States shape: {states.shape}")
        # log(INFO, f"Actions shape: {actions.shape}")
        # log(INFO, f"Next states shape: {next_states.shape}")
        # log(INFO, f"Rewards shape: {rewards.shape}")

        # 更新Critic
        target_actions = self.target_actor(next_states)  # 形状: [batch_size, action_dim]
        target_q = self.target_critic(next_states, target_actions)  # 输入形状: [batch_size, state_dim], [batch_size, action_dim]
        expected_q = rewards + self.gamma * target_q

        current_q = self.critic(states, actions)  # 输入形状: [batch_size, state_dim], [batch_size, action_dim]
        critic_loss = nn.MSELoss()(current_q, expected_q.detach())

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # 更新Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # 软更新目标网络
        for t_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            t_param.data.copy_(self.tau * param.data + (1 - self.tau) * t_param.data)
        for t_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            t_param.data.copy_(self.tau * param.data + (1 - self.tau) * t_param.data)


    def save_experience(self, state, action, reward, next_state):
        """存储经验"""
        state = np.array(state)  # 将维度从 [batch_size, num_clients, state_dim] 转换为 [batch_size, state_dim]
        next_state = np.array(next_state)  # 确保 next_state 的维度为 [batch_size, state_dim]
        self.replay_buffer.append((state, action, reward, next_state))