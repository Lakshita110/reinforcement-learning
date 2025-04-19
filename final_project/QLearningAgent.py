import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from tqdm import trange


class DQNAgent:
    def __init__(
        self,
        env,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_size=10000,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        target_update_freq=10,
        device="cpu"
    ):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.q_net = QNetwork(env.action_space.n).to(device)
        self.target_net = QNetwork(env.action_space.n).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.steps = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.env.get_available_actions())
        else:
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_tensor).detach().cpu().numpy()[0]
            valid_actions = self.env.get_available_actions()
            return max(valid_actions, key=lambda a: q_values[a])

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.tensor(
            np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.tensor(
            np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        x = x.unsqueeze(1).float()  # (B, 1, 6, 7)
        return self.net(x)


# Let DQNAgent be exportable
__all__ = ["DQNAgent"]
