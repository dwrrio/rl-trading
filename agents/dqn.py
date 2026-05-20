import random
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):

    def __init__(self, state_size, action_size):

        super(DQN, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, action_size)

        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=10000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = DQN(
            state_size,
            action_size
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        self.criterion = nn.MSELoss()

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        q_values = self.model(state)
        action = torch.argmax(q_values).item()

        return action

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(
            self.memory,
            self.batch_size
        )

        for state, action, reward, next_state, done in batch:

            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.tensor(
                reward,
                dtype=torch.float32
            ).to(self.device)

            target = reward

            if not done:
                next_q = torch.max(
                    self.model(next_state)
                )

                target = reward + (
                    self.gamma * next_q
                )

            current_q = self.model(state)[action]

            loss = self.criterion(
                current_q,
                target
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
