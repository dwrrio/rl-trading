import pandas as pd
import numpy as np

from env.trading_env import TradingEnv
from agents.dqn import DQNAgent
from data_loader import load_multi_stock_data

df = load_multi_stock_data()
env = TradingEnv(df)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


agent = DQNAgent(
    state_size,
    action_size
)


episodes = 50

for episode in range(episodes):

    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:

        action = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        agent.replay()
        state = next_state
        total_reward += reward

    print(f"Episode: {episode + 1}")
    print(f"Reward: {total_reward:.2f}")
    print(f"Portfolio Value: {info['portfolio_value']:.2f}")
    print(f"Epsilon: {agent.epsilon:.4f}")
