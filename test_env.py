import time
import pandas as pd

from env.trading_env import TradingEnv


df = pd.read_csv("data/apple.csv")

env = TradingEnv(df)
obs, info = env.reset()
done = False
total_reward = 0

while not done:

    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

    print("Observation:", obs)
    print("Reward:", reward)
    print("Info:", info)
    print("-" * 50)
    time.sleep(8)

print("Final Portfolio Value:", info["portfolio_value"])
print("Total Reward:", total_reward)
