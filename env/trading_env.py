import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pandas as pd


class TradingEnv(gym.Env):

    def __init__(self, df, initial_cash=10000):

        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        
        """
        0 = HOLD
        1 = BUY
        2 = SELL
        """
        self.action_space = spaces.Discrete(3)

        """
        Observation:

        [
            Open,
            High,
            Low,
            Close,
            Volume,
            Cash,
            Shares Held
        ]
        """
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.portfolio_value = self.initial_cash
        observation = self._get_observation()
        info = {}

        return observation, info

    def _get_observation(self):

        row = self.df.loc[self.current_step]

        observation = np.array([
            row["Open"],
            row["High"],
            row["Low"],
            row["Close"],
            row["Volume"],
            self.cash,
            self.shares_held
        ], dtype=np.float32)

        return observation

    def step(self, action):

        current_price = self.df.loc[self.current_step, "Close"]
        previous_portfolio_value = self.portfolio_value

        if action == 0:
            pass
        
        elif action == 1:
            if self.cash >= current_price:
                self.cash -= current_price
                self.shares_held += 1

        elif action == 2:
            if self.shares_held > 0:
                self.cash += current_price
                self.shares_held -= 1

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        next_price = self.df.loc[self.current_step, "Close"]

        self.portfolio_value = (
            self.cash +
            (self.shares_held * next_price)
        )

        reward = (
            self.portfolio_value -
            previous_portfolio_value
        )

        observation = self._get_observation()

        info = {
            "cash": self.cash,
            "shares_held": self.shares_held,
            "portfolio_value": self.portfolio_value
        }

        return observation, reward, done, False, info
