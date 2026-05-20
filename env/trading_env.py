import gymnasium as gym

from gymnasium import spaces

import numpy as np


class TradingEnv(gym.Env):

    def __init__(
        self,
        df,
        initial_cash=10000
    ):

        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash

        # ACTION SPACE

        # 0 = HOLD
        # 1 = BUY AAPL
        # 2 = SELL AAPL
        # 3 = BUY NVDA
        # 4 = SELL NVDA
        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.positions = {
            "AAPL": 0,
            "NVDA": 0
        }

        self.portfolio_value = self.initial_cash
        observation = self._get_observation()
        info = {}

        return observation, info

    def _get_observation(self):

        row = self.df.iloc[self.current_step]

        observation = np.array([
            row["AAPL_Open"],
            row["AAPL_High"],
            row["AAPL_Low"],
            row["AAPL_Close"],
            row["AAPL_Volume"],
            row["NVDA_Open"],
            row["NVDA_High"],
            row["NVDA_Low"],
            row["NVDA_Close"],
            row["NVDA_Volume"],
            self.cash,
            self.positions["AAPL"],
            self.positions["NVDA"]

        ], dtype=np.float32)

        return observation

    def step(self, action):

        row = self.df.iloc[
            self.current_step
        ]

        previous_portfolio_value = self.portfolio_value

        aapl_price = row["AAPL_Close"]
        nvda_price = row["NVDA_Close"]

        # HOLD
        if action == 0:
            pass

        # BUY AAPL
        elif action == 1:
            if self.cash >= aapl_price:
                self.cash -= aapl_price
                self.positions["AAPL"] += 1

        # SELL AAPL
        elif action == 2:
            if self.positions["AAPL"] > 0:
                self.cash += aapl_price
                self.positions["AAPL"] -= 1

        # BUY NVDA
        elif action == 3:
            if self.cash >= nvda_price:
                self.cash -= nvda_price
                self.positions["NVDA"] += 1

        # SELL NVDA
        elif action == 4:
            if self.positions["NVDA"] > 0:
                self.cash += nvda_price
                self.positions["NVDA"] -= 1

        self.current_step += 1
        done = self.current_step >= (len(self.df) - 1)
        next_row = self.df.iloc[self.current_step]

        self.portfolio_value = (
            self.cash
            + (self.positions["AAPL"] * next_row["AAPL_Close"])
            + (self.positions["NVDA"] * next_row["NVDA_Close"])
        )

        reward = self.portfolio_value - previous_portfolio_value
        observation = self._get_observation()

        info = {
            "cash": self.cash,
            "portfolio_value":
                self.portfolio_value,
            "aapl_shares":
                self.positions["AAPL"],
            "nvda_shares":
                self.positions["NVDA"]
        }

        return (
            observation,
            reward,
            done,
            False,
            info
        )
