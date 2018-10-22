import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import pandas as pd

def load_data(tickers, granularity):
    """
    Loads the low, high, and close prices of all the tickers from csv files
    """
    dfs = []

    for ticker in tickers:
        filename = '/Users/Simon/code/crypto/coinbase_data/data/' + ticker + '_' + str(granularity) +'.csv'
        df = pd.read_csv(
            filename,
            usecols=['time','low','high','close'],
            parse_dates=['time'],
            index_col='time',
        )

        # rename columns to include ticker name
        df.columns = [ticker[:3] + "_" + col for col in df.columns]
        dfs.append(df)

    data = pd.concat([df for df in dfs], axis=1, join='outer')
    
    # Remember, we filled NaNs forward, this is especially relevant for LTC
    data.fillna(method='ffill', axis=0, inplace=True)
    
    # Our cash (USD) is simply equal to 1, because all cryptos are quoted in USD
    cash = np.array([1 for i in range(3*len(data))]).reshape(-1,3)
    cash = pd.DataFrame(data=cash, index=data.index, columns=['cash_low', 'cash_high', 'cash_close'])

    assert len(data) == len(cash)
    data = pd.concat([cash, data], axis=1)

    return data


class PortfolioTradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    """
    Defines our trading environment in which the agent acts via determining
    the portfolio weights it wants to hold at each timestep 

    It can be created via env = gym.make('crypto-trading-v0')
    """
    def __init__(self):
        
        # This is what coinbase pro currently charges as a taker-fee
        self.COMMISSION_RATE = 0.003

        self.granularity = 900  #900 seconds = 15 minutes
        tickers = ['BTC-USD', 'ETH-USD', 'LTC-USD']

        # Loading the low, high, close prices
        self.data = load_data(tickers, self.granularity)

        # We add one because we can also hold cash
        self.num_assets = len(tickers) + 1

        # One episode spans 30 days 
        self.timesteps_per_episode = 30*24*60*60 / self.granularity 

        # Action space
        # Agent picks an action that corresponds to the the portfolio weights of each asset
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)

        # Observation space (state)
        low_state = np.zeros((self.num_assets,3)) 
        # We have 4 columns. 3 for the features (low, high, close) and 1 for the previous weights
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_assets, 4), dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        # The initial total value of our portfolio
        self.portfolio_value = 1.

        self.timestep = 0

        # At time 0, we hold all our money in cash only
        initial_weights = np.zeros(self.num_assets)
        initial_weights[0] = 1.

        # We start every episode at a random point in time of our data 
        self.position = np.random.randint(low=0,high=len(self.data)-self.timesteps_per_episode-1) 
        initial_prices = self.data.iloc[self.position].values
        initial_state = initial_prices.reshape(self.num_assets,3)

        # Normalize with regard to the closing price
        self.initial_closing_prices = np.expand_dims(initial_state[:,-1], axis=1) 
        self.state = initial_state / self.initial_closing_prices 

        # Add the weights to the state and return it 
        initial_weights = np.expand_dims(initial_weights, axis=1)
        self.state = np.concatenate((self.state, initial_weights), axis=1)

        return self.state

    def step(self, action):
        # Check if we feed a correct action 
        assert action.shape == self.action_space.shape
        assert np.isclose(sum(action), 1.0), 'portfolio weights need to sum to 1'

        done = False

        # 1. Calculate the fees that we have to pay due to the change of weights 
        prev_weights = self.state[:,-1]
        delta_weights = np.abs(action - prev_weights)
        delta_weights[0] = 0.
        fees = self.COMMISSION_RATE * delta_weights 

        # 2. Retrieve new prices from our data
        prev_prices = self.state[:,2]

        self.timestep += 1
        self.position += 1

        self.state = self.data.iloc[self.position].values
        self.state = self.state.reshape(self.num_assets,3)

        # Normalize with regard to the first closing price of the episode
        self.state /= self.initial_closing_prices 

        # Third column is the closing prices
        prices = self.state[:,2]

        # 3. Calculate new portfolio value and reward
        prev_portfolio_value = self.portfolio_value

        holdings = prev_portfolio_value * (prices / prev_prices * action - fees)

        self.portfolio_value = np.sum(holdings)
        reward = np.log(self.portfolio_value / prev_portfolio_value) 

        # 4. Update the weights
        weights = holdings / self.portfolio_value

        # Add the weights to the observation space
        weights = np.expand_dims(weights, axis=1)
        self.state = np.concatenate((self.state, weights), axis=1)

        # Check whether the episode is finished
        done = bool(self.timestep % self.timesteps_per_episode == 0)

        return self.state, reward, done, {}

    def render(self, mode='human'):
        raise NotImplementedError


def main():
    pass

if __name__ == '__main__':
    main()
