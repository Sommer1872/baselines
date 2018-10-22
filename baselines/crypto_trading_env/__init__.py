from gym.envs.registration import register

register(
    id='crypto-trading-v0',
    entry_point='baselines.crypto_trading_env.envs:PortfolioTradingEnv',
)
