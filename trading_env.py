import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioAllocationEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, data_path="oos_predictions.parquet", top_k=20, rebalance_freq=10, initial_capital=100_000_000.0, tx_cost=0.0015):
        super(PortfolioAllocationEnv, self).__init__()
        
        self.top_k = top_k
        self.rebalance_freq = rebalance_freq
        self.initial_capital = initial_capital
        self.tx_cost = tx_cost
        
        self.cash_yield_per_epoch = (0.065 / 252) * self.rebalance_freq 
        
        print("Loading OOS Signal Matrix into Environment...")
        self.df = pd.read_parquet(data_path)
        self.unique_dates = np.sort(self.df['Date'].unique())
        self.rebalance_dates = self.unique_dates[::self.rebalance_freq]
        
        # State Space: Alphas (20) + Weights (21) + Drawdown (1) + Breadth (1) = 43
        self.obs_dim = self.top_k + (self.top_k + 1) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.top_k + 1,), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        
        self.current_weights = np.zeros(self.top_k + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0  # Start in Cash
        
        self.history = []
        return self._get_observation(), {}
        
    def _get_observation(self):
        if self.current_step >= len(self.rebalance_dates):
            return np.zeros(self.obs_dim, dtype=np.float32)
            
        current_date = self.rebalance_dates[self.current_step]
        day_df = self.df[self.df['Date'] == current_date]
        
        market_breadth = day_df['alpha_score'].mean() if not day_df.empty else 0.0
        top_candidates = day_df.nlargest(self.top_k, 'alpha_score')
        
        alphas = np.zeros(self.top_k, dtype=np.float32)
        self.current_targets = np.zeros(self.top_k, dtype=np.float32)
        
        if not top_candidates.empty:
            num_valid = min(self.top_k, len(top_candidates))
            alphas[:num_valid] = top_candidates['alpha_score'].values[:num_valid]
            self.current_targets[:num_valid] = top_candidates['target_raw_ret'].values[:num_valid]
            
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        
        obs = np.concatenate([
            alphas,
            self.current_weights,
            np.array([drawdown, market_breadth], dtype=np.float32)
        ])
        
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
    def step(self, action):
        # 1. Action Normalization & Zero-Sum Protection
        action = np.clip(action, 0.0, 1.0)
        action_sum = np.sum(action)
        
        if action_sum == 0:
            # If the agent outputs all zeros, default to 100% Cash/TREPS
            target_weights = np.zeros(self.top_k + 1, dtype=np.float32)
            target_weights[-1] = 1.0
        else:
            target_weights = action / action_sum
            
        # 2. Relaxed Liquidity Constraint: Allow up to 15% concentration per asset
        asset_weights = target_weights[:-1]
        excess_weight = np.sum(np.maximum(asset_weights - 0.15, 0.0))
        asset_weights = np.minimum(asset_weights, 0.15)
        
        target_weights[:-1] = asset_weights
        target_weights[-1] += excess_weight
        
        # Final normalization with epsilon to mathematically prevent NaN corruption
        target_weights = target_weights / (np.sum(target_weights) + 1e-9)
        
        # 3. Execution Friction & Accounting
        turnover = np.sum(np.abs(target_weights - self.current_weights)) / 2.0
        transaction_costs = turnover * self.tx_cost
        
        cash_ret = target_weights[-1] * self.cash_yield_per_epoch
        equity_ret = np.sum(target_weights[:-1] * self.current_targets)
        net_ret = equity_ret + cash_ret - transaction_costs
        
        self.portfolio_value *= (1.0 + net_ret)
        
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
            
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        
        # 4. The "Electric Fence" Reward Function
        reward = net_ret * 100.0  
        
        if drawdown > 0.15:
            reward -= (drawdown ** 3) * 1000.0
            
        reward -= turnover * 0.05 
        
        self.current_weights = target_weights
        self.current_step += 1
        
        done = bool(self.current_step >= len(self.rebalance_dates) - 1)
        
        info = {
            'portfolio_value': self.portfolio_value,
            'net_return': net_ret,
            'drawdown': drawdown,
            'turnover': turnover,
            'cash_weight': target_weights[-1]
        }
        self.history.append(info)
        
        if done:
            total_ret = (self.portfolio_value / self.initial_capital) - 1.0
            print(f"Epoch Complete | Ret: {total_ret*100:.2f}% | DD: {drawdown*100:.2f}% | Cash: {target_weights[-1]*100:.1f}%")
            
        return self._get_observation(), float(reward), done, False, info
