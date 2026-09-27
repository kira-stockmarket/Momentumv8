import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioAllocationEnv(gym.Env):
    """
    Advanced Financial Management (AFM) Institutional Portfolio Environment.
    State Space: Alpha scores, current allocations, drawdown, and market breadth.
    Action Space: Continuous vector representing target weights (including Cash).
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_path="oos_predictions.parquet", top_k=20, rebalance_freq=10, initial_capital=10000000.0, tx_cost=0.0015):
        super(PortfolioAllocationEnv, self).__init__()
        
        self.top_k = top_k
        self.rebalance_freq = rebalance_freq
        self.initial_capital = initial_capital
        self.tx_cost = tx_cost
        
        print("Loading OOS Signal Matrix into Gymnasium Environment...")
        self.df = pd.read_parquet(data_path)
        
        # Isolate rebalancing epochs to precisely match the 10-day forward return horizon
        self.unique_dates = np.sort(self.df['Date'].unique())
        self.rebalance_dates = self.unique_dates[::self.rebalance_freq]
        
        # State Dimension: 20 Alphas + 21 Weights + 1 Drawdown + 1 Breadth = 43
        self.obs_dim = self.top_k + (self.top_k + 1) + 2
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        # Action Space: 21 continuous dimensions [0.0 to 1.0] representing asset weights + 1 cash buffer
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.top_k + 1,), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        
        # Portfolio initializes in 100% Cash (Index 20)
        self.current_weights = np.zeros(self.top_k + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0 
        
        self.history = []
        return self._get_observation(), {}
        
    def _get_observation(self):
        if self.current_step >= len(self.rebalance_dates):
            return np.zeros(self.obs_dim, dtype=np.float32)
            
        current_date = self.rebalance_dates[self.current_step]
        day_df = self.df[self.df['Date'] == current_date]
        
        # Market regime proxy: average cross-sectional alpha for the session
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
        
        # Ensure matrix bounds prevent Torch nan-propagation
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs.astype(np.float32)
        
    def step(self, action):
        # 1. Action Normalization
        action = np.clip(action, 0.0, 1.0)
        target_weights = action / (np.sum(action) + 1e-9)
        
        # 2. Institutional Execution Costs
        turnover = np.sum(np.abs(target_weights - self.current_weights)) / 2.0
        transaction_costs = turnover * self.tx_cost
        
        # 3. Mark-to-Market Accounting (Cash yields 0.0)
        asset_weights = target_weights[:-1]
        port_ret = np.sum(asset_weights * self.current_targets)
        net_ret = port_ret - transaction_costs
        
        self.portfolio_value *= (1.0 + net_ret)
        
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
            
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        
        # 4. Institutional Reward Shaping (Sortino Proxy)
        reward = net_ret 
        
        # Exponential penalization for severe drawdowns, forcing the agent to learn downside protection
        if drawdown > 0.05:
            reward -= (drawdown ** 2) * 5.0 
            
        reward -= turnover * 0.0005 
        
        self.current_weights = target_weights
        self.current_step += 1
        
        done = bool(self.current_step >= len(self.rebalance_dates) - 1)
        truncated = False
        
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
            print(f"Epoch Complete | Compounded Return: {total_ret*100:.2f}% | Peak Drawdown: {drawdown*100:.2f}% | Final Cash Position: {target_weights[-1]*100:.1f}%")
            
        return self._get_observation(), float(reward), done, truncated, info
