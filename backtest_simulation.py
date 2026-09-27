import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from trading_env import PortfolioAllocationEnv
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_PATH = "oos_predictions.parquet"
MODEL_PATH = "best_ppo_agent.zip"
INITIAL_CAPITAL = 100_000_000.0  # ₹10 Crore Base Capital
OUTPUT_CSV = "institutional_backtest_results.csv"

# --- INDIAN STATUTORY TAXES & SLIPPAGE ---
# Incorporating strict institutional friction
stt = 0.001          # 0.1% STT on buy and sell
stamp_duty = 0.00015 # 0.015% on buy side only
nse_fee = 0.0000307  # 0.00307% NSE transaction charge
sebi_fee = 0.000001  # Rs 10 per crore
gst = (nse_fee + sebi_fee) * 0.18
slippage = 0.0015    # 15 bps market impact per leg for large capital

buy_leg_cost = stt + stamp_duty + nse_fee + sebi_fee + gst + slippage
sell_leg_cost = stt + nse_fee + sebi_fee + gst + slippage
avg_tx_friction = (buy_leg_cost + sell_leg_cost) / 2.0  # Approx 0.0026 (26 bps)

def calculate_tearsheet(history_df, initial_capital, rebalance_freq):
    """Calculates institutional risk and performance metrics."""
    history_df['equity_curve'] = history_df['portfolio_value'] / initial_capital
    history_df['epoch_return'] = history_df['equity_curve'].pct_change().fillna(0)
    
    total_return = history_df['equity_curve'].iloc[-1] - 1.0
    
    # Annualization factor based on trading days
    epochs_per_year = 252 / rebalance_freq 
    cagr = (history_df['equity_curve'].iloc[-1] ** (1 / (len(history_df) / epochs_per_year))) - 1.0
    
    annual_vol = history_df['epoch_return'].std() * np.sqrt(epochs_per_year)
    
    risk_free_rate = 0.07 # 7% India 10Y Yield proxy
    sharpe_ratio = (cagr - risk_free_rate) / (annual_vol + 1e-9)
    
    downside_returns = history_df[history_df['epoch_return'] < 0]['epoch_return']
    downside_vol = downside_returns.std() * np.sqrt(epochs_per_year)
    sortino_ratio = (cagr - risk_free_rate) / (downside_vol + 1e-9)
    
    max_drawdown = history_df['drawdown'].max()
    win_rate = (history_df['net_return'] > 0).mean()
    
    print("\n" + "="*50)
    print("      INSTITUTIONAL PERFORMANCE TEARSHEET")
    print("="*50)
    print(f"Initial Capital   : ₹{initial_capital:,.2f}")
    print(f"Final Capital     : ₹{history_df['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Absolute Return   : {total_return * 100:.2f}%")
    print(f"Net CAGR          : {cagr * 100:.2f}%")
    print(f"Annual Volatility : {annual_vol * 100:.2f}%")
    print("-" * 50)
    print(f"Max Drawdown      : -{max_drawdown * 100:.2f}%")
    print(f"Sharpe Ratio      : {sharpe_ratio:.2f}")
    print(f"Sortino Ratio     : {sortino_ratio:.2f}")
    print(f"Win Rate (Epochs) : {win_rate * 100:.1f}%")
    print(f"Avg Cash Position : {history_df['cash_weight'].mean() * 100:.1f}%")
    print("="*50)

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("Missing prerequisite files. Ensure Step 3 and 4 completed successfully.")
        sys.exit(1)

    print("Initializing environment with T+1 real-world execution friction...")
    env = PortfolioAllocationEnv(
        data_path=DATA_PATH, 
        top_k=20, 
        rebalance_freq=10,
        initial_capital=INITIAL_CAPITAL,
        tx_cost=avg_tx_friction 
    )

    print("Loading optimized PPO weights...")
    model = PPO.load(MODEL_PATH)
    
    obs, _ = env.reset()
    done = False
    
    print("Executing deterministic backtest (Zero Look-Ahead Bias)...")
    while not done:
        # deterministic=True forces the agent to take the statistically optimal action
        # rather than exploring stochastically as it did during training.
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

    history_df = pd.DataFrame(env.history)
    history_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nExecution log exported to {OUTPUT_CSV}")
    
    calculate_tearsheet(history_df, INITIAL_CAPITAL, env.rebalance_freq)
