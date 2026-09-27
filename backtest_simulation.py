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
OUTPUT_CSV = "institutional_backtest_results_holdout.csv"
EVAL_SPLIT_FILE = "eval_start_date.txt"  # written by the fixed train_ppo.py

# --- INDIAN STATUTORY TAXES & SLIPPAGE ---
stt = 0.001
stamp_duty = 0.00015
nse_fee = 0.0000307
sebi_fee = 0.000001
gst = (nse_fee + sebi_fee) * 0.18
slippage = 0.0015
buy_leg_cost = stt + stamp_duty + nse_fee + sebi_fee + gst + slippage
sell_leg_cost = stt + nse_fee + sebi_fee + gst + slippage
avg_tx_friction = (buy_leg_cost + sell_leg_cost) / 2.0


def calculate_tearsheet(history_df, initial_capital, rebalance_freq):
    """Calculates institutional risk and performance metrics."""
    history_df['equity_curve'] = history_df['portfolio_value'] / initial_capital
    history_df['epoch_return'] = history_df['equity_curve'].pct_change().fillna(0)
    total_return = history_df['equity_curve'].iloc[-1] - 1.0

    epochs_per_year = 252 / rebalance_freq
    cagr = (history_df['equity_curve'].iloc[-1] ** (1 / (len(history_df) / epochs_per_year))) - 1.0
    annual_vol = history_df['epoch_return'].std() * np.sqrt(epochs_per_year)
    risk_free_rate = 0.07
    sharpe_ratio = (cagr - risk_free_rate) / (annual_vol + 1e-9)

    downside_returns = history_df[history_df['epoch_return'] < 0]['epoch_return']
    downside_vol = downside_returns.std() * np.sqrt(epochs_per_year)
    sortino_ratio = (cagr - risk_free_rate) / (downside_vol + 1e-9)

    max_drawdown = history_df['drawdown'].max()
    win_rate = (history_df['net_return'] > 0).mean()

    print("\n" + "=" * 50)
    print(" HELD-OUT PERFORMANCE TEARSHEET (never seen by the agent)")
    print("=" * 50)
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
    print("=" * 50)


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("Missing prerequisite files. Ensure Step 3 and 4 completed successfully.")
        sys.exit(1)

    # --- FIX: refuse to run without a genuine holdout window --------------------
    # This is the guardrail that prevents ever again evaluating the agent on
    # data it trained on. eval_start_date.txt is produced by the fixed
    # train_ppo.py; if it's missing, that means training didn't record a
    # split, and this script should not silently fall back to evaluating on
    # the full (training-contaminated) file.
    if not os.path.exists(EVAL_SPLIT_FILE):
        print(f"Critical Error: {EVAL_SPLIT_FILE} not found.")
        print("Re-run the fixed train_ppo.py first — it writes this file to mark")
        print("which dates the agent was NOT trained on. Running this backtest")
        print("without it would evaluate on data the agent has already seen.")
        sys.exit(1)

    with open(EVAL_SPLIT_FILE) as f:
        eval_start_date = f.read().strip()
    # -----------------------------------------------------------------------------

    print(f"Initializing environment on HELD-OUT window only (dates >= {eval_start_date})...")
    print("Initializing environment with T+1 real-world execution friction...")
    env = PortfolioAllocationEnv(
        data_path=DATA_PATH,
        top_k=20,
        rebalance_freq=10,
        initial_capital=INITIAL_CAPITAL,
        tx_cost=avg_tx_friction,
        start_date=eval_start_date,  # <-- the fix: only the untouched tail
    )

    print("Loading optimized PPO weights...")
    model = PPO.load(MODEL_PATH)

    obs, _ = env.reset()
    done = False
    print("Executing deterministic backtest on held-out data (Zero Look-Ahead Bias)...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

    history_df = pd.DataFrame(env.history)
    history_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nExecution log exported to {OUTPUT_CSV}")
    calculate_tearsheet(history_df, INITIAL_CAPITAL, env.rebalance_freq)
