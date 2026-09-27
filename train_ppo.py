import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import PortfolioAllocationEnv
from typing import Callable

# --- Configuration ---
DATA_PATH = "oos_predictions.parquet"
MODEL_OUTPUT = "best_ppo_agent.zip"
TOTAL_TIMESTEPS = 250000  # 25 Lakh steps

# --- FIX: chronological train/eval split -----------------------------------
# Previously this script trained on the exact same file that
# backtest_simulation.py later evaluated the agent on. TRAIN_FRACTION and
# PURGE_DAYS carve out a held-out tail of oos_predictions.parquet that the
# agent NEVER sees during these 2.5M timesteps. PURGE_DAYS mirrors the
# HORIZON_PURGE used when the LightGBM ranker's OOS predictions were built,
# so that a training-window date's forward-looking target_raw_ret (which
# looks rebalance_freq days ahead) can't reach into the eval window.
TRAIN_FRACTION = 0.70
PURGE_DAYS = 10
EVAL_SPLIT_FILE = "eval_start_date.txt"
# -----------------------------------------------------------------------------


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate decay to stabilize policy updates over 2.5M steps."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def get_train_eval_dates(data_path, train_fraction=TRAIN_FRACTION, purge_days=PURGE_DAYS):
    """Returns (train_end_date, eval_start_date), split strictly by calendar
    date order with a purge gap between them. Never shuffles rows."""
    dates_only = pd.read_parquet(data_path, columns=["Date"])
    unique_dates = np.sort(dates_only['Date'].unique())
    n = len(unique_dates)

    split_idx = int(n * train_fraction)
    train_end_date = unique_dates[split_idx]

    eval_start_idx = min(split_idx + purge_days, n - 1)
    eval_start_date = unique_dates[eval_start_idx]

    print("=" * 60)
    print("CHRONOLOGICAL TRAIN / EVAL SPLIT")
    print(f"  Train window : {unique_dates[0]} -> {train_end_date}  ({split_idx} days)")
    print(f"  Purge gap    : {purge_days} trading days (excluded from both)")
    print(f"  Eval window  : {eval_start_date} -> {unique_dates[-1]}  "
          f"({n - eval_start_idx} days, HELD OUT — never trained on)")
    print("=" * 60)
    return str(train_end_date), str(eval_start_date)


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Critical Error: {DATA_PATH} not found. Must complete Step 3 first.")
        sys.exit(1)

    train_end_date, eval_start_date = get_train_eval_dates(DATA_PATH)

    # Persist the split so backtest_simulation.py evaluates ONLY on the
    # untouched holdout window, without duplicating the split logic.
    with open(EVAL_SPLIT_FILE, "w") as f:
        f.write(eval_start_date)

    print("Initializing Advanced Financial Management (AFM) PPO Allocator...")
    env = PortfolioAllocationEnv(
        data_path=DATA_PATH,
        top_k=20,
        rebalance_freq=10,
        initial_capital=10_000_000.0,
        tx_cost=0.0015,
        end_date=train_end_date,   # <-- the fix: PPO only ever sees the train window
    )

    vec_env = DummyVecEnv([lambda: env])

    ppo_params = {
        "policy": "MlpPolicy",
        "env": vec_env,
        "learning_rate": linear_schedule(3e-4),
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.02,
        "max_grad_norm": 0.5,
        "verbose": 1
    }

    model = PPO(**ppo_params)

    print(f"Starting Reinforcement Learning Phase: Executing {TOTAL_TIMESTEPS:,} market transitions...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
    except Exception as e:
        print(f"Training interrupted: {e}")
        sys.exit(1)

    print(f"Training Complete. Serializing PyTorch model weights to {MODEL_OUTPUT}...")
    model.save(MODEL_OUTPUT)

    print("NOTE: any in-training rollout stats above are on the TRAIN window only")
    print("and are not a performance claim. Run backtest_simulation.py next — it")
    print(f"will evaluate strictly on dates >= {eval_start_date}, which this agent")
    print("has never seen.")
