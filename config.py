"""Single source of truth for parameters that MUST agree across the pipeline.

Previously FORWARD_HORIZON (feature_pipeline.py) and rebalance_freq (trading_env.py,
train_ppo.py, backtest_simulation.py) were hard-coded independently. If they diverge,
every training label is silently misaligned with the real holding period.
"""
import os

# --- Strategy geometry -------------------------------------------------------
FORWARD_HORIZON = 10        # trading days a label looks ahead (feature_pipeline.py)
REBALANCE_FREQ = 10         # trading days between rebalances (must equal FORWARD_HORIZON)
TOP_K = 20                  # candidate slots shown to the PPO agent
MAX_ASSET_WEIGHT = 0.15     # per-name cap applied inside trading_env.py
CASH_YIELD_ANNUAL = 0.065   # LiquidBees / TREPS proxy, same as trading_env.py

assert FORWARD_HORIZON == REBALANCE_FREQ, "label horizon must match holding period"

# --- Indian equity delivery costs (same components as backtest_simulation.py) --
STT = 0.001
STAMP_DUTY = 0.00015
NSE_FEE = 0.0000307
SEBI_FEE = 0.000001
GST = (NSE_FEE + SEBI_FEE) * 0.18
SLIPPAGE = 0.0015           # per leg
BUY_LEG_COST = STT + STAMP_DUTY + NSE_FEE + SEBI_FEE + GST + SLIPPAGE
SELL_LEG_COST = STT + NSE_FEE + SEBI_FEE + GST + SLIPPAGE

# --- Paths -------------------------------------------------------------------
DATA_DIR = "data"
FEATURE_DIR = "features"
OOS_PREDICTIONS_PATH = "oos_predictions.parquet"
LGBM_MODEL_PATH = "best_lightgbm_ranker.joblib"
PPO_MODEL_PATH = "best_ppo_agent.zip"
FORWARD_TEST_DIR = "forward_test"

# --- Forward-test guards -------------------------------------------------------
INITIAL_CAPITAL = float(os.getenv("PORTFOLIO_CAPITAL", "100000000"))
MAX_DATA_AGE_DAYS = 5       # calendar days; covers weekend + one holiday
MIN_SYMBOL_COVERAGE = 0.90  # share of live symbols that must have a bar on the latest date
