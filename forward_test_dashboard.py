import os
import json
import warnings
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION REPOSITORY PATHS (MAPPED TO REPO)
# =====================================================================
SCORED_FEATURE_STORE_PATH = "oos_predictions.parquet" 
SCORE_COLUMN_NAME = "prediction"                  # Adjust if your script named it 'alpha_score'
PPO_MODEL_PATH = "best_ppo_agent.zip"             

CURRENT_PORTFOLIO_PATH = "current_portfolio.json"
TARGET_PORTFOLIO_PATH = "target_portfolio.json"
ORDERS_OUTPUT_CSV = "execution_orders.csv"

MAX_ASSET_WEIGHT = 0.15      # 15% Concentration Ceiling (AFM Risk Limit)
MIN_WEIGHT_THRESHOLD = 0.02  # 2% Minimum allocation threshold

# =====================================================================
# 1. LOAD PRE-SCORED MARKET DATA (OOS PREDICTIONS)
# =====================================================================
def load_latest_scored_slice(filepath, score_col):
    print(f"Loading out-of-sample predictions from: {filepath}...")
    df = pd.read_parquet(filepath)

    date_col = next((c for c in df.columns if c.lower() in ['date', 'timestamp', 'datetime']), None)
    ticker_col = next((c for c in df.columns if c.lower() in ['ticker', 'symbol', 'instrument']), None)

    if not date_col or not ticker_col:
        raise ValueError("Dataset must contain identifiable 'date' and 'ticker' columns.")
    if score_col not in df.columns:
        # Fallback check if the column was named differently in your LightGBM script
        fallback = 'alpha_score' if 'alpha_score' in df.columns else None
        if fallback:
            score_col = fallback
        else:
            raise ValueError(f"Score column '{score_col}' not found. Available columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    latest_date = df[date_col].max()
    print(f"Latest market snapshot identified: {latest_date.strftime('%Y-%m-%d')}")

    latest_slice = df[df[date_col] == latest_date].copy()
    print(f"Loaded {len(latest_slice)} active tickers for execution.")
    return latest_slice, ticker_col, score_col, latest_date

# =====================================================================
# 2. EXTRACT TOP ALPHA CANDIDATES 
# =====================================================================
def extract_top_candidates(df_slice, ticker_col, score_col):
    top_20_df = df_slice.sort_values(by=score_col, ascending=False).head(20).reset_index(drop=True)
    top_20_tickers = top_20_df[ticker_col].values

    exclude_cols = [ticker_col.lower(), score_col.lower(), 'date', 'datetime', 'timestamp', 'target', 'return']
    feature_cols = [c for c in top_20_df.columns if c.lower() not in exclude_cols]
    
    top_20_features = top_20_df[feature_cols].select_dtypes(include=[np.number]).values
    return top_20_tickers, top_20_features

# =====================================================================
# 3. RUN PPO INFERENCE & ENFORCE 15% CAP + CASH SHIELD
# =====================================================================
def run_risk_engine(top_20_tickers, top_20_features, model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PPO agent model missing at {model_path}")

    print(f"Loading PPO Risk Manager from: {model_path}...")
    ppo_agent = PPO.load(model_path)

    obs = top_20_features.flatten()
    expected_dim = ppo_agent.observation_space.shape[0]

    if len(obs) != expected_dim:
        obs = np.pad(obs, (0, expected_dim - len(obs))) if len(obs) < expected_dim else obs[:expected_dim]

    action, _ = ppo_agent.predict(obs, deterministic=True)

    exp_weights = np.exp(action - np.max(action))
    raw_weights = exp_weights / np.sum(exp_weights)

    clipped_weights = np.clip(raw_weights[:len(top_20_tickers)], 0.0, MAX_ASSET_WEIGHT)
    total_equity_weight = np.sum(clipped_weights)
    cash_shield_weight = max(0.0, 1.0 - float(total_equity_weight))

    target_portfolio = {}
    for ticker, weight in zip(top_20_tickers, clipped_weights):
        if weight >= MIN_WEIGHT_THRESHOLD:
            target_portfolio[ticker] = round(float(weight), 4)

    target_portfolio['CASH_LIQUIDBEES'] = round(cash_shield_weight, 4)
    return target_portfolio

# =====================================================================
# 4. GENERATE REBALANCE ORDERS & DELTAS
# =====================================================================
def generate_order_sheet(target_portfolio, current_portfolio_path):
    if os.path.exists(current_portfolio_path):
        with open(current_portfolio_path, 'r') as f:
            current_portfolio = json.load(f)
    else:
        current_portfolio = {'CASH_LIQUIDBEES': 1.0}

    orders = []

    for asset, curr_w in current_portfolio.items():
        if asset == 'CASH_LIQUIDBEES': continue
        tgt_w = target_portfolio.get(asset, 0.0)

        if tgt_w == 0.0:
            orders.append({'action': 'SELL ALL', 'ticker': asset, 'current_pct': curr_w * 100, 'target_pct': 0.0, 'delta_pct': -curr_w * 100, 'note': 'Rank drop'})
        elif tgt_w < (curr_w - 0.01):
            orders.append({'action': 'TRIM', 'ticker': asset, 'current_pct': curr_w * 100, 'target_pct': tgt_w * 100, 'delta_pct': (tgt_w - curr_w) * 100, 'note': 'Trimming excess'})

    for asset, tgt_w in target_portfolio.items():
        if asset == 'CASH_LIQUIDBEES': continue
        curr_w = current_portfolio.get(asset, 0.0)

        if curr_w == 0.0:
            orders.append({'action': 'BUY NEW', 'ticker': asset, 'current_pct': 0.0, 'target_pct': tgt_w * 100, 'delta_pct': tgt_w * 100, 'note': 'New Top Alpha'})
        elif tgt_w > (curr_w + 0.01):
            orders.append({'action': 'ADD MORE', 'ticker': asset, 'current_pct': curr_w * 100, 'target_pct': tgt_w * 100, 'delta_pct': (tgt_w - curr_w) * 100, 'note': 'Scaling up'})
        elif abs(tgt_w - curr_w) <= 0.01:
            orders.append({'action': 'HOLD', 'ticker': asset, 'current_pct': curr_w * 100, 'target_pct': tgt_w * 100, 'delta_pct': 0.0, 'note': 'In tolerance'})

    return orders, current_portfolio

# =====================================================================
# 5. CLI & GITHUB ACTIONS RENDERING
# =====================================================================
def display_dashboard(target_portfolio, orders, latest_date):
    print("\n" + "=" * 70)
    print(f" 🚀 NIFTY 500 MOMENTUM DASHBOARD | CYCLE DATE: {latest_date.strftime('%Y-%m-%d')}")
    print("=" * 70)

    print("\n[ TARGET WEIGHT ALLOCATION ]")
    print("-" * 70)
    for asset, weight in sorted(target_portfolio.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset:<18} | Target: {weight*100:>6.2f}%")

    print("\n[ EXECUTION ACTION SHEET ]")
    print("-" * 70)
    for order in orders:
        badge = f"[{order['action']}]"
        print(f"  {badge:<12} {order['ticker']:<15} | Current: {order['current_pct']:>5.2f}% -> Target: {order['target_pct']:>5.2f}% | {order['note']}")

    cash_target = target_portfolio.get('CASH_LIQUIDBEES', 0.0)
    print("-" * 70)
    print(f"🛡️  DEFENSIVE CASH SHIELD : {cash_target*100:.2f}% into LiquidBeES / TREPS")
    print("=" * 70 + "\n")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        markdown = [
            f"# 🚀 Nifty 500 Momentum Rebalance Dashboard ({latest_date.strftime('%Y-%m-%d')})\n",
            f"**Defensive Cash Shield:** `{cash_target*100:.2f}%` | **Total Equity:** `{(1-cash_target)*100:.2f}%`\n",
            "### 📋 Broker Action Sheet\n",
            "| Action | Ticker | Current Weight | Target Weight | Change | Note |",
            "| :--- | :--- | :---: | :---: | :---: | :--- |"
        ]
        for o in orders:
            markdown.append(f"| **{o['action']}** | `{o['ticker']}` | {o['current_pct']:.2f}% | {o['target_pct']:.2f}% | {o['delta_pct']:+.2f}% | {o['note']} |")
        markdown.append(f"\n> **Risk Guidance:** Route **{cash_target*100:.2f}%** to LiquidBeES to maintain defensive barrier.\n")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown))

if __name__ == "__main__":
    df_slice, ticker_col, score_col, latest_date = load_latest_scored_slice(SCORED_FEATURE_STORE_PATH, SCORE_COLUMN_NAME)
    top_tickers, top_features = extract_top_candidates(df_slice, ticker_col, score_col)
    target_portfolio = run_risk_engine(top_tickers, top_features, PPO_MODEL_PATH)
    orders, current_portfolio = generate_order_sheet(target_portfolio, CURRENT_PORTFOLIO_PATH)
    display_dashboard(target_portfolio, orders, latest_date)

    with open(TARGET_PORTFOLIO_PATH, 'w') as f:
        json.dump(target_portfolio, f, indent=4)
    pd.DataFrame(orders).to_csv(ORDERS_OUTPUT_CSV, index=False)
