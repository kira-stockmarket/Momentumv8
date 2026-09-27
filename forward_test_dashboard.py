import pandas as pd
import numpy as np
import lightgbm as lgb
from stable_baselines3 import PPO
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. LOAD MODELS & LATEST MARKET DATA
# ==========================================
def load_production_models():
    print("Loading Alpha Engine (LightGBM)...")
    lgb_model = lgb.Booster(model_file='lgb_alpha_model.txt')
    
    print("Loading Risk Agent (PPO)...")
    ppo_agent = PPO.load("best_ppo_agent.zip")
    
    return lgb_model, ppo_agent

def generate_target_portfolio(latest_data_path, lgb_model, ppo_agent):
    # Load today's live Nifty 500 feature data
    df = pd.read_csv(latest_data_path)
    tickers = df['ticker'].values
    features = df.drop(columns=['ticker', 'date']).values
    
    # ---------------------------------------
    # PHASE 1: ALPHA GENERATION (LightGBM)
    # ---------------------------------------
    # Predict momentum scores for all 500 stocks
    alpha_scores = lgb_model.predict(features)
    
    # Rank and extract the Top 20 Candidates
    top_20_idx = np.argsort(alpha_scores)[-20:][::-1]
    top_20_tickers = tickers[top_20_idx]
    top_20_features = features[top_20_idx]
    
    # ---------------------------------------
    # PHASE 2: RISK MANAGEMENT (PPO Agent)
    # ---------------------------------------
    # Flatten the state to pass to the RL agent (same shape as training)
    observation_state = top_20_features.flatten()
    
    # Get deterministic execution weights from the trained agent
    action, _states = ppo_agent.predict(observation_state, deterministic=True)
    
    # ---------------------------------------
    # PHASE 3: INSTITUTIONAL SIZING
    # ---------------------------------------
    # Convert RL actions to softmax probabilities to ensure they sum to <= 1.0
    exp_weights = np.exp(action - np.max(action))
    raw_weights = exp_weights / exp_weights.sum()
    
    # Enforce the 15% Maximum Asset Concentration Limit
    target_weights = np.clip(raw_weights, 0.0, 0.15)
    
    # Calculate the remaining capital for the Defensive Cash Shield
    total_equity_exposure = np.sum(target_weights)
    cash_weight = 1.0 - total_equity_exposure
    
    # Build the final dictionary
    target_portfolio = {ticker: weight for ticker, weight in zip(top_20_tickers, target_weights) if weight > 0.01}
    target_portfolio['CASH_LIQUIDBEES'] = cash_weight
    
    return target_portfolio

# ==========================================
# 2. GENERATE EXECUTION DASHBOARD
# ==========================================
def print_execution_dashboard(target_portfolio, current_portfolio=None):
    if current_portfolio is None:
        current_portfolio = {} # Simulating an empty portfolio for the first run
        
    print("\n" + "="*55)
    print(" 🚀 NIFTY 500 MOMENTUM: FORWARD TESTING DASHBOARD")
    print("="*55)
    
    print("\n[ TARGET PORTFOLIO ALLOCATION ]")
    print("-" * 55)
    for asset, weight in sorted(target_portfolio.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset:<20} | Target: {weight*100:>5.2f}%")
        
    print("\n[ ACTION SHEET: BROKER EXECUTION ]")
    print("-" * 55)
    
    # 1. SELL ORDERS (Assets we hold but AI dropped, or need trimming)
    sells = []
    for asset, curr_weight in current_portfolio.items():
        if asset == 'CASH_LIQUIDBEES': continue
        tgt_weight = target_portfolio.get(asset, 0.0)
        
        if tgt_weight == 0.0:
            sells.append(f"🔴 SELL ALL    -> {asset:<15} (Momentum decay detected)")
        elif tgt_weight < curr_weight:
            trim_amt = curr_weight - tgt_weight
            sells.append(f"🟡 TRIM        -> {asset:<15} (Reduce by {trim_amt*100:.2f}%)")
            
    # 2. BUY ORDERS (New alphas or scaling up)
    buys = []
    for asset, tgt_weight in target_portfolio.items():
        if asset == 'CASH_LIQUIDBEES': continue
        curr_weight = current_portfolio.get(asset, 0.0)
        
        if curr_weight == 0.0:
            buys.append(f"🟢 BUY NEW     -> {asset:<15} (Allocate {tgt_weight*100:.2f}%)")
        elif tgt_weight > curr_weight:
            add_amt = tgt_weight - curr_weight
            buys.append(f"🔵 ADD MORE    -> {asset:<15} (Increase by {add_amt*100:.2f}%)")

    # 3. HOLD ORDERS
    holds = []
    for asset, curr_weight in current_portfolio.items():
        if asset == 'CASH_LIQUIDBEES': continue
        tgt_weight = target_portfolio.get(asset, 0.0)
        if abs(tgt_weight - curr_weight) < 0.01 and tgt_weight > 0:
            holds.append(f"⚪ HOLD        -> {asset:<15} (Target matched)")

    for action in sells + buys + holds:
        print(f"  {action}")

    print("-" * 55)
    cash_tgt = target_portfolio.get('CASH_LIQUIDBEES', 0.0)
    print(f"🛡️  DEFENSE : Route {cash_tgt*100:.2f}% of total equity to LiquidBeES / TREPS")
    print("="*55)

if __name__ == "__main__":
    # Simulate loading models and generating today's execution sheet
    # lgb_model, ppo_agent = load_production_models()
    # target_portfolio = generate_target_portfolio("latest_nifty500_features.csv", lgb_model, ppo_agent)
    
    # MOCK DATA FOR DASHBOARD PREVIEW
    mock_target_portfolio = {
        'INFY': 0.15,
        'ASIANPAINT': 0.12,
        'BAJFINANCE': 0.15,
        'TRENT': 0.10,
        'ZOMATO': 0.08,
        'HAL': 0.10,
        'CASH_LIQUIDBEES': 0.30 
    }
    
    mock_current_portfolio = {
        'INFY': 0.15,          # Matched (Hold)
        'ASIANPAINT': 0.18,    # Trim to 12%
        'RELIANCE': 0.10,      # Sell All (Momentum dropped)
        'TRENT': 0.05,         # Add More to 10%
        'CASH_LIQUIDBEES': 0.52
    }
    
    print_execution_dashboard(mock_target_portfolio, mock_current_portfolio)
