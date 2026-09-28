import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION & HONEST COSTS
# =====================================================================
DATA_PATH = "nifty500_features_scored.csv" # Or oos_predictions.parquet
SCORE_COL = "alpha_score"                  # LightGBM predictions column

INITIAL_CAPITAL = 100_000_000.0            # ₹10 Crore
SLIPPAGE_BPS = 20                          # 0.20% slippage per leg
STT_BPS = 10                               # 0.10% STT + Brokerage per leg
TOTAL_COST_PER_LEG = (SLIPPAGE_BPS + STT_BPS) / 10000.0  # 0.30% per trade

# =====================================================================
# 1. DATA VALIDATION GUARD
# =====================================================================
def validate_and_clean_prices(df):
    """Nullifies impossible price glitches (unadjusted splits) to prevent label corruption."""
    print("Validating price data for glitches...")
    df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    
    # Identify price spikes > 50% in a single 10-day period
    df['period_return'] = df.groupby('ticker')['close'].pct_change()
    glitch_mask = (df['period_return'] > 0.50) | (df['period_return'] < -0.50)
    
    glitch_count = glitch_mask.sum()
    if glitch_count > 0:
        print(f"⚠️ Cleaned {glitch_count} impossible price bars.")
        df.loc[glitch_mask, 'close'] = np.nan
        df['close'] = df.groupby('ticker')['close'].ffill()
        
    return df.drop(columns=['period_return'])

# =====================================================================
# 2. TREND REGIME FILTER (Equal-Weight Nifty 500)
# =====================================================================
def calculate_regime_filter(df):
    """Calculates the blended trend filter using the EW universe index."""
    print("Calculating Blended Trend Regime Filter...")
    # Create Equal-Weight Index
    ew_index = df.groupby('date')['close'].mean().reset_index()
    ew_index = ew_index.sort_values('date')
    
    ew_index['sma50'] = ew_index['close'].rolling(50).mean()
    ew_index['sma100'] = ew_index['close'].rolling(100).mean()
    ew_index['sma200'] = ew_index['close'].rolling(200).mean()
    
    # Calculate exposure: 0.0, 0.33, 0.67, or 1.0 depending on how many SMAs are cleared
    ew_index['up_signals'] = (
        (ew_index['close'] > ew_index['sma50']).astype(int) + 
        (ew_index['close'] > ew_index['sma100']).astype(int) + 
        (ew_index['close'] > ew_index['sma200']).astype(int)
    )
    ew_index['equity_exposure'] = ew_index['up_signals'] / 3.0
    
    return ew_index.set_index('date')['equity_exposure'].to_dict()

# =====================================================================
# 3. THE RULE-BASED BACKTEST ENGINE
# =====================================================================
def run_honest_backtest(df, regime_dict):
    dates = sorted(df['date'].unique())
    
    portfolio_value = INITIAL_CAPITAL
    current_holdings = {} # {ticker: weight}
    
    results = []
    
    print(f"Running Honest Backtest over {len(dates)} periods...")
    
    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i+1]
        
        df_slice = df[df['date'] == current_date].copy()
        if df_slice.empty: continue
            
        # 1. Get Regime Target Exposure
        target_equity_exposure = regime_dict.get(current_date, 1.0)
        
        # 2. Rank by Alpha Score
        df_slice['rank'] = df_slice[SCORE_COL].rank(ascending=False, method='first')
        
        target_holdings = []
        
        if target_equity_exposure > 0:
            # HYSTERESIS: Keep current holdings if they rank 45 or better
            for ticker in current_holdings.keys():
                ticker_data = df_slice[df_slice['ticker'] == ticker]
                if not ticker_data.empty and ticker_data['rank'].iloc[0] <= 45:
                    target_holdings.append(ticker)
                    
            # NEW ENTRIES: Fill remaining slots up to 15 from the Top 15
            top_15 = df_slice[df_slice['rank'] <= 15].sort_values('rank')['ticker'].tolist()
            for ticker in top_15:
                if len(target_holdings) >= 15: break
                if ticker not in target_holdings:
                    target_holdings.append(ticker)
                    
        # 3. Calculate Target Weights
        target_weights = {}
        if target_holdings:
            weight_per_stock = target_equity_exposure / len(target_holdings)
            for ticker in target_holdings:
                target_weights[ticker] = weight_per_stock
        
        cash_weight = 1.0 - sum(target_weights.values())
        
        # 4. Calculate Turnover & Execution Costs
        turnover = 0.0
        all_assets = set(current_holdings.keys()).union(set(target_weights.keys()))
        for asset in all_assets:
            curr_w = current_holdings.get(asset, 0.0)
            tgt_w = target_weights.get(asset, 0.0)
            turnover += abs(tgt_w - curr_w)
            
        turnover /= 2.0 # One-sided turnover
        execution_cost_pct = turnover * TOTAL_COST_PER_LEG
        
        # Deduct execution friction from portfolio NAV
        portfolio_value *= (1.0 - execution_cost_pct)
        
        # 5. Calculate Period Returns (Buy & Hold to next date)
        next_df_slice = df[df['date'] == next_date]
        period_return = 0.0
        
        for ticker, weight in target_weights.items():
            curr_px = df_slice[df_slice['ticker'] == ticker]['close'].values
            next_px = next_df_slice[next_df_slice['ticker'] == ticker]['close'].values
            
            if len(curr_px) > 0 and len(next_px) > 0:
                ret = (next_px[0] / curr_px[0]) - 1.0
                period_return += weight * ret
                
        # Add Cash Yield (Approx 6.5% annualized / 252 * 10 days)
        period_return += cash_weight * (0.065 / 25.2)
        
        # 6. Update NAV
        portfolio_value *= (1.0 + period_return)
        current_holdings = target_weights.copy()
        
        results.append({
            'date': next_date,
            'portfolio_value': portfolio_value,
            'net_return': period_return - execution_cost_pct,
            'turnover': turnover,
            'equity_exposure': target_equity_exposure
        })
        
    return pd.DataFrame(results)

# =====================================================================
# 4. TEARSHEET GENERATION
# =====================================================================
def generate_tearsheet(results_df):
    results_df['cumulative_return'] = results_df['portfolio_value'] / INITIAL_CAPITAL
    results_df['peak'] = results_df['portfolio_value'].cummax()
    results_df['drawdown'] = (results_df['portfolio_value'] - results_df['peak']) / results_df['peak']
    
    total_return = (results_df['portfolio_value'].iloc[-1] / INITIAL_CAPITAL) - 1.0
    years = len(results_df) * 10 / 252.0
    cagr = (1 + total_return) ** (1 / years) - 1.0
    
    vol = results_df['net_return'].std() * np.sqrt(25.2)
    sharpe = (cagr - 0.065) / vol if vol > 0 else 0
    max_dd = results_df['drawdown'].min()
    avg_turnover = results_df['turnover'].mean()
    
    print("\n==================================================")
    print("      RULE-BASED HONEST BACKTEST TEARSHEET")
    print("==================================================")
    print(f"Final Capital     : ₹{results_df['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Net CAGR          : {cagr*100:.2f}%")
    print(f"Annual Volatility : {vol*100:.2f}%")
    print(f"Sharpe Ratio      : {sharpe:.2f}")
    print(f"Max Drawdown      : {max_dd*100:.2f}%")
    print(f"Avg Bi-weekly T/O : {avg_turnover*100:.2f}%")
    print("==================================================\n")
    
    results_df.to_csv("honest_backtest_results.csv", index=False)

if __name__ == "__main__":
    if DATA_PATH.endswith(".parquet"):
        df = pd.read_parquet(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)
        
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Clean Data
    df_clean = validate_and_clean_prices(df)
    
    # 2. Compute Regime
    regime_dict = calculate_regime_filter(df_clean)
    
    # 3. Run Simulation
    results_df = run_honest_backtest(df_clean, regime_dict)
    
    # 4. Print Tearsheet
    generate_tearsheet(results_df)
