import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUTPUT_FILE = "features_dataset.parquet"
FORWARD_HORIZON = 10  # 10-day swing horizon

def compute_garman_klass_vol(df, window=20):
    log_hl = (np.log(df['High'] / df['Low'])) ** 2
    log_co = (np.log(df['Close'] / df['Open'])) ** 2
    rs = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(rs.rolling(window=window).mean() * 252)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def compute_atr(df, period=14):
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def extract_stock_features(symbol, file_path):
    df = pd.read_parquet(file_path).sort_index()
    if len(df) < 250:
        return None  # Drop tickers with insufficient history for 200-day filters

    feat = pd.DataFrame(index=df.index)
    feat['symbol'] = symbol

    close = df['Close']
    open_p = df['Open']
    high = df['High']
    low = df['Low']
    vol = df['Volume']

    # --- 1. Multi-Horizon Returns & Price Relative to Moving Averages ---
    for w in [2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 200]:
        feat[f'ret_{w}d'] = close.pct_change(w).astype('float32')
        sma = close.rolling(w).mean()
        feat[f'dist_sma_{w}'] = ((close / (sma + 1e-9)) - 1.0).astype('float32')
        ema = close.ewm(span=w, adjust=False).mean()
        feat[f'dist_ema_{w}'] = ((close / (ema + 1e-9)) - 1.0).astype('float32')

    # --- 2. Momentum & Oscillators ---
    for rsi_len in [7, 10, 14, 21, 28]:
        feat[f'rsi_{rsi_len}'] = compute_rsi(close, rsi_len).astype('float32')
    
    # MACD Variations
    for fast, slow, sig in [(8, 21, 5), (12, 26, 9), (16, 36, 12)]:
        fast_ema = close.ewm(span=fast, adjust=False).mean()
        slow_ema = close.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=sig, adjust=False).mean()
        feat[f'macd_hist_{fast}_{slow}'] = ((macd_line - signal_line) / (close + 1e-9)).astype('float32')

    # --- 3. Volatility & Statistical Moments ---
    for w in [10, 20, 40, 60]:
        feat[f'gk_vol_{w}'] = compute_garman_klass_vol(df, w).astype('float32')
        feat[f'ret_std_{w}'] = close.pct_change().rolling(w).std().astype('float32')
        feat[f'skew_{w}'] = close.pct_change().rolling(w).skew().astype('float32')
        feat[f'kurt_{w}'] = close.pct_change().rolling(w).kurt().astype('float32')
        
        # Bollinger Band Width
        b_mean = close.rolling(w).mean()
        b_std = close.rolling(w).std()
        feat[f'bb_width_{w}'] = ((2 * b_std * 2) / (b_mean + 1e-9)).astype('float32')
        feat[f'bb_pos_{w}'] = ((close - (b_mean - 2 * b_std)) / (4 * b_std + 1e-9)).astype('float32')

    atr14 = compute_atr(df, 14)
    feat['norm_atr_14'] = (atr14 / (close + 1e-9)).astype('float32')
    feat['norm_atr_28'] = (compute_atr(df, 28) / (close + 1e-9)).astype('float32')

    # --- 4. Volume Dynamics & Liquidity ---
    for w in [5, 10, 20, 50]:
        vol_mean = vol.rolling(w).mean()
        vol_std = vol.rolling(w).std()
        feat[f'vol_zscore_{w}'] = ((vol - vol_mean) / (vol_std + 1e-9)).astype('float32')
        feat[f'vol_ratio_{w}'] = (vol / (vol_mean + 1e-9)).astype('float32')

    # Chaikin Money Flow & Money Flow Index proxies
    mf_mult = ((close - low) - (high - close)) / ((high - low) + 1e-9)
    mf_vol = mf_mult * vol
    for w in [14, 28]:
        feat[f'cmf_{w}'] = (mf_vol.rolling(w).sum() / (vol.rolling(w).sum() + 1e-9)).astype('float32')

    # --- 5. Microstructure & Gap Dynamics ---
    feat['gap_open_pct'] = ((open_p - close.shift(1)) / (close.shift(1) + 1e-9)).astype('float32')
    feat['gap_to_atr'] = ((open_p - close.shift(1)) / (atr14 + 1e-9)).astype('float32')
    feat['bar_range_to_atr'] = ((high - low) / (atr14 + 1e-9)).astype('float32')
    feat['upper_shadow_ratio'] = ((high - np.maximum(open_p, close)) / ((high - low) + 1e-9)).astype('float32')
    feat['lower_shadow_ratio'] = ((np.minimum(open_p, close) - low) / ((high - low) + 1e-9)).astype('float32')

    # --- 6. Target Label (Volatility-Adjusted Forward Return) ---
    # Realistic execution: Buy at next day's Open (t+1), exit at t+1+FORWARD_HORIZON Close
    fwd_exec_price = open_p.shift(-1)
    fwd_exit_price = close.shift(-(1 + FORWARD_HORIZON))
    raw_fwd_ret = (fwd_exit_price / fwd_exec_price) - 1.0

    # Risk-adjusted target
    gk_vol = feat['gk_vol_20'].replace(0, np.nan)
    feat['target_fwd_risk_adj'] = (raw_fwd_ret / (gk_vol * np.sqrt(FORWARD_HORIZON / 252.0))).astype('float32')
    feat['target_raw_ret'] = raw_fwd_ret.astype('float32')

    return feat

def build_universe_feature_store():
    files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    all_frames = []
    
    print(f"Building feature store from {len(files)} equity records...")
    for f in files:
        sym = os.path.basename(f).replace(".parquet", "")
        extracted = extract_stock_features(sym, f)
        if extracted is not None:
            all_frames.append(extracted)

    full_df = pd.concat(all_frames, axis=0).reset_index()
    full_df.rename(columns={'index': 'Date'}, inplace=True)
    full_df.dropna(subset=['target_fwd_risk_adj'], inplace=True)

    # Cross-Sectional Ranking Target: convert target to percentile rank [0, 1] per date
    full_df['target_rank'] = full_df.groupby('Date')['target_fwd_risk_adj'].rank(pct=True).astype('float32')

    # Save final optimized feature matrix
    full_df.to_parquet(OUTPUT_FILE, engine='pyarrow', compression='snappy')
    print(f"Engineered feature dataset successfully written to {OUTPUT_FILE}")
    print(f"Dimensions: {full_df.shape[0]} rows, {full_df.shape[1]} columns.")

if __name__ == "__main__":
    build_universe_feature_store()
