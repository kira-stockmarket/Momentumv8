import os
import glob
import numpy as np
import pandas as pd

INPUT_DIR = "data"
OUTPUT_DIR = "features"
FORWARD_HORIZON = 10

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_garman_klass_vol(df, window=20):
    log_hl = (np.log((df['High'] + 1e-9) / (df['Low'] + 1e-9))) ** 2
    log_co = (np.log((df['Close'] + 1e-9) / (df['Open'] + 1e-9))) ** 2
    rs = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(np.maximum(rs.rolling(window=window).mean() * 252, 0.0))

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

def process_single_stock(symbol, file_path):
    df = pd.read_parquet(file_path).sort_index()
    df = df.ffill()
    
    if len(df) < 250:
        return 

    close = df['Close']
    open_p = df['Open']
    high = df['High']
    low = df['Low']
    vol = df['Volume']

    # Pre-allocate a dictionary to prevent DataFrame memory fragmentation
    f_dict = {}

    # --- Feature Block 1: Trend & Moving Average Distances ---
    windows = [3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 200]
    for w in windows:
        f_dict[f'ret_{w}d'] = close.pct_change(w)
        sma = close.rolling(w).mean()
        f_dict[f'dist_sma_{w}'] = (close / (sma + 1e-9)) - 1.0
        ema = close.ewm(span=w, adjust=False).mean()
        f_dict[f'dist_ema_{w}'] = (close / (ema + 1e-9)) - 1.0
        f_dict[f'price_rank_{w}d'] = close.rolling(w).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)

    # --- Feature Block 2: Momentum Oscillators ---
    rsi_windows = [7, 10, 14, 21, 28, 40, 60]
    for r in rsi_windows:
        f_dict[f'rsi_{r}'] = compute_rsi(close, r)
        f_dict[f'rsi_roc_{r}'] = f_dict[f'rsi_{r}'].diff(3)

    macd_configs = [(8, 21, 5), (12, 26, 9), (16, 36, 12), (20, 50, 15)]
    for fast, slow, sig in macd_configs:
        fast_ema = close.ewm(span=fast, adjust=False).mean()
        slow_ema = close.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=sig, adjust=False).mean()
        f_dict[f'macd_hist_{fast}_{slow}'] = (macd_line - signal_line) / (close + 1e-9)

    # --- Feature Block 3: Volatility & Statistical Moments ---
    vol_windows = [10, 20, 40, 60, 90]
    for w in vol_windows:
        f_dict[f'gk_vol_{w}'] = compute_garman_klass_vol(df, w)
        f_dict[f'ret_std_{w}'] = close.pct_change().rolling(w).std()
        f_dict[f'skew_{w}'] = close.pct_change().rolling(w).skew()
        f_dict[f'kurt_{w}'] = close.pct_change().rolling(w).kurt()
        
        b_mean = close.rolling(w).mean()
        b_std = close.rolling(w).std()
        f_dict[f'bb_width_{w}'] = (2 * b_std * 2) / (b_mean + 1e-9)
        f_dict[f'bb_pos_{w}'] = (close - (b_mean - 2 * b_std)) / (4 * b_std + 1e-9)

    atr_windows = [7, 14, 28]
    for w in atr_windows:
        atr = compute_atr(df, w)
        f_dict[f'norm_atr_{w}'] = atr / (close + 1e-9)

    # --- Feature Block 4: Volume & Liquidity Dynamics ---
    for w in [5, 10, 20, 50, 100]:
        vol_mean = vol.rolling(w).mean()
        vol_std = vol.rolling(w).std()
        f_dict[f'vol_zscore_{w}'] = (vol - vol_mean) / (vol_std + 1e-9)
        f_dict[f'vol_ratio_{w}'] = vol / (vol_mean + 1e-9)

    mf_mult = ((close - low) - (high - close)) / ((high - low) + 1e-9)
    mf_vol = mf_mult * vol
    for w in [14, 28, 50]:
        f_dict[f'cmf_{w}'] = mf_vol.rolling(w).sum() / (vol.rolling(w).sum() + 1e-9)

    # --- Feature Block 5: Microstructure ---
    atr14 = compute_atr(df, 14)
    f_dict['gap_open_pct'] = (open_p - close.shift(1)) / (close.shift(1) + 1e-9)
    f_dict['gap_to_atr'] = (open_p - close.shift(1)) / (atr14 + 1e-9)
    f_dict['bar_range_to_atr'] = (high - low) / (atr14 + 1e-9)
    f_dict['upper_shadow_ratio'] = (high - np.maximum(open_p, close)) / ((high - low) + 1e-9)
    f_dict['lower_shadow_ratio'] = (np.minimum(open_p, close) - low) / ((high - low) + 1e-9)

    # --- Feature Block 6: Risk-Adjusted Target Label ---
    fwd_exec_price = open_p.shift(-1)
    fwd_exit_price = close.shift(-(1 + FORWARD_HORIZON))
    raw_fwd_ret = (fwd_exit_price / fwd_exec_price) - 1.0

    gk_vol = f_dict['gk_vol_20'].replace(0, np.nan)
    f_dict['target_fwd_risk_adj'] = raw_fwd_ret / (gk_vol * np.sqrt(FORWARD_HORIZON / 252.0))
    f_dict['target_raw_ret'] = raw_fwd_ret

    # --- Compile Dictionary into DataFrame (Eliminates Fragmentation) ---
    feat = pd.DataFrame(f_dict, index=df.index)

    # --- Strict Cleaning & Downcasting ---
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.dropna(inplace=True) 
    
    float_cols = feat.select_dtypes(include=['float64']).columns
    feat[float_cols] = feat[float_cols].astype('float32')

    out_path = os.path.join(OUTPUT_DIR, f"{symbol}.parquet")
    feat.to_parquet(out_path, engine='pyarrow', compression='snappy')
    print(f"Processed {symbol}: {feat.shape[0]} rows, {feat.shape[1]} features.")

def build_distributed_features():
    files = glob.glob(os.path.join(INPUT_DIR, "*.parquet"))
    print(f"Starting distributed feature engineering for {len(files)} assets...")
    
    for f in files:
        symbol = os.path.basename(f).replace(".parquet", "")
        try:
            process_single_stock(symbol, f)
        except Exception as e:
            print(f"Failed processing {symbol}: {str(e)}")

if __name__ == "__main__":
    build_distributed_features()
