import pandas as pd
import yfinance as yf
import os
import time

# Create target directory for the Parquet files
os.makedirs("data", exist_ok=True)

# Load the exact Nifty 500 list provided in the repository root
df = pd.read_csv("ind_nifty500list 2.csv")

for symbol in df['Symbol']:
    ticker = f"{symbol}.NS"
    print(f"Fetching max historical data for {ticker}...")
    
    try:
        # Download data silently to avoid cluttered logs
        data = yf.download(ticker, period="max", auto_adjust=True, progress=False)
        
        if not data.empty:
            # Flatten multi-index columns if returned by newer versions of yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            # Ensure the index is a proper DatetimeIndex for time-series alignment
            data.index = pd.to_datetime(data.index)
            
            # Explicitly cast to float32 to save memory (crucial when scaling to 200+ features)
            # Volume remains int64 or float64 depending on size, but float32 is usually safe
            data = data.astype('float32')
                
            # Save as Parquet using PyArrow with Snappy compression
            data.to_parquet(f"data/{symbol}.parquet", engine='pyarrow', compression='snappy')
            
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")
    
    # 0.5-second pause to prevent Yahoo Finance from rate-limiting the GitHub Action IP
    time.sleep(0.5)

print("Parquet data pipeline execution completed successfully.")
