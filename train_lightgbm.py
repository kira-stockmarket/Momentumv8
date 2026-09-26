import os
import glob
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import joblib
import sqlite3

# --- Configuration ---
FEATURE_DIR = "features"
OOS_OUTPUT_FILE = "oos_predictions.parquet"
DB_FILE = "sqlite:///optuna_study.db"
STUDY_NAME = "nifty500_lambdarank_119"
HORIZON_PURGE = 10
TIMEOUT_SECONDS = 19800  # 5.5 hours

def load_and_compile_data():
    """Aggregates individual asset files into a cross-sectional dataset in RAM."""
    print("Loading distributed feature files into RAM...")
    files = glob.glob(os.path.join(FEATURE_DIR, "*.parquet"))
    
    if not files:
        raise FileNotFoundError("No parquet files found in features directory.")
        
    df_list = []
    for f in files:
        sym = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f).reset_index()
        if 'index' in df.columns:
            df.rename(columns={'index': 'Date'}, inplace=True)
        df['symbol'] = sym
        df_list.append(df)
        
    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    del df_list
    gc.collect()
    
    # Sort strictly by Date then Symbol to create valid Query Groups
    full_df.sort_values(['Date', 'symbol'], inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    
    print(f"Aggregated Dataset: {full_df.shape[0]} rows.")
    return full_df

def prepare_ranking_data(df):
    """Buckets targets into cross-sectional quantiles (0 to 4) per day."""
    print("Computing daily cross-sectional quantiles...")
    df['relevance'] = df.groupby('Date')['target_fwd_risk_adj'].transform(
        lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop')
    )
    
    # Drop NaNs and strictly reset the index to prevent slice boundary errors
    df.dropna(subset=['relevance'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    exclude_cols = ['Date', 'symbol', 'target_fwd_risk_adj', 'target_raw_ret', 'relevance']
    features = [c for c in df.columns if c not in exclude_cols]
    
    return df, features

def create_purged_cv_splits(df, n_splits=5, purge_days=10):
    """Institutional Purged Walk-Forward Splits."""
    unique_dates = np.sort(df['Date'].unique())
    n_days = len(unique_dates)
    split_size = n_days // (n_splits + 1)
    splits = []
    
    for i in range(1, n_splits + 1):
        train_end_idx = i * split_size
        val_end_idx = (i + 1) * split_size if i < n_splits else n_days
        
        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[train_end_idx + purge_days : val_end_idx]
        
        train_idx = df.index[df['Date'].isin(train_dates)]
        val_idx = df.index[df['Date'].isin(val_dates)]
        splits.append((train_idx, val_idx))
        
    return splits

def objective(trial, df, features, cv_splits):
    """Optuna objective to maximize NDCG@20."""
    param = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [20],
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 1000),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.8),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        "n_estimators": 600,
        "n_jobs": -1,  
        "verbose": -1
    }
    
    cv_scores = []
    
    for train_idx, val_idx in cv_splits:
        # Changed from .iloc to .loc to map index labels perfectly
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
        
        q_train = train_df.groupby('Date').size().values
        q_val = val_df.groupby('Date').size().values
        
        train_data = lgb.Dataset(train_df[features], label=train_df['relevance'], group=q_train)
        val_data = lgb.Dataset(val_df[features], label=val_df['relevance'], group=q_val)
        
        model = lgb.train(
            param,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False)
            ]
        )
        
        cv_scores.append(model.best_score['valid_0']['ndcg@20'])
        
        del train_df, val_df, train_data, val_data, model
        gc.collect()
        
    return np.mean(cv_scores)

def generate_oos_predictions(df, features, cv_splits, best_params):
    """Trains fold-by-fold to record strictly Out-Of-Sample predictions."""
    print("Generating Out-Of-Sample (OOS) signal matrix for the RL agent...")
    oos_list = []
    
    best_params['objective'] = 'lambdarank'
    best_params['metric'] = 'ndcg'
    best_params['verbose'] = -1
    best_params['n_jobs'] = -1
    best_params['n_estimators'] = 800
    
    for train_idx, val_idx in cv_splits:
        # Changed from .iloc to .loc to map index labels perfectly
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
        
        q_train = train_df.groupby('Date').size().values
        train_data = lgb.Dataset(train_df[features], label=train_df['relevance'], group=q_train)
        
        fold_model = lgb.train(best_params, train_data)
        
        val_preds = val_df[['Date', 'symbol', 'target_fwd_risk_adj', 'target_raw_ret']].copy()
        val_preds['alpha_score'] = fold_model.predict(val_df[features])
        oos_list.append(val_preds)
        
    oos_df = pd.concat(oos_list).sort_values(['Date', 'symbol'])
    
    oos_df.to_parquet(OOS_OUTPUT_FILE, engine='pyarrow', compression='snappy')
    print(f"OOS Matrix exported: {oos_df.shape[0]} simulated execution rows.")

    print("Training final production model...")
    q_all = df.groupby('Date').size().values
    full_data = lgb.Dataset(df[features], label=df['relevance'], group=q_all)
    final_model = lgb.train(best_params, full_data)
    joblib.dump(final_model, 'best_lightgbm_ranker.joblib')
    print("Final model serialized to best_lightgbm_ranker.joblib")

if __name__ == "__main__":
    raw_df = load_and_compile_data()
    df, features = prepare_ranking_data(raw_df)
    cv_splits = create_purged_cv_splits(df, n_splits=5, purge_days=HORIZON_PURGE)
    
    print(f"Starting distributed Optuna optimization (~{TIMEOUT_SECONDS/3600:.1f} hours max)...")
    
    study = optuna.create_study(
        study_name=STUDY_NAME, 
        storage=DB_FILE, 
        load_if_exists=True,
        direction="maximize"
    )
    
    try:
        study.optimize(
            lambda trial: objective(trial, df, features, cv_splits),
            timeout=TIMEOUT_SECONDS,
            n_trials=1000
        )
    except Exception as e:
        print(f"Optimization halted early: {e}")
    
    print("\nOptimization Complete.")
    
    try:
        best_trial = study.best_trial
        print("Best Trial NDCG:", best_trial.value)
        print("Best Parameters:", best_trial.params)
        
        generate_oos_predictions(df, features, cv_splits, best_trial.params)
        
    except ValueError:
        print("Critical Error: No trials completed successfully. Check the Optuna logs above.")
