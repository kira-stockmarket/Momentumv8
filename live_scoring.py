"""Live alpha scoring.

Takes the latest OHLCV files in data/, builds features with the SAME code used in training
(feature_pipeline.build_feature_frame, include_targets=False), and scores every symbol with
the frozen best_lightgbm_ranker.joblib.

This is the piece that was missing: oos_predictions.parquet can never contain the most recent
~11 trading days (labels need 10 future days), so it cannot be a live signal source.

Only information available at the close of the latest bar is used.
"""
import glob
import os

import joblib
import numpy as np
import pandas as pd

import config
from feature_pipeline import build_feature_frame

# ~6 years of history: EMA(200) recursion has decayed to ~1e-7 of its seed by then, so the
# truncated history reproduces full-history features to float precision.
TAIL_ROWS = 2000
MIN_HISTORY = 250  # same minimum as training (feature_pipeline.process_single_stock)


class DataIntegrityError(RuntimeError):
    """Raised when market data is stale/incomplete. The caller must NOT trade on it."""


def load_market_panel(data_dir=config.DATA_DIR, asof=None):
    """Returns {symbol: OHLCV frame (date-sorted, forward-filled)}. asof=None -> everything;
    asof=Timestamp -> strictly point-in-time slice (used for testing/replay)."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise DataIntegrityError(f"No parquet files found in '{data_dir}'.")
    panel = {}
    for f in files:
        sym = os.path.basename(f)[:-len(".parquet")]
        try:
            df = pd.read_parquet(f).sort_index()
        except Exception as e:  # corrupt/partial file must not take the whole run down silently
            print(f"WARNING: could not read {sym}: {e}")
            continue
        if asof is not None:
            df = df[df.index <= pd.Timestamp(asof)]
        if df.empty:
            continue
        panel[sym] = df.ffill()
    return panel


def determine_latest_date(panel, today=None, allow_stale=False):
    """Latest bar date plus integrity checks (coverage + staleness)."""
    last_dates = pd.Series({s: df.index.max() for s, df in panel.items()})
    latest = last_dates.max()
    # Only count symbols that are actually still trading (a bar in the last 30 days).
    live = last_dates[last_dates >= latest - pd.Timedelta(days=30)]
    coverage = float((live == latest).mean())
    if coverage < config.MIN_SYMBOL_COVERAGE:
        raise DataIntegrityError(
            f"Only {coverage:.1%} of live symbols have a bar on {latest.date()} "
            f"(need >= {config.MIN_SYMBOL_COVERAGE:.0%}). Likely a partial/failed download.")
    if not allow_stale:
        today = pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
        age = (today - latest.normalize()).days
        if age > config.MAX_DATA_AGE_DAYS:
            raise DataIntegrityError(
                f"Latest market data is {age} days old ({latest.date()}); limit is "
                f"{config.MAX_DATA_AGE_DAYS}. Refusing to generate a signal from stale data.")
    return latest, coverage


def score_universe(panel, latest_date, model_path=config.LGBM_MODEL_PATH):
    """Scores every symbol that has a bar on latest_date. Returns a DataFrame:
    Date, symbol, alpha_score, Close, Open, adv20_value (INR traded value, 20d mean)."""
    rows, skipped = [], {"no_bar": 0, "short_history": 0, "nan_features": 0}
    for sym, df in panel.items():
        if df.index.max() != latest_date:
            skipped["no_bar"] += 1
            continue
        if len(df) < MIN_HISTORY:
            skipped["short_history"] += 1
            continue
        feat = build_feature_frame(df.tail(TAIL_ROWS), include_targets=False)
        last = feat.iloc[-1].replace([np.inf, -np.inf], np.nan)
        if last.isna().any():
            skipped["nan_features"] += 1
            continue
        adv = float((df["Close"] * df["Volume"]).tail(20).mean())
        rows.append((sym, last, float(df["Close"].iloc[-1]), float(df["Open"].iloc[-1]), adv))
    if not rows:
        raise DataIntegrityError("No symbol produced a valid feature row.")

    X = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])
    model = joblib.load(model_path)
    needed = list(model.feature_name())
    missing = [c for c in needed if c not in X.columns]
    if missing:
        raise DataIntegrityError(f"Feature mismatch vs trained model; missing: {missing[:5]}...")
    X = X[needed].astype("float32")  # exact training column order + dtype
    scores = model.predict(X)
    if not np.isfinite(scores).all():
        raise DataIntegrityError("Model produced non-finite scores.")

    out = pd.DataFrame({
        "Date": latest_date,
        "symbol": X.index.values,
        "alpha_score": scores.astype("float64"),
        "Close": [r[2] for r in rows],
        "Open": [r[3] for r in rows],
        "adv20_value": [r[4] for r in rows],
    })
    return out.sort_values("alpha_score", ascending=False).reset_index(drop=True), skipped


def reference_score_bounds(oos_path=config.OOS_PREDICTIONS_PATH, last_n_dates=750, margin=0.25):
    """The PPO agent was trained on alpha scores from the walk-forward FOLD models, but live
    scores come from the final model trained on all data. LambdaRank scores are only
    rank-meaningful, so their scale can differ. This derives the range the agent has actually
    seen (per-date top-20 mean and market-breadth mean) so drift can be detected."""
    o = pd.read_parquet(oos_path, columns=["Date", "alpha_score"])
    g = o.sort_values("alpha_score", ascending=False).groupby("Date")["alpha_score"]
    top20 = g.apply(lambda s: s.head(config.TOP_K).mean())
    breadth = o.groupby("Date")["alpha_score"].mean()
    ref = pd.DataFrame({"top20_mean": top20, "breadth": breadth}).sort_index().tail(last_n_dates)
    bounds = {}
    for c in ref.columns:
        lo, hi = float(ref[c].min()), float(ref[c].max())
        pad = (hi - lo) * margin
        bounds[c] = (lo - pad, hi + pad, lo, hi)
    return bounds


def check_signal_health(scores, bounds):
    """Compares live score statistics against what the PPO agent saw in training."""
    top20_mean = float(scores["alpha_score"].head(config.TOP_K).mean())
    breadth = float(scores["alpha_score"].mean())
    report = {"top20_mean": top20_mean, "breadth": breadth, "ok": True, "reasons": []}
    for name, val in (("top20_mean", top20_mean), ("breadth", breadth)):
        lo, hi, rlo, rhi = bounds[name]
        if not (lo <= val <= hi):
            report["ok"] = False
            report["reasons"].append(
                f"{name}={val:.4f} outside range seen in training [{rlo:.4f}, {rhi:.4f}] (+25% margin)")
    return report


if __name__ == "__main__":
    panel = load_market_panel()
    latest, cov = determine_latest_date(panel, allow_stale=True)
    scores, skipped = score_universe(panel, latest)
    print(f"Latest bar: {latest.date()} | coverage {cov:.1%} | scored {len(scores)} | skipped {skipped}")
    print(scores.head(10).to_string(index=False))
    print(check_signal_health(scores, reference_score_bounds()))
