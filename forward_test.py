"""Forward test (paper-trading) engine for the Momentum v8 strategy.

Replaces forward_test_dashboard.py, which fed the PPO agent an observation it was never
trained on (leaked label columns, zero-padded), used a softmax instead of the training
normalisation, and read a signal file that can never be current.

What one daily run does
  1. Load market data, verify it is fresh and complete (else refuse, exit code 2).
  2. Mark the paper portfolio to market for every new trading day, and execute any pending
     rebalance at the NEXT SESSION'S OPEN, charging costs on real stock-level trades.
  3. If >= REBALANCE_FREQ trading days have passed since the last decision: score the universe
     with the frozen LightGBM model, rebuild the exact 43-dim training observation, run the PPO
     policy, and stage new target weights + an order sheet.
  4. Persist state (forward_test/state.json) and append to the daily / rebalance logs.

Timing convention (matches how labels were built in feature_pipeline.py): decide on the close
of day D using data up to D; execute at the open of D+1. Live, place orders as AMO / pre-open.

Portfolio values are tracked by daily adjusted-price ratios, which makes splits, bonuses and
dividends economically neutral (the download uses auto_adjust=True). Dividends are therefore
treated as reinvested; LiquidBees trading costs on the cash sleeve are ignored.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

import config
import live_scoring as ls

STATE_VERSION = 1
CASH_YIELD_DAILY = config.CASH_YIELD_ANNUAL / 252.0
FLAT = 1e-4          # weights/values below this are treated as "not held"
ADV_FLAG = 0.05      # flag a target position larger than 5% of 20d average traded value


# ----------------------------------------------------------------------------------------
# Policy <-> portfolio mapping. These two functions MUST mirror trading_env.py exactly;
# test_forward_parity.py verifies that against the real environment.
# ----------------------------------------------------------------------------------------
def build_observation(sorted_alphas, last_slot_weights, drawdown, breadth, top_k=config.TOP_K):
    """[top-k alpha scores (rank order, zero padded) | previous slot weights (k+1) | drawdown | breadth]"""
    alphas = np.zeros(top_k, dtype=np.float32)
    a = np.asarray(sorted_alphas, dtype=np.float64)[:top_k]
    alphas[:len(a)] = a
    obs = np.concatenate([
        alphas,
        np.asarray(last_slot_weights, dtype=np.float32),
        np.array([drawdown, breadth], dtype=np.float32),
    ])
    return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def apply_action_to_weights(action, top_k=config.TOP_K, cap=config.MAX_ASSET_WEIGHT):
    """Raw PPO action -> slot weights (top_k assets + cash), identical to PortfolioAllocationEnv.step:
    clip [0,1] -> divide by sum (NOT softmax) -> cap each asset, push excess to cash -> renormalise."""
    action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
    total = np.sum(action)
    if total == 0:
        w = np.zeros(top_k + 1, dtype=np.float32)
        w[-1] = 1.0
    else:
        w = action / total
    asset = w[:-1]
    excess = np.sum(np.maximum(asset - cap, 0.0))
    asset = np.minimum(asset, cap)
    w[:-1] = asset
    w[-1] += excess
    return w / (np.sum(w) + 1e-9)


def compute_target_slots(policy, sorted_alphas, last_slot_weights, drawdown, breadth):
    obs = build_observation(sorted_alphas, last_slot_weights, drawdown, breadth)
    expected = int(policy.observation_space.shape[0])
    if obs.shape[0] != expected:  # never pad/truncate silently again
        raise RuntimeError(f"Observation has {obs.shape[0]} dims but policy expects {expected}.")
    action, _ = policy.predict(obs, deterministic=True)
    return apply_action_to_weights(action), action, obs


def load_policy(path=config.PPO_MODEL_PATH):
    from stable_baselines3 import PPO
    return PPO.load(path, device="cpu")


# ----------------------------------------------------------------------------------------
# Price helpers
# ----------------------------------------------------------------------------------------
def _row_at(df, d):
    """(open, close) if the symbol traded and has valid prices on d, else None (halted/missing)."""
    i = df.index.get_indexer([d])[0]
    if i < 0:
        return None
    o, c = float(df["Open"].iloc[i]), float(df["Close"].iloc[i])
    if not (math.isfinite(o) and math.isfinite(c) and o > 0 and c > 0):
        return None
    return o, c


def _last_close(df, d):
    i = df.index.searchsorted(d, side="right") - 1
    return float(df["Close"].iloc[i]) if i >= 0 else float("nan")


def trading_calendar(panel, since, until):
    """Dates on which at least half the universe traded (ignores one-off junk dates)."""
    cnt = Counter()
    for df in panel.values():
        cnt.update(df.index[df.index >= since])
    need = 0.5 * len(panel)
    return sorted(d for d, c in cnt.items() if c >= need and d <= until)


# ----------------------------------------------------------------------------------------
# Portfolio accounting
# ----------------------------------------------------------------------------------------
def portfolio_nav(state):
    return float(state["cash"] + sum(state["positions"].values()))


def new_state(capital, inception_date):
    return {
        "version": STATE_VERSION,
        "inception_date": inception_date.date().isoformat(),
        "initial_capital": capital,
        "cash": capital,
        "positions": {},                       # symbol -> INR value at last processed close
        "peak_value": capital,
        "last_processed_date": inception_date.date().isoformat(),
        "last_decision_date": None,
        "last_slot_weights": [0.0] * config.TOP_K + [1.0],   # env starts fully in cash
        "pending": None,
    }


def execute_rebalance(state, panel, d, pend):
    """Trade from current stock-level values to the pending target at the OPEN of d.
    Costs are charged on the actual notional bought and sold (per leg)."""
    pos, cash = state["positions"], state["cash"]
    targets = pend["target_weights"]
    nav_open = cash + sum(pos.values())
    sells, buys, skipped = {}, {}, []
    for s in set(pos) | set(targets):
        df = panel.get(s)
        if df is None or _row_at(df, d) is None:      # halted / no bar: cannot trade today
            skipped.append(s)
            continue
        cur = pos.get(s, 0.0)
        delta = targets.get(s, 0.0) * nav_open - cur
        if delta < -1e-9:
            sells[s] = min(-delta, cur)
        elif delta > 1e-9:
            buys[s] = delta

    sell_total = sum(sells.values())
    cash_after_sells = cash + sell_total * (1.0 - config.SELL_LEG_COST)
    need = sum(buys.values()) * (1.0 + config.BUY_LEG_COST)
    scale = 1.0 if need <= cash_after_sells else max(cash_after_sells, 0.0) / need
    buys = {s: v * scale for s, v in buys.items()}
    buy_total = sum(buys.values())

    for s, v in sells.items():
        pos[s] -= v
        if pos[s] < 1.0:
            del pos[s]
    for s, v in buys.items():
        pos[s] = pos.get(s, 0.0) + v
    state["cash"] = max(cash_after_sells - buy_total * (1.0 + config.BUY_LEG_COST), 0.0)

    cost = sell_total * config.SELL_LEG_COST + buy_total * config.BUY_LEG_COST
    return {
        "decision_date": pend["decision_date"], "exec_date": d.date().isoformat(),
        "slot_turnover": pend["slot_turnover"],
        "stock_turnover": (sell_total + buy_total) / 2.0 / nav_open if nav_open else 0.0,
        "cost_inr": cost, "cost_pct_nav": cost / nav_open if nav_open else 0.0,
        "n_positions": len(pos), "target_cash_weight": pend["target_cash_weight"],
        "skipped": ",".join(sorted(skipped)), "health": pend.get("health", ""),
        "holdings": ";".join(f"{s}:{w:.3f}" for s, w in sorted(targets.items(), key=lambda x: -x[1])),
    }


def advance_day(state, panel, prev_d, d):
    """Roll the portfolio from the close of prev_d to the close of d."""
    pos, info = state["positions"], {"executed": None, "halted": []}
    # close(prev) -> open(d): overnight gap
    for s in list(pos):
        df = panel.get(s)
        bar = _row_at(df, d) if df is not None else None
        pc = _last_close(df, prev_d) if df is not None else float("nan")
        if bar is None or not math.isfinite(pc) or pc <= 0:
            info["halted"].append(s)
            continue
        pos[s] *= bar[0] / pc
    # pending rebalance executes at the open of the first session after the decision
    pend = state.get("pending")
    if pend and pd.Timestamp(pend["decision_date"]) < d:
        info["executed"] = execute_rebalance(state, panel, d, pend)
        state["pending"] = None
    # open(d) -> close(d)
    for s in list(pos):
        df = panel.get(s)
        bar = _row_at(df, d) if df is not None else None
        if bar is not None:
            pos[s] *= bar[1] / bar[0]
    state["cash"] *= 1.0 + CASH_YIELD_DAILY
    nav = portfolio_nav(state)
    state["peak_value"] = max(state["peak_value"], nav)
    state["last_processed_date"] = d.date().isoformat()
    return info


# ----------------------------------------------------------------------------------------
# Decision
# ----------------------------------------------------------------------------------------
def decide(state, panel, latest, scores, policy, health):
    nav = portfolio_nav(state)
    drawdown = (state["peak_value"] - nav) / state["peak_value"]
    top = scores.head(config.TOP_K)
    breadth = float(scores["alpha_score"].mean())
    slot_w, action, _ = compute_target_slots(
        policy, top["alpha_score"].values, np.array(state["last_slot_weights"]), drawdown, breadth)

    targets = {sym: float(slot_w[i]) for i, sym in enumerate(top["symbol"]) if slot_w[i] > FLAT}
    slot_turnover = float(np.abs(slot_w - np.array(state["last_slot_weights"])).sum() / 2.0)
    state["pending"] = {
        "decision_date": latest.date().isoformat(),
        "target_weights": targets,
        "slot_turnover": slot_turnover,
        "target_cash_weight": float(1.0 - sum(targets.values())),
        "health": "; ".join(health["reasons"]) if health else "",
    }
    state["last_slot_weights"] = [float(x) for x in slot_w]
    state["last_decision_date"] = latest.date().isoformat()
    return build_order_sheet(state, panel, scores, targets, nav, latest)


def build_order_sheet(state, panel, scores, targets, nav, latest):
    pos = state["positions"]
    adv = scores.set_index("symbol")["adv20_value"].to_dict()
    ref_px = scores.set_index("symbol")["Close"].to_dict()
    rows = []
    for s in sorted(set(pos) | set(targets)):
        cur_w = pos.get(s, 0.0) / nav
        tgt_w = targets.get(s, 0.0)
        delta = tgt_w - cur_w
        px = ref_px.get(s)
        if px is None and s in panel:
            px = _last_close(panel[s], latest)
        if tgt_w <= FLAT and cur_w > FLAT:
            act = "SELL ALL"
        elif cur_w <= FLAT and tgt_w > FLAT:
            act = "BUY NEW"
        elif delta > 0.005:
            act = "ADD"
        elif delta < -0.005:
            act = "TRIM"
        else:
            act = "HOLD"
        value = abs(delta) * nav
        a = adv.get(s, float("nan"))
        rows.append({
            "decision_date": latest.date().isoformat(), "symbol": s, "action": act,
            "current_pct": round(cur_w * 100, 3), "target_pct": round(tgt_w * 100, 3),
            "delta_pct": round(delta * 100, 3), "ref_close": px,
            "est_shares": int(value // px) if px else None, "est_value_inr": round(value),
            "adv20_value_inr": a, "target_pct_of_adv": (tgt_w * nav / a) if a and a > 0 else float("nan"),
            "liquidity_flag": bool(a and a > 0 and tgt_w * nav / a > ADV_FLAG),
            "execute": "next session open (AMO / pre-open)",
        })
    order = {"SELL ALL": 0, "TRIM": 1, "BUY NEW": 2, "ADD": 3, "HOLD": 4}
    return pd.DataFrame(rows).sort_values(["action", "symbol"], key=lambda c: c.map(order) if c.name == "action" else c)


def is_rebalance_day(state, panel, latest):
    if state["last_decision_date"] is None:
        return True
    last = pd.Timestamp(state["last_decision_date"])
    return len(trading_calendar(panel, last + pd.Timedelta(days=1), latest)) >= config.REBALANCE_FREQ


# ----------------------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------------------
def _paths(state_dir):
    j = lambda n: os.path.join(state_dir, n)
    return {k: j(v) for k, v in dict(state="state.json", daily="daily_log.csv", rebal="rebalance_log.csv",
                                     orders="execution_orders.csv", target="target_portfolio.json",
                                     summary="summary.md").items()}


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _append_csv(path, rows, key):
    new = pd.DataFrame(rows)
    if new.empty:
        return
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=key, keep="last")
    new.to_csv(path, index=False)


def fetch_benchmark(start, end):
    """Best-effort Nifty 500 (fallback Nifty 50) closes for comparison. Never affects trading."""
    try:
        import yfinance as yf
        for tk in ("^CRSLDX", "^NSEI"):
            h = yf.download(tk, start=start, end=end + pd.Timedelta(days=1), progress=False, auto_adjust=True)
            if isinstance(h.columns, pd.MultiIndex):
                h.columns = [c[0] for c in h.columns]
            if not h.empty:
                return h["Close"]
    except Exception as e:
        print(f"NOTE: benchmark unavailable ({e}); continuing without it.")
    return None


def write_summary(paths, state, latest, notes):
    nav = portfolio_nav(state)
    cap = state["initial_capital"]
    lines = [f"# Forward test - {latest.date()}", "",
             f"**NAV:** INR {nav:,.0f}  |  **Since inception ({state['inception_date']}):** {nav / cap - 1:+.2%}  |  "
             f"**Drawdown:** {(state['peak_value'] - nav) / state['peak_value']:.2%}  |  "
             f"**Cash:** {state['cash'] / nav:.1%}", ""]
    if os.path.exists(paths["daily"]):
        d = pd.read_csv(paths["daily"])
        if len(d) > 1 and d["benchmark_close"].notna().sum() > 1:
            b = d["benchmark_close"].dropna()
            first_nav = d.loc[b.index[0], "nav"]
            lines += [f"**Benchmark since first data point:** {b.iloc[-1] / b.iloc[0] - 1:+.2%} vs strategy "
                      f"{nav / first_nav - 1:+.2%}", ""]
        r = d["daily_return"].dropna()
        if len(r) >= 20:
            lines += [f"**Ann. vol:** {r.std() * math.sqrt(252):.1%}  |  **Days live:** {len(d)}", ""]
    for n in notes:
        lines.append(f"> {n}")
    if state["positions"]:
        lines += ["", "### Holdings", "| Symbol | Weight |", "|---|---|"]
        lines += [f"| {s} | {v / nav:.2%} |" for s, v in sorted(state["positions"].items(), key=lambda x: -x[1])]
    if state.get("pending"):
        lines += ["", f"### Pending rebalance (decided {state['pending']['decision_date']}, executes next open)"]
        if os.path.exists(paths["orders"]):
            o = pd.read_csv(paths["orders"])
            lines += ["| Action | Symbol | Current | Target | Est. shares | Liquidity flag |", "|---|---|---|---|---|---|"]
            lines += [f"| {r.action} | {r.symbol} | {r.current_pct:.2f}% | {r.target_pct:.2f}% | {r.est_shares} | "
                      f"{'YES' if r.liquidity_flag else ''} |" for r in o.itertuples() if r.action != "HOLD"]
    text = "\n".join(lines)
    with open(paths["summary"], "w") as f:
        f.write(text)
    gh = os.getenv("GITHUB_STEP_SUMMARY")
    if gh:
        with open(gh, "a", encoding="utf-8") as f:
            f.write(text + "\n")


# ----------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------
def run(state_dir=config.FORWARD_TEST_DIR, asof=None, allow_stale=False, panel=None, scorer=None,
        policy=None, check_health=True, use_benchmark=True, today=None):
    os.makedirs(state_dir, exist_ok=True)
    paths = _paths(state_dir)
    if panel is None:
        panel = ls.load_market_panel(asof=asof)
    latest, coverage = ls.determine_latest_date(panel, today=today, allow_stale=allow_stale or asof is not None)
    notes, exit_code = [], 0

    if os.path.exists(paths["state"]):
        with open(paths["state"]) as f:
            state = json.load(f)
        if state.get("version") != STATE_VERSION:
            raise RuntimeError(f"Unsupported state version {state.get('version')}")
        if pd.Timestamp(state["last_processed_date"]) > latest:
            raise RuntimeError(f"State is at {state['last_processed_date']} but data ends {latest.date()}; "
                               "refusing to move backwards.")
        fresh = False
    else:
        state, fresh = new_state(config.INITIAL_CAPITAL, latest), True
        print(f"Initialised forward test at {latest.date()} with INR {state['initial_capital']:,.0f}")

    daily_rows, rebal_rows = [], []
    last_proc = pd.Timestamp(state["last_processed_date"])
    if fresh:
        daily_rows.append({"date": latest.date().isoformat(), "nav": portfolio_nav(state), "daily_return": np.nan,
                           "drawdown": 0.0, "cash_weight": 1.0, "n_positions": 0, "executed": 0,
                           "stock_turnover": np.nan, "cost_inr": 0.0, "halted": 0, "benchmark_close": np.nan})

    new_days = [d for d in trading_calendar(panel, last_proc + pd.Timedelta(days=1), latest)]
    for d in new_days:
        prev_nav = portfolio_nav(state)
        info = advance_day(state, panel, pd.Timestamp(state["last_processed_date"]), d)
        nav = portfolio_nav(state)
        ex = info["executed"]
        if ex:
            rebal_rows.append(ex)
        daily_rows.append({
            "date": d.date().isoformat(), "nav": nav, "daily_return": nav / prev_nav - 1.0,
            "drawdown": (state["peak_value"] - nav) / state["peak_value"], "cash_weight": state["cash"] / nav,
            "n_positions": len(state["positions"]), "executed": int(ex is not None),
            "stock_turnover": ex["stock_turnover"] if ex else np.nan, "cost_inr": ex["cost_inr"] if ex else 0.0,
            "halted": len(info["halted"]), "benchmark_close": np.nan})
        if info["halted"]:
            notes.append(f"{d.date()}: no trade/price for held {','.join(info['halted'])} (value carried flat)")

    if use_benchmark and asof is None and daily_rows:
        bench = fetch_benchmark(pd.Timestamp(daily_rows[0]["date"]) - pd.Timedelta(days=5), latest)
        if bench is not None:
            bench.index = pd.to_datetime(bench.index).tz_localize(None)
            for r in daily_rows:
                r["benchmark_close"] = float(bench.asof(pd.Timestamp(r["date"])))

    decision_orders = None
    if state["last_decision_date"] == latest.date().isoformat():
        notes.append(f"Decision for {latest.date()} already made (idempotent re-run).")
    elif is_rebalance_day(state, panel, latest):
        if scorer is None:
            scorer = lambda p, d: ls.score_universe(p, d)
        scores, skipped = scorer(panel, latest)
        health = None
        if check_health:
            health = ls.check_signal_health(scores, ls.reference_score_bounds())
        if health and not health["ok"]:
            exit_code = 3
            notes.append("SIGNAL HEALTH FAIL - no rebalance staged, holding current portfolio: "
                         + "; ".join(health["reasons"]))
        else:
            if policy is None:
                policy = load_policy()
            decision_orders = decide(state, panel, latest, scores, policy, health)
            notes.append(f"Rebalance staged for next open ({len(state['pending']['target_weights'])} names, "
                         f"cash {state['pending']['target_cash_weight']:.1%}); universe scored: {len(scores)}, "
                         f"skipped: {skipped}")
    else:
        notes.append("Not a rebalance day - holding.")

    # persist (state last, atomically, so a crash never leaves a half-updated portfolio)
    _append_csv(paths["daily"], daily_rows, ["date"])
    _append_csv(paths["rebal"], rebal_rows, ["decision_date", "exec_date"])
    if decision_orders is not None:
        decision_orders.to_csv(paths["orders"], index=False)
        _atomic_write_json(paths["target"], {"decision_date": latest.date().isoformat(),
                                             **state["pending"]["target_weights"],
                                             "CASH": state["pending"]["target_cash_weight"]})
    _atomic_write_json(paths["state"], state)
    write_summary(paths, state, latest, notes)
    for n in notes:
        print(n)
    print(f"NAV {portfolio_nav(state):,.0f} as of {latest.date()} (data coverage {coverage:.1%})")
    return exit_code


def main():
    ap = argparse.ArgumentParser(description="Momentum v8 forward test (paper trading)")
    ap.add_argument("--state-dir", default=config.FORWARD_TEST_DIR)
    ap.add_argument("--asof", help="point-in-time replay: ignore data after this date (testing)")
    ap.add_argument("--allow-stale", action="store_true")
    a = ap.parse_args()
    try:
        code = run(state_dir=a.state_dir, asof=a.asof, allow_stale=a.allow_stale)
    except ls.DataIntegrityError as e:
        print(f"DATA INTEGRITY ERROR - no action taken: {e}", file=sys.stderr)
        sys.exit(2)
    sys.exit(code)


if __name__ == "__main__":
    main()
