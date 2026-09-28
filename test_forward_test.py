"""Tests for the forward-test engine.  Run:  python test_forward_test.py   (or: pytest -q)

PARITY tests (need oos_predictions.parquet + best_ppo_agent.zip) prove the live decision path is
byte-for-byte the same logic the agent was trained under (trading_env.py).
ENGINE tests use a small synthetic market with hand-computable answers.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

import config
import forward_test as ft
import live_scoring as ls

HAVE_ASSETS = os.path.exists(config.OOS_PREDICTIONS_PATH) and os.path.exists(config.PPO_MODEL_PATH)


# ------------------------------------------------------------------------------ parity
def _env():
    from trading_env import PortfolioAllocationEnv
    start = open("eval_start_date.txt").read().strip() if os.path.exists("eval_start_date.txt") else None
    return PortfolioAllocationEnv(data_path=config.OOS_PREDICTIONS_PATH, top_k=config.TOP_K,
                                  rebalance_freq=config.REBALANCE_FREQ, initial_capital=1e8,
                                  tx_cost=0.0026, start_date=start)


def _env_inputs(env):
    """The raw quantities env._get_observation() consumes at its current step."""
    d = env.rebalance_dates[env.current_step]
    day = env.df[env.df["Date"] == d]
    top = day.nlargest(config.TOP_K, "alpha_score")
    dd = (env.peak_value - env.portfolio_value) / env.peak_value
    return top["alpha_score"].values, env.current_weights.copy(), dd, day["alpha_score"].mean()


def test_action_to_weights_matches_env():
    env = _env(); env.reset()
    rng = np.random.default_rng(1)
    worst = 0.0
    for i in range(80):
        a = rng.random(config.TOP_K + 1).astype(np.float32)
        a[rng.random(a.shape) < 0.5] = 0.0          # sparse actions like the trained agent's
        if i % 10 == 0:
            a[:] = 0.0                               # all-zero -> 100% cash branch
        if i % 7 == 0:
            a[:2] = 5.0                              # forces the 15% cap / excess-to-cash branch
        env.step(a)
        worst = max(worst, float(np.abs(env.current_weights - ft.apply_action_to_weights(a)).max()))
    assert worst < 1e-6, f"weight mapping differs from trading_env.py (max diff {worst})"
    print(f"  ok  action->weights parity vs env over 80 actions (max diff {worst:.1e})")


def test_observation_matches_env():
    env = _env(); obs, _ = env.reset()
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(60):
        alphas, last_w, dd, breadth = _env_inputs(env)
        worst = max(worst, float(np.abs(obs - ft.build_observation(alphas, last_w, dd, breadth)).max()))
        a = rng.random(config.TOP_K + 1).astype(np.float32) * (rng.random(config.TOP_K + 1) < 0.4)
        obs, *_ = env.step(a)
    assert worst < 1e-6, f"observation differs from trading_env.py (max diff {worst})"
    print(f"  ok  observation parity vs env over 60 steps (max diff {worst:.1e})")


def test_real_policy_decisions_match_env():
    """Drive the env with the trained agent, and independently recompute every decision with the
    forward-test functions from (alphas, previous weights, drawdown, breadth). Must be identical."""
    policy = ft.load_policy()
    env = _env(); obs, _ = env.reset()
    worst = 0.0
    for _ in range(70):
        alphas, last_w, dd, breadth = _env_inputs(env)
        mine, _, my_obs = ft.compute_target_slots(policy, alphas, last_w, dd, breadth)
        action, _ = policy.predict(obs, deterministic=True)
        obs, *_ = env.step(action)
        worst = max(worst, float(np.abs(env.current_weights - mine).max()))
    assert worst < 1e-6, f"live decision path diverges from training env (max diff {worst})"
    print(f"  ok  trained-agent decisions identical to env over 70 steps (max diff {worst:.1e})")


# ------------------------------------------------------------------------------ engine
SYMS = ("AAA", "BBB", "CCC", "DDD", "EEE")


def make_panel(n_days=45, seed=0):
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    panel = {}
    for k, s in enumerate(SYMS):
        close = 100 * (1 + k) * np.cumprod(1 + rng.normal(0.001, 0.01, n_days))
        open_ = close * (1 + rng.normal(0, 0.004, n_days))
        panel[s] = pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.005,
                                 "Low": np.minimum(open_, close) * 0.995, "Close": close,
                                 "Volume": 1e6}, index=dates)
    return panel


def upto(panel, i):
    d = next(iter(panel.values())).index[i]
    return {s: df[df.index <= d] for s, df in panel.items()}


def stub_scorer(panel, latest):
    rows = []
    for k, (s, df) in enumerate(panel.items()):
        rows.append({"Date": latest, "symbol": s, "alpha_score": 1.0 - 0.1 * k, "Close": float(df["Close"].loc[latest]),
                     "Open": float(df["Open"].loc[latest]), "adv20_value": 1e9})
    return pd.DataFrame(rows).sort_values("alpha_score", ascending=False).reset_index(drop=True), {}


class StubPolicy:
    """Wants AAA and BBB heavily (cap will bite) and lots of cash."""
    class _S:
        shape = (config.TOP_K * 2 + 3,)
    observation_space = _S()

    def predict(self, obs, deterministic=True):
        a = np.zeros(config.TOP_K + 1, dtype=np.float32)
        a[0], a[1], a[-1] = 0.3, 0.3, 0.4
        return a, None


def _run(tmp, panel, **kw):
    kw.setdefault("scorer", stub_scorer); kw.setdefault("policy", StubPolicy())
    kw.setdefault("check_health", False); kw.setdefault("use_benchmark", False); kw.setdefault("allow_stale", True)
    return ft.run(state_dir=tmp, panel=panel, **kw)


def _state(tmp):
    return json.load(open(os.path.join(tmp, "state.json")))


def test_execution_at_open_with_costs_matches_hand_calc():
    tmp = tempfile.mkdtemp(); panel = make_panel()
    _run(tmp, upto(panel, 0))
    st = _state(tmp)
    tw = st["pending"]["target_weights"]
    assert set(tw) == {"AAA", "BBB"} and all(abs(w - 0.15) < 1e-6 for w in tw.values()), tw   # 15% cap enforced
    assert abs(st["pending"]["target_cash_weight"] - 0.70) < 1e-6
    _run(tmp, upto(panel, 1))
    st = _state(tmp)
    d1 = panel["AAA"].index[1]
    cap = config.INITIAL_CAPITAL
    w = {s: tw[s] for s in ("AAA", "BBB")}
    exp_pos = {s: w[s] * cap * panel[s].loc[d1, "Close"] / panel[s].loc[d1, "Open"] for s in w}
    bought = sum(w.values()) * cap
    exp_cash = (cap - bought * (1 + config.BUY_LEG_COST)) * (1 + ft.CASH_YIELD_DAILY)
    for s in exp_pos:
        assert abs(st["positions"][s] / exp_pos[s] - 1) < 1e-9, (s, st["positions"][s], exp_pos[s])
    assert abs(st["cash"] / exp_cash - 1) < 1e-9
    reb = pd.read_csv(os.path.join(tmp, "rebalance_log.csv"))
    assert abs(reb["cost_inr"].iloc[0] - bought * config.BUY_LEG_COST) < 1e-3
    assert abs(reb["stock_turnover"].iloc[0] - bought / 2 / cap) < 1e-9
    print("  ok  T+1-open execution, 15% cap, per-leg costs, cash yield match hand calculation")
    shutil.rmtree(tmp)


def test_idempotent_rerun():
    tmp = tempfile.mkdtemp(); panel = make_panel()
    _run(tmp, upto(panel, 0)); _run(tmp, upto(panel, 3))
    s1 = json.dumps(_state(tmp), sort_keys=True); n1 = len(pd.read_csv(os.path.join(tmp, "daily_log.csv")))
    _run(tmp, upto(panel, 3))
    assert json.dumps(_state(tmp), sort_keys=True) == s1, "re-running the same day changed state"
    assert len(pd.read_csv(os.path.join(tmp, "daily_log.csv"))) == n1, "duplicate log rows"
    print("  ok  re-running the same day is a no-op (no double trades / duplicate rows)")
    shutil.rmtree(tmp)


def test_catch_up_equals_day_by_day():
    panel = make_panel()
    a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
    _run(a, upto(panel, 0)); _run(b, upto(panel, 0))
    for i in range(1, 8):
        _run(a, upto(panel, i))            # one day at a time
    _run(b, upto(panel, 7))                # missed 7 days, then one catch-up run
    sa, sb = _state(a), _state(b)
    assert abs(sa["cash"] - sb["cash"]) < 1e-6
    assert all(abs(sa["positions"][k] - sb["positions"][k]) < 1e-6 for k in sa["positions"])
    print("  ok  a multi-day outage catches up to exactly the same portfolio")
    shutil.rmtree(a); shutil.rmtree(b)


def test_decisions_every_rebalance_freq_days():
    tmp = tempfile.mkdtemp(); panel = make_panel(45)
    for i in range(0, 36):
        _run(tmp, upto(panel, i))
    reb = pd.read_csv(os.path.join(tmp, "rebalance_log.csv"))
    dec = [panel["AAA"].index.get_loc(pd.Timestamp(x)) for x in reb["decision_date"]]
    assert dec == [0, 10, 20, 30][:len(dec)] and len(dec) >= 3, dec
    print(f"  ok  rebalances decided on days {dec} (every {config.REBALANCE_FREQ} trading days)")
    shutil.rmtree(tmp)


def test_halted_symbol_is_carried_flat_and_not_traded():
    tmp = tempfile.mkdtemp(); panel = make_panel()
    _run(tmp, upto(panel, 0)); _run(tmp, upto(panel, 1))
    v_before = _state(tmp)["positions"]["AAA"]
    p = upto(panel, 2)
    p["AAA"] = p["AAA"].iloc[:-1]          # AAA has no bar on day 2 (halted)
    old_cov, config.MIN_SYMBOL_COVERAGE = config.MIN_SYMBOL_COVERAGE, 0.5   # tiny test universe
    try:
        _run(tmp, p)
    finally:
        config.MIN_SYMBOL_COVERAGE = old_cov
    assert abs(_state(tmp)["positions"]["AAA"] - v_before) < 1e-9
    print("  ok  halted holding is carried flat without crashing")
    shutil.rmtree(tmp)


def test_guards_refuse_bad_data():
    panel = make_panel()
    tmp = tempfile.mkdtemp()
    try:
        ft.run(state_dir=tmp, panel=panel, scorer=stub_scorer, policy=StubPolicy(), check_health=False,
               use_benchmark=False, today=pd.Timestamp("2027-01-01"))
        raise AssertionError("stale data was accepted")
    except ls.DataIntegrityError as e:
        assert "old" in str(e)
    p = make_panel()
    for s in ("CCC", "DDD", "EEE"):
        p[s] = p[s].iloc[:-1]
    try:
        _run(tmp, p)
        raise AssertionError("partial download was accepted")
    except ls.DataIntegrityError as e:
        assert "Only" in str(e)
    print("  ok  stale data and partial downloads are refused")
    shutil.rmtree(tmp)


def test_signal_health_failure_holds_and_flags():
    tmp = tempfile.mkdtemp(); panel = make_panel()
    orig = ls.reference_score_bounds
    ls.reference_score_bounds = lambda *a, **k: {"top20_mean": (100, 101, 100, 101), "breadth": (-1, 1, -1, 1)}
    try:
        code = _run(tmp, upto(panel, 0), check_health=True)
    finally:
        ls.reference_score_bounds = orig
    assert code == 3 and _state(tmp)["pending"] is None
    print("  ok  out-of-distribution signal -> no trade staged, exit code 3")
    shutil.rmtree(tmp)


def test_feature_pipeline_is_point_in_time():
    """Features at t computed from data <= t must equal those computed with the future present."""
    if not os.path.exists(os.path.join(config.DATA_DIR, "RELIANCE.parquet")):
        print("  skip point-in-time feature test (no data/)"); return
    from feature_pipeline import build_feature_frame
    df = pd.read_parquet(os.path.join(config.DATA_DIR, "RELIANCE.parquet")).sort_index().ffill()
    full = build_feature_frame(df, include_targets=False)
    for k in (-300, -60, -15):
        t = df.index[k]
        cut = build_feature_frame(df[df.index <= t], include_targets=False).iloc[-1]
        assert np.nanmax(np.abs(cut.values.astype(float) - full.loc[t].values.astype(float))) == 0.0
    print("  ok  features are strictly point-in-time (no look-ahead)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        if t.__name__.startswith(("test_action", "test_observation", "test_real")) and not HAVE_ASSETS:
            print(f"  skip {t.__name__} (needs oos_predictions.parquet and best_ppo_agent.zip)"); continue
        try:
            t()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"  FAIL {t.__name__}: {e}")
    print("\nALL TESTS PASSED" if not failed else f"\n{failed} TEST(S) FAILED")
    sys.exit(1 if failed else 0)
