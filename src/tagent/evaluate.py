"""Decision-quality and portfolio metrics for any policy's decisions.

Input: DataFrame with columns [date, symbol, action] plus joined labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COST_BPS = 10  # round-trip cost incl. spread/slippage, per trade


def decision_metrics(d: pd.DataFrame, horizon: int = 20) -> dict:
    fr, fx = f"fwd_ret_{horizon}d", f"fwd_excess_{horizon}d"
    buys = d[d.action == "BUY"]
    rest = d[d.action != "BUY"]
    out = {
        "n": len(d),
        "buy_rate": len(buys) / max(len(d), 1),
        "n_buy": len(buys),
        "buy_tb_win_rate": (buys.tb_label == 1).mean() if len(buys) else np.nan,
        "buy_tb_loss_rate": (buys.tb_label == -1).mean() if len(buys) else np.nan,
        "base_tb_win_rate": (d.tb_label == 1).mean(),
        "buy_mean_fwd_ret": buys[fr].mean(),
        "buy_mean_excess": buys[fx].mean(),
        "nonbuy_mean_fwd_ret": rest[fr].mean(),
        "all_mean_fwd_ret": d[fr].mean(),
        # edge of BUY selection over the universe average on the same dates
        "buy_selection_edge": (buys[fr] - d.groupby("date")[fr].transform("mean")
                               .loc[buys.index]).mean() if len(buys) else np.nan,
        "avoid_mean_fwd_ret": d.loc[d.action == "AVOID", fr].mean(),
        "wait_mean_fwd_ret": d.loc[d.action == "WAIT", fr].mean(),
    }
    out.update(rank_ic(d, horizon))
    out["buy_profit_factor"] = _profit_factor(buys[fr] - COST_BPS / 1e4)
    out["buy_t_stat"] = _t(buys[fr] - d.groupby("date")[fr].transform("mean").loc[buys.index])
    return out


ACTION_SCORE = {"BUY": 1.0, "HOLD": 0.0, "WAIT": 0.0, "SELL": -1.0, "AVOID": -1.0}


def daily_rank_ic(d: pd.DataFrame, horizon: int = 20, score_col: str | None = None) -> pd.Series:
    """Per-date Spearman IC between the score and the forward excess return.
    Dates with a constant score or fewer than 5 names are skipped."""
    fx = f"fwd_excess_{horizon}d"
    s = d[score_col] if score_col else d.action.map(ACTION_SCORE)
    tmp = pd.DataFrame({"date": d.date, "s": s, "y": d[fx]}).dropna()
    return tmp.groupby("date").apply(
        lambda g: g.s.rank().corr(g.y.rank()) if g.s.nunique() > 1 and len(g) > 4 else np.nan,
        include_groups=False).dropna()


def nw_tstat(x: pd.Series, lags: int) -> float:
    """t-stat of the mean with a Newey-West (Bartlett) variance."""
    v = x.values - x.mean()
    var = (v @ v) / len(v)
    for L in range(1, lags + 1):
        var += 2 * (1 - L / (lags + 1)) * (v[L:] @ v[:-L]) / len(v)
    return float(x.mean() / np.sqrt(var / len(v)))


def rank_ic(d: pd.DataFrame, horizon: int = 20, score_col: str | None = None,
            nw_lags: int | None = None) -> dict:
    """Primary metric: mean per-date Spearman IC between the policy's score and the
    forward excess return, with a Newey-West t-stat (overlapping horizons make
    consecutive dates autocorrelated)."""
    ics = daily_rank_ic(d, horizon, score_col)
    if len(ics) < 3:
        return {"rank_ic": np.nan, "rank_ic_t_nw": np.nan, "ic_dates": len(ics)}
    lags = nw_lags if nw_lags is not None else max(1, horizon // 5)  # decisions every 5 days
    return {"rank_ic": float(ics.mean()), "rank_ic_t_nw": nw_tstat(ics, lags), "ic_dates": len(ics)}


def _profit_factor(r: pd.Series) -> float:
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    return float(g / l) if l > 0 else np.nan


def _t(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else np.nan


def portfolio_curve(d: pd.DataFrame, horizon: int = 20) -> pd.Series:
    """Non-overlapping rebalance every `horizon` decision dates: equal-weight all
    BUYs (cash if none), net of costs. Returns per-period returns indexed by date."""
    fr = f"fwd_ret_{horizon}d"
    dates = sorted(d.date.unique())
    rets = {}
    for t in dates:
        sl = d[(d.date == t) & (d.action == "BUY")]
        rets[t] = (sl[fr].mean() - COST_BPS / 1e4) if len(sl) else 0.0
    return pd.Series(rets).sort_index()


def portfolio_metrics(p: pd.Series, periods_per_year: float) -> dict:
    eq = (1 + p).cumprod()
    dd = eq / eq.cummax() - 1
    sd = p.std(ddof=1)
    down = p[p < 0].std(ddof=1)
    return {
        "total_return": eq.iloc[-1] - 1 if len(eq) else np.nan,
        "sharpe": p.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else np.nan,
        "sortino": p.mean() / down * np.sqrt(periods_per_year) if down and down > 0 else np.nan,
        "max_drawdown": dd.min() if len(dd) else np.nan,
        "periods": len(p),
        "pct_invested": (p != 0).mean(),
    }
