"""The risk agent: one judgement, four actions. Research and live code both use this module,
so a prompt built for training and one built on a live day cannot differ.

The judgement is whether a stock is safe to own for the next 5 sessions, meaning it never
trades 4% below the entry price. With no position that reads BUY or AVOID; long, HOLD or SELL.
The prompt names no company and no date, so the model cannot recall what happened next.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import wide

HORIZON = 5
STOP = 0.04
SYSTEM = "You are the risk desk of a paper trading book. Judge only from the numbers given. Answer with JSON only."
NUMBERS = ["atr_pct", "vol_60d", "vol_vs_spy_60d", "worst_day_21d", "best_day_21d", "est_days_to_earnings",
           "days_since_earnings", "earnings_reaction_atr", "vix", "spy_ret_20d", "spy_above_sma200"]
ACTIONS = {("none", True): "BUY", ("none", False): "AVOID", ("long", True): "HOLD", ("long", False): "SELL"}
PREFIX = '{"action": "'
BAD_SYMBOLS = {"PARA"}  # the price file holds the wrong instrument: ~$100,000 a share, near-zero volume


def adjusted(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    k = wide(bars, "adj_close") / wide(bars, "close")
    return wide(bars, "open") * k, wide(bars, "low") * k, wide(bars, "close") * k


def price_numbers(bars: pd.DataFrame) -> pd.DataFrame:
    """The numbers not already in build_features. Uses bars up to each date only."""
    close = adjusted(bars)[2]
    r = close.pct_change(fill_method=None)
    x = r.sub(r["SPY"], axis=0)
    parts = {"vol_60d": r.rolling(60).std(), "vol_vs_spy_60d": x.rolling(60).std(),
             "worst_day_21d": r.rolling(21).min(), "best_day_21d": r.rolling(21).max()}
    return pd.concat({k: v.stack(future_stack=True) for k, v in parts.items()}, axis=1).rename_axis(["date", "symbol"])


def outcomes(bars: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Enter at the next open. ret: close of t+horizon against entry. worst: lowest low until then."""
    o, low, c = adjusted(bars)
    entry = o.shift(-1)
    parts = {"ret": c.shift(-horizon) / entry - 1, "worst": low.shift(-horizon).rolling(horizon).min() / entry - 1}
    d = pd.concat({k: v.stack(future_stack=True) for k, v in parts.items()}, axis=1).rename_axis(["date", "symbol"])
    d = d.dropna()
    d["safe"] = d.worst > -STOP
    return d


def _pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:+.{nd}f}%"


def messages(r, position: str) -> list[dict]:
    """r: anything with the NUMBERS as attributes; position: 'none' or 'long'."""
    days = int(round(r.est_days_to_earnings))
    user = (f"Position: {position}\n"
            f"Volatility: ATR {r.atr_pct * 100:.1f}% of price, daily 60-day {r.vol_60d * 100:.1f}%, against SPY {r.vol_vs_spy_60d * 100:.1f}%\n"
            f"Last 21 sessions: worst day {_pct(r.worst_day_21d)}, best day {_pct(r.best_day_21d)}\n"
            f"Earnings: next in about {days} sessions, last {int(r.days_since_earnings)} sessions ago, reaction {r.earnings_reaction_atr:+.1f} ATR\n"
            f"Market: VIX {r.vix:.1f}, SPY 20-day {_pct(r.spy_ret_20d)}, SPY above 200-day average: {'yes' if r.spy_above_sma200 else 'no'}\n\n"
            f"Will this stock stay above 4% below entry for the next 5 sessions?\n"
            f"With no position answer BUY if yes, AVOID if no. If long answer HOLD if yes, SELL if no.\n"
            'Return {"action": "..."}')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def example(r, position: str) -> dict:
    return {"messages": messages(r, position) + [{"role": "assistant", "content": '{"action": "%s"}' % ACTIONS[(position, bool(r.safe))]}]}


def report(d: pd.DataFrame, score: str = "p_safe") -> dict:
    """d: date, score column, ret, worst. Stocks are ranked by the score on each date."""
    d = d.dropna(subset=[score, "worst"])
    q = d.groupby("date")[score].rank(pct=True)
    hit = d.worst <= -STOP
    safe, risky = q > 2 / 3, q <= 1 / 3
    day = d.assign(hit=hit).groupby("date").apply(
        lambda g: roc_auc_score(~g.hit, g[score]) if 0 < g.hit.mean() < 1 and g[score].nunique() > 1 else np.nan, include_groups=False).dropna()
    gap = hit[risky].groupby(d.date[risky]).mean() - hit[safe].groupby(d.date[safe]).mean()
    return {"cases": len(d), "dates": d.date.nunique(), "auc_same_day": round(day.mean(), 4),
            "stop_safe": round(hit[safe].mean(), 4), "stop_all": round(hit.mean(), 4), "stop_risky": round(hit[risky].mean(), 4),
            "gap_t": round(gap.mean() / gap.std() * np.sqrt(len(gap)), 1),
            "ret_safe": round(d.ret[safe].mean(), 5), "ret_risky": round(d.ret[risky].mean(), 5)}


def simulate(d: pd.DataFrame, score: str = "p_safe", cost: float = 0.001, enter: float = 2 / 3, leave: float = 1 / 3) -> pd.DataFrame:
    """Paper book following the actions, one decision date to the next, equal weight.

    Flat names in the top slice are bought; held names falling to the bottom slice are sold,
    so the middle slice holds what it owns and buys nothing. Decision dates must be HORIZON
    sessions apart, since each holding earns `ret`. cost is per side. Returns one row per date.
    """
    held, rows = set(), []
    for day, g in d.dropna(subset=[score]).sort_values("date").groupby("date"):
        q = g.set_index("symbol")[score].rank(pct=True)
        live = held & set(q.index)
        buys = set(q.index[q > enter]) - live
        sells = {s for s in live if q[s] <= leave}
        new = (live - sells) | buys
        r = g.set_index("symbol").ret
        n = max(len(new), 1)
        gross = r.reindex(sorted(new)).mean() if new else 0.0
        trades = (len(buys) + len(sells) + len(held - live)) / n
        rows.append({"date": day, "ret": gross - trades * cost, "held": len(new), "share": len(new) / len(q), "turnover": trades})
        held = new
    return pd.DataFrame(rows)


def book_stats(x: pd.Series) -> dict:
    c = (1 + x).cumprod()
    per_year = 252 / HORIZON
    return {"sharpe": round(x.mean() / x.std() * np.sqrt(per_year), 2), "worst_fall": round(float((c / c.cummax() - 1).min()), 3),
            "yearly": round(float(c.iloc[-1] ** (per_year / len(x)) - 1), 3)}


def slot_book(d: pd.DataFrame, score: str = "p_safe", enter: float = .7, leave: float = .6, cost: float = 0.001) -> pd.DataFrame:
    """Each stock has its own fixed slot, so every call stands alone and needs no ranking.

    A slot is bought when the score reaches `enter` and sold when it falls below `leave`; the
    gap between them stops the book churning. A sold slot sits in cash at 0%, so the whole
    book steps aside when most stocks look unsafe. Returns one row per decision date.
    """
    held, rows = set(), []
    for day, g in d.dropna(subset=[score]).sort_values("date").groupby("date"):
        p = g.set_index("symbol")[score]
        live = held & set(p.index)
        buys = set(p.index[p >= enter]) - live
        sells = {s for s in live if p[s] < leave}
        new = (live - sells) | buys
        r = g.set_index("symbol").ret
        n = len(p)
        rows.append({"date": day, "ret": (r.reindex(sorted(new)).sum() - (len(buys) + len(sells)) * cost) / n,
                     "share": len(new) / n, "turnover": (len(buys) + len(sells)) / n})
        held = new
    return pd.DataFrame(rows)
