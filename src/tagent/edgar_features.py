"""Point-in-time features from cached EDGAR data (see edgar.py).

Earnings events: 8-K filings with Item 2.02 (results of operations). A filing
accepted at/after 16:00 New York time reacts in the next session. At decision date t we only
use an event whose 2-day reaction window [e-1, e+1] closed on or before t.
The next earnings date is NOT taken from future filings (that would leak); it is
estimated as 63 sessions after the last one.

Fundamentals: XBRL company facts. Only facts filed strictly before t are used,
and for each reporting period the latest filing before t wins (so restatements
appear only once they were published). Outputs are ratios, never raw dollars, so
they do not identify the company.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .edgar import EDGAR_DIR, ticker_cik_map

EVENT_COLS = ["days_since_earnings", "earnings_reaction_atr", "drift_since_earnings",
              "est_days_to_earnings", "earnings_in_horizon"]
FUND_COLS = ["rev_yoy", "eps_yoy", "fcf_margin", "net_margin", "shares_yoy", "fund_age_days"]

REVENUE = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
           "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenuesNetOfInterestExpense"]
MAX_AGE_DAYS = 400  # revenue-based fields older than this are treated as missing


# ---------------------------------------------------------------- earnings events
def _filings(cik: int) -> pd.DataFrame:
    d = EDGAR_DIR / "submissions"
    main = d / f"CIK{cik:010d}.json"
    if not main.exists():
        return pd.DataFrame()
    j = json.loads(main.read_text())
    parts = [j["filings"]["recent"]]
    for f in j["filings"].get("files", []):
        p = d / f["name"]
        if p.exists():
            parts.append(json.loads(p.read_text()))
    cols = ["form", "filingDate", "acceptanceDateTime", "items", "accessionNumber", "primaryDocument"]
    return pd.concat([pd.DataFrame({c: p.get(c, []) for c in cols}) for p in parts], ignore_index=True)


def earnings_sessions(cik: int, sessions: pd.DatetimeIndex) -> np.ndarray:
    """Index (into `sessions`) of the first session that could react to each earnings 8-K."""
    f = _filings(cik)
    if f.empty:
        return np.array([], dtype=int)
    f = f[(f.form == "8-K") & f["items"].fillna("").str.contains("2.02", regex=False)]
    # acceptanceDateTime is UTC; the 16:00 cutoff applies in New York time
    acc = pd.to_datetime(f.acceptanceDateTime, utc=True, errors="coerce") \
        .dt.tz_convert("America/New_York").dt.tz_localize(None)
    day = acc.dt.normalize() + pd.to_timedelta((acc.dt.hour >= 16).astype(int), unit="D")
    idx = np.unique(sessions.searchsorted(day.dropna().values))
    return idx[idx < len(sessions)]


def event_features(px: pd.Series, spy: pd.Series, atr_pct: pd.Series, ev: np.ndarray,
                   at_idx: np.ndarray) -> pd.DataFrame:
    """px, spy, atr_pct aligned on the session index; at_idx = decision session indices."""
    c, s, a = px.values, spy.values, atr_pct.values
    out = np.full((len(at_idx), len(EVENT_COLS)), np.nan)
    if len(ev):
        done = ev + 1                                  # reaction window closes at e+1
        k = np.searchsorted(done, at_idx, side="right") - 1
        for row, (t, j) in enumerate(zip(at_idx, k)):
            if j < 0:
                continue
            e = ev[j]
            if e < 1 or e + 1 >= len(c):
                continue
            react = (c[e + 1] / c[e - 1] - 1) - (s[e + 1] / s[e - 1] - 1)
            since = t - e
            out[row] = [since, react / a[e - 1] if a[e - 1] > 0 else np.nan,
                        (c[t] / c[e + 1] - 1) - (s[t] / s[e + 1] - 1),
                        max(0, 63 - since), float(63 - since <= 20)]
    return pd.DataFrame(out, columns=EVENT_COLS)


# ---------------------------------------------------------------- fundamentals
def _facts(cik: int) -> pd.DataFrame:
    p = EDGAR_DIR / "companyfacts" / f"CIK{cik:010d}.json"
    if not p.exists():
        return pd.DataFrame()
    j = json.loads(p.read_text())["facts"]
    want = {"us-gaap": REVENUE + ["EarningsPerShareDiluted", "NetIncomeLoss",
                                  "NetCashProvidedByUsedInOperatingActivities",
                                  "PaymentsToAcquirePropertyPlantAndEquipment"],
            "dei": ["EntityCommonStockSharesOutstanding"]}
    rows = []
    for ns, names in want.items():
        for name in names:
            for unit, vals in j.get(ns, {}).get(name, {}).get("units", {}).items():
                for v in vals:
                    rows.append((name, v.get("start"), v["end"], v["val"], v["filed"]))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["concept", "start", "end", "val", "filed"])
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["dur"] = (df.end - df.start).dt.days
    rev = df.concept.isin(REVENUE)
    df.loc[rev, "prio"] = df.loc[rev, "concept"].map({c: i for i, c in enumerate(REVENUE)})
    df.loc[rev, "concept"] = "Revenue"
    return df[df.filed >= "2012-01-01"]


def _latest_per_period(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["filed", "prio"], ascending=[True, False], na_position="first")
    return df.drop_duplicates(["concept", "start", "end"], keep="last")


def _yoy(df: pd.DataFrame, concept: str, lo: int, hi: int, pct: bool = True) -> tuple[float, pd.Timestamp | None]:
    x = df[(df.concept == concept) & df.dur.between(lo, hi)]
    if x.empty:
        return np.nan, None
    last = x.loc[x.end.idxmax()]
    prev = x[(x.end <= last.end - pd.Timedelta(days=350)) & (x.end >= last.end - pd.Timedelta(days=380))
             & (abs(x.dur - last.dur) <= 10)]
    if prev.empty:
        return np.nan, last.end
    p = prev.iloc[-1].val
    if not p:
        return np.nan, last.end
    g = last.val / p - 1 if pct and p > 0 else (last.val - p) / abs(p)
    return float(np.clip(g, -3, 3)), last.end


def fundamentals_at(df: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    a = _latest_per_period(df[df.filed < cutoff])
    if a.empty:
        return {}
    out = {}
    out["rev_yoy"], rev_end = _yoy(a, "Revenue", 80, 100)
    out["eps_yoy"], _ = _yoy(a, "EarningsPerShareDiluted", 80, 100, pct=False)
    ann = a[a.dur.between(350, 380)]
    ends = ann[ann.concept == "Revenue"].end
    if len(ends):
        e = ends.max()
        at = ann[ann.end == e].sort_values("filed").groupby("concept").val.last()
        rev = at.get("Revenue")
        if rev and rev > 0:
            ocf, capex = at.get("NetCashProvidedByUsedInOperatingActivities"), at.get(
                "PaymentsToAcquirePropertyPlantAndEquipment", 0.0)
            out["fcf_margin"] = float(np.clip((ocf - capex) / rev, -2, 2)) if ocf is not None else np.nan
            ni = at.get("NetIncomeLoss")
            out["net_margin"] = float(np.clip(ni / rev, -2, 2)) if ni is not None else np.nan
    sh = a[a.concept == "EntityCommonStockSharesOutstanding"].sort_values("end")
    if len(sh) >= 2:
        last = sh.iloc[-1]
        prev = sh[(sh.end <= last.end - pd.Timedelta(days=300)) & (sh.end >= last.end - pd.Timedelta(days=430))]
        if len(prev) and prev.iloc[-1].val:
            out["shares_yoy"] = float(np.clip(last.val / prev.iloc[-1].val - 1, -1, 1))
    out["_period_end"] = rev_end if rev_end is not None else a.end.max()
    if rev_end is None or (cutoff - rev_end).days > MAX_AGE_DAYS:
        out["rev_yoy"] = np.nan
    if "fcf_margin" in out and (cutoff - e).days > MAX_AGE_DAYS + 365:
        out["fcf_margin"] = out["net_margin"] = np.nan
    return out


def fundamental_features(cik: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per decision date; recomputed only when a new filing appears."""
    df = _facts(cik)
    res = pd.DataFrame(index=dates, columns=FUND_COLS, dtype=float)
    if df.empty:
        return res
    filed = np.sort(df.filed.dropna().unique())
    cache, last_key = {}, None
    for d in dates:
        k = np.searchsorted(filed, np.datetime64(d), side="left")  # filings strictly before d
        if k != last_key:
            cache = fundamentals_at(df, d) if k > 0 else {}
            last_key = k
        if cache:
            row = {c: cache.get(c, np.nan) for c in FUND_COLS[:-1]}
            pe = cache.get("_period_end")
            row["fund_age_days"] = (d - pe).days if pe is not None else np.nan
            res.loc[d] = row
    return res


# ---------------------------------------------------------------- all symbols
def edgar_features(bars: pd.DataFrame, atr_wide: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """(date, symbol)-indexed frame of event + fundamental features at the given dates."""
    from .data import wide
    px = wide(bars, "adj_close")
    sessions = px.index
    spy = px["SPY"]
    atr = atr_wide.reindex(sessions)
    at_idx = sessions.get_indexer(dates)
    cmap = ticker_cik_map()
    parts = []
    syms = sorted(atr_wide.columns)
    for i, s in enumerate(syms):
        cik = cmap.get(s)
        if cik is None or s not in px:
            continue
        ev = earnings_sessions(cik, sessions)
        e = event_features(px[s], spy, atr[s] if s in atr else pd.Series(np.nan, index=sessions), ev, at_idx)
        f = fundamental_features(cik, dates).reset_index(drop=True)
        part = pd.concat([e, f], axis=1)
        part["date"], part["symbol"] = dates, s
        parts.append(part)
        if i % 100 == 0:
            print(f"  edgar features {i+1}/{len(syms)} {s}", flush=True)
    return pd.concat(parts, ignore_index=True).set_index(["date", "symbol"])
