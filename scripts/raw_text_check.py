"""Cheap filter before any 9B run on raw text: can a plain word model link earnings releases to what
happens next? Train period only; masked text; models fit on earlier months score later ones.

  PYTHONPATH=src nice -n 10 .venv/bin/python -W ignore scripts/raw_text_check.py

Targets: the reaction to the release (a check that the method can find a relationship when one
exists) and the excess return over the next 20 and 60 sessions. Luck: the same test with
outcomes shuffled inside each month.
"""
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from tagent.data import ROOT
from tagent.edgar import EDGAR_DIR
from tagent.evaluate import nw_tstat

D = ROOT / "reports" / "diagnostics"


def month_ic(d, y):
    a, b = d.groupby("month").score.rank(), d.groupby("month")[y].rank()
    a, b = a - a.groupby(d.month).transform("mean"), b - b.groupby(d.month).transform("mean")
    n = d.groupby("month").score.transform("size")
    ok = n >= 15
    a, b, m = a[ok], b[ok], d.month[ok]
    return ((a * b).groupby(m).sum() / np.sqrt((a * a).groupby(m).sum() * (b * b).groupby(m).sum()).replace(0, np.nan)).dropna()


def walk(t, X, y, gap, rng=None):
    return month_ic(walk_scores(t, X, y, gap, rng), "target")


def walk_scores(t, X, y, gap, rng=None):
    yy = t[y].copy()
    if rng is not None:
        yy = yy.groupby(t.month).transform(lambda s: rng.permutation(s.values))
    months = np.array(sorted(t.month.unique()))
    score = pd.Series(np.nan, index=t.index)
    pos = np.arange(len(t))
    for f in np.array_split(months[len(months) // 3:], 6):
        start = pd.Period(f[0], "M").start_time - pd.Timedelta(days=gap)
        a = pos[(t.date < start).values & yy.notna().values]
        b = pos[t.month.isin(f).values]
        lab = yy.iloc[a].rank(pct=True).values - .5
        score.iloc[b] = Ridge(alpha=30.0).fit(X[a], lab).predict(X[b])
    return t.assign(score=score, target=yy).dropna(subset=["score", "target"])


def main():
    s = pd.read_parquet(ROOT / "data" / "features" / "earnings_text_scores_v2_Qwen3.5-9B-4bit.parquet")
    s = s[s.valid & s.is_release].copy()
    s["date"] = pd.to_datetime(s.acceptance, utc=True).dt.tz_localize(None).astype("datetime64[ns]").dt.normalize()
    ev = pd.read_parquet(D / "signal_map_events.parquet")
    ev["date"] = ev.date.astype("datetime64[ns]")
    t = pd.merge_asof(s.sort_values("date"), ev.sort_values("date"), on="date", by="symbol", direction="forward", tolerance=pd.Timedelta(days=5),
                      suffixes=("", "_e")).dropna(subset=["react_atr"])
    files = {p.stem: p for p in (EDGAR_DIR / "earnings_text_v2").glob("*.json")}
    sample = json.loads(next(iter(files.values())).read_text())
    print("text file keys", list(sample)[:10], flush=True)
    key = "masked" if "masked" in sample else "text"

    def body(r):
        p = files.get(f"{int(r.cik)}_{r.accession.replace('-', '')}")
        return json.loads(p.read_text()).get(key, "") if p else ""

    t["text"] = [body(r) for r in t.itertuples()]
    t = t[t.text.str.len() > 500].sort_values("date").reset_index(drop=True)
    print(len(t), "events with text,", t.symbol.nunique(), "names, mean chars", int(t.text.str.len().mean()), flush=True)
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=15, max_df=.6, max_features=40000, sublinear_tf=True, stop_words="english").fit_transform(t.text).tocsr()
    rows = []
    for y, gap in (("react_atr", 10), ("y5", 15), ("y20", 45), ("y60", 100)):
        ic = walk(t, X, y, gap)
        rng = np.random.default_rng(0)
        null = [walk(t, X, y, gap, rng).mean() for _ in range(40)]
        rows.append({"target": y, "ic": round(ic.mean(), 4), "t": round(nw_tstat(ic, 1), 2), "months": len(ic), "positive_months": round((ic > 0).mean(), 2),
                     "shuffled_mean": round(float(np.mean(null)), 4), "shuffled_sd": round(float(np.std(null)), 4),
                     "share_shuffled_as_good": round(float(np.mean(np.array(null) >= ic.mean())), 3)})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(D / "raw_text_check.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(res.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()


def identity_check():
    """Is the text being read, or does it only say which company or sector this is?"""
    import scipy.sparse as sp
    from tagent.data import gics_sectors
    s = pd.read_parquet(ROOT / "data" / "features" / "earnings_text_scores_v2_Qwen3.5-9B-4bit.parquet")
    s = s[s.valid & s.is_release].copy()
    s["date"] = pd.to_datetime(s.acceptance, utc=True).dt.tz_localize(None).astype("datetime64[ns]").dt.normalize()
    ev = pd.read_parquet(D / "signal_map_events.parquet")
    ev["date"] = ev.date.astype("datetime64[ns]")
    t = pd.merge_asof(s.sort_values("date"), ev.sort_values("date"), on="date", by="symbol", direction="forward", tolerance=pd.Timedelta(days=5),
                      suffixes=("", "_e")).dropna(subset=["react_atr"])
    files = {p.stem: p for p in (EDGAR_DIR / "earnings_text_v2").glob("*.json")}
    t["text"] = [json.loads(files[f"{int(r.cik)}_{r.accession.replace('-', '')}"].read_text()).get("text", "") if f"{int(r.cik)}_{r.accession.replace('-', '')}" in files else "" for r in t.itertuples()]
    t = t[t.text.str.len() > 500].sort_values("date").reset_index(drop=True)
    t["sector"] = t.symbol.map(gics_sectors()).fillna("Unknown")
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=15, max_df=.6, max_features=40000, sublinear_tf=True, stop_words="english").fit_transform(t.text).tocsr()
    name = sp.csr_matrix(pd.get_dummies(t.symbol).values.astype(float))
    sect = sp.csr_matrix(pd.get_dummies(t.sector).values.astype(float))
    rows = []
    for label, M in (("text", X), ("company identity only", name), ("sector only", sect)):
        for y, gap in (("y20", 45), ("y60", 100)):
            ic = walk(t, M, y, gap)
            rows.append({"inputs": label, "target": y, "ic": round(ic.mean(), 4), "t": round(nw_tstat(ic, 1), 2)})
    # text with the sector's and company's own past level taken out of the target
    t["y60_in_sector"] = t.y60 - t.groupby(["month", "sector"]).y60.transform("mean")
    t["y60_vs_own_past"] = t.y60 - t.groupby("symbol").y60.transform(lambda x: x.shift(2).expanding().mean())
    for y in ("y60_in_sector", "y60_vs_own_past"):
        ic = walk(t, X, y, 100)
        rows.append({"inputs": "text", "target": y, "ic": round(ic.mean(), 4), "t": round(nw_tstat(ic, 1), 2)})
    yr = walk(t, X, "y60", 100)
    res = pd.DataFrame(rows)
    res.to_csv(D / "raw_text_identity.csv", index=False)
    print(res.to_string(index=False), flush=True)
    print(yr.groupby(pd.PeriodIndex(yr.index, freq="M").year).mean().round(3).to_dict(), flush=True)
