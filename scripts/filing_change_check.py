"""Cheap filter before any 9B run on filing changes: does what changed in a 10-K or 10-Q, against the
same filing a year earlier, say anything about the next 60 sessions? Train period only: bars are cut
at 2025-01-01, so no outcome reaches validation.

  PYTHONPATH=src nice -n 10 .venv/bin/python -W ignore scripts/filing_change_check.py build
  PYTHONPATH=src nice -n 10 .venv/bin/python -W ignore scripts/filing_change_check.py test

Decision at the close of the first session after the filing is public, entry at the next open, exit
at the close 60 sessions after the decision. Outcome: return minus sector mean, over the stock's own
volatility. Six signals, fixed before looking. The first five expect less change to be better, as
published (Cohen, Malloy and Nguyen, Lazy Prices), and are not fitted:
  stable, stable_10k, stable_10q   wording similarity, ranked within month and form
  little_new                       small share of new prose
  steady_length                    small change in length
  words                            word model on the added and removed sentences, fit on earlier months

Bar, written before any result. One signal must do all of this:
  1. IC positive with t at least 2
  2. IC above company identity and above sector identity on the same events
  3. best t over the six signals reached by shuffled outcomes in at most 5% of runs
  4. IC positive in both 2017-2021 and 2022-2024
  5. BUY book after 10 bps a side beats owning everything, t at least 2
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from raw_text_check import month_ic, walk, walk_scores
from tagent.data import ROOT, load_bars
from tagent.evaluate import nw_tstat
from tagent.filing_changes import FILINGS, brief, prior, read
from tagent.risk_agent import BAD_SYMBOLS, adjusted

D = ROOT / "reports" / "diagnostics"
FEAT = ROOT / "data" / "features" / "filing_changes.parquet"
CUT = pd.Timestamp("2025-01-01")
H, COST, GAP, RUNS = 60, .001, 100, 100
FIXED = ["stable", "stable_10k", "stable_10q", "little_new", "steady_length"]


def index():
    i = pd.read_csv(FILINGS / "index.csv").drop_duplicates(["cik", "accession"])
    return pd.concat([prior(g) for _, g in i.groupby("cik")]).dropna(subset=["prior"])


def one(r):
    cik, accession, old = r
    b = brief(read(cik, old), read(cik, accession))
    return {"accession": accession, **b, "added": "\n".join(b["added"]), "removed": "\n".join(b["removed"])}


def build():
    i = index()
    rows = pd.read_parquet(FEAT).to_dict("records") if FEAT.exists() else []
    seen = {r["accession"] for r in rows}
    todo = [(r.cik, r.accession, r.prior) for r in i.itertuples() if r.accession not in seen]
    print(len(i), "filings with a year-earlier match,", len(todo), "to do", flush=True)
    with ProcessPoolExecutor(5) as ex:
        for n, r in enumerate(ex.map(one, todo, chunksize=10), 1):
            rows.append(r)
            if n % 1000 == 0:
                pd.DataFrame(rows).to_parquet(FEAT)
                print(n, "of", len(todo), flush=True)
    pd.DataFrame(rows).to_parquet(FEAT)
    print("done", len(rows), flush=True)


def market():
    """Price tables, the universe and each stock's 60-session outcome, for every session."""
    panel = pd.read_parquet(D / "signal_map_panel.parquet", columns=["date", "symbol", "sector"])
    bars = load_bars()
    bars = bars[(bars.date < CUT) & ~bars.symbol.isin(BAD_SYMBOLS) & bars.symbol.isin(panel.symbol.unique())]
    o, _, c = adjusted(bars)
    member = panel.assign(on=True).pivot(index="date", columns="symbol", values="on").reindex(index=c.index, columns=c.columns)
    member = member.ffill(limit=7).fillna(False).astype(bool)
    sector = panel.groupby("symbol").sector.last().reindex(c.columns)
    ret = (c.shift(-H) / o.shift(-1) - 1).clip(-.9, 3).where(member)
    return o, c, member, sector, ret, ret.T.groupby(sector).mean().T, c.pct_change(fill_method=None).rolling(60).std()


def place(t, c, sector, ret, peers, vol):
    """t has pos, the decision session. Adds the date, the clean outcome and the reaction into it."""
    t = t[t.symbol.isin(c.columns) & (t.pos < len(c.index))].copy()
    t["date"] = c.index[t.pos]
    t["col"] = c.columns.get_indexer(t.symbol)
    t["sector"] = t.symbol.map(sector)
    t["ret"] = ret.values[t.pos, t.col]
    t["excess"] = t.ret - peers.values[t.pos, peers.columns.get_indexer(t.sector)]
    t["y"] = t.excess / (vol.values[t.pos, t.col] * np.sqrt(H))
    before = c.values[np.maximum(t.pos - 2, 0), t.col]
    t["reaction"] = (c.values[t.pos, t.col] / before - 1) / (vol.values[t.pos, t.col] * np.sqrt(2))
    return t


def events():
    """One row per filing, with the decision session and the clean outcome. Also the price tables."""
    t = index().merge(pd.read_parquet(FEAT), on="accession")
    o, c, member, sector, ret, peers, vol = market()
    days = c.index
    ny = pd.to_datetime(t.acceptance, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    t["pos"] = days.searchsorted(ny.dt.normalize() + pd.to_timedelta((ny.dt.hour >= 16).astype(int), "D"))
    t = place(t, c, sector, ret, peers, vol)
    print(len(t), "filings on a session;", int(t.y.notna().sum()), "with an outcome;", int((t.old_sentences < 50).sum()), "with under 50 old sentences", flush=True)
    t = t[t.y.notna() & (t.old_sentences >= 50) & (t.new_sentences >= 50)].sort_values("date").reset_index(drop=True)
    t["month"] = t.date.dt.to_period("M").astype(str)
    t["raw_stable"] = t.similarity
    t["raw_little_new"] = -t.new_prose / t.new_sentences
    t["raw_steady_length"] = -np.log(t.new_sentences / t.old_sentences).abs()
    for k in ("stable", "little_new", "steady_length"):
        t[k] = t.groupby(["month", "form"])["raw_" + k].rank(pct=True)
    return t, o, c, member


def ic_of(d, score, y):
    return month_ic(d.assign(score=d[score], target=d[y]).dropna(subset=["score", "target"]), "target")


def fixed_ics(t, y):
    k, q = t[t.form == "10-K"], t[t.form == "10-Q"]
    return {"stable": ic_of(t, "stable", y), "stable_10k": ic_of(k, "stable", y), "stable_10q": ic_of(q, "stable", y),
            "little_new": ic_of(t, "little_new", y), "steady_length": ic_of(t, "steady_length", y)}


def row(name, ic):
    a, b = ic[ic.index < "2022"], ic[ic.index >= "2022"]
    return {"signal": name, "ic": round(ic.mean(), 4), "t": round(nw_tstat(ic, 3), 2), "months": len(ic), "positive_months": round((ic > 0).mean(), 2),
            "ic_to_2021": round(a.mean(), 4), "t_to_2021": round(nw_tstat(a, 3), 2), "ic_2022_on": round(b.mean(), 4), "t_2022_on": round(nw_tstat(b, 3), 2)}


def buys(d, col):
    """A BUY needs a top-30% score against the last year's filings of the same form. Only past filings
    set the line, so the call could have been made on the day."""
    line = pd.concat([g.set_index("date")[col].rolling("365D", closed="left", min_periods=100).quantile(.7).set_axis(g.index) for _, g in d.groupby("form")])
    low = pd.concat([g.set_index("date")[col].rolling("365D", closed="left", min_periods=100).quantile(.3).set_axis(g.index) for _, g in d.groupby("form")])
    return d[col] > line.reindex(d.index), d[col] < low.reindex(d.index), line.reindex(d.index).notna()


def book(d, col, o, c, member):
    """Hold each BUY from next open to the exit, equal weight, 10 bps a side, against owning everything."""
    buy, sell, ok = buys(d, col)
    held = np.zeros(c.shape, bool)
    for r in d[buy].itertuples():
        held[r.pos + 1:r.pos + H + 1, r.col] = True
    was = np.vstack([np.zeros((1, c.shape[1]), bool), held[:-1]])
    enter, leave = held & ~was, ~held & was
    day = c.pct_change(fill_method=None).values
    day = np.where(enter, (c / o - 1).values, day)
    n = held.sum(1)
    live = n > 0
    net = np.nansum(np.where(held, day, 0), 1)[live] / n[live] - (enter.sum(1) + leave.sum(1))[live] / n[live] * COST
    every = c.pct_change(fill_method=None).where(member).mean(axis=1).values[live]
    x = pd.DataFrame({"book": net, "all": every}, index=c.index[live]).dropna()
    x = x[x.index <= d.date.max()]
    gap = x.book - x["all"]
    e, s = d[buy & ok], d[sell & ok]
    out = {"signal": col, "buys": len(e), "held": int(np.median(n[live])), "book_yearly": round(x.book.mean() * 252, 4), "all_yearly": round(x["all"].mean() * 252, 4),
           "sharpe": round(x.book.mean() / x.book.std() * np.sqrt(252), 2), "all_sharpe": round(x["all"].mean() / x["all"].std() * np.sqrt(252), 2),
           "edge_points": round(gap.mean() * 25200, 2), "edge_t": round(nw_tstat(gap, 20), 2)}
    for name, part in (("to_2021", gap[gap.index < "2022"]), ("2022_on", gap[gap.index >= "2022"]), ("2023_on", gap[gap.index >= "2023"])):
        out["edge_" + name], out["t_" + name] = round(part.mean() * 25200, 2), round(nw_tstat(part, 20), 2)
    out.update({"buy_excess": round(e.excess.mean() - 2 * COST, 4), "buy_beat_sector": round((e.excess > 0).mean(), 3), "sell_excess": round(s.excess.mean(), 4), "sell_beat_sector": round((s.excess > 0).mean(), 3)})
    return out


def test():
    pd.set_option("display.width", 240, "display.max_columns", 30)
    t, o, c, member = events()
    print(t.groupby([t.date.dt.year, "form"]).size().unstack().to_string())
    print(t.groupby("form")[["similarity", "new_prose", "gone_prose", "old_sentences", "y", "excess"]].median().round(3).to_string())
    print("sessions from filed date to decision", (t.date - pd.to_datetime(t.filed)).dt.days.describe().round(1).to_dict(), flush=True)
    t.drop(columns=["added", "removed"]).to_parquet(D / "filing_change_events.parquet")

    words = sp.hstack([TfidfVectorizer(ngram_range=(1, 2), min_df=15, max_df=.6, max_features=n, sublinear_tf=True, stop_words="english").fit_transform(t[k])
                       for k, n in (("added", 40000), ("removed", 20000))]).tocsr()
    name = sp.csr_matrix(pd.get_dummies(t.symbol).values.astype(float))
    sect = sp.csr_matrix(pd.get_dummies(t.sector).values.astype(float))
    print("features", words.shape, flush=True)

    scored = walk_scores(t, words, "y", GAP)
    t["words"] = scored.score.reindex(t.index)
    ics = {**fixed_ics(t, "y"), "words": ic_of(t, "words", "y")}
    res = pd.DataFrame([row(k, v) for k, v in ics.items()])
    # the fitted model starts later, so the baselines and the fixed signals are also given on its months
    rest = t[t.words.notna()]
    base = pd.DataFrame([row("company identity", walk(t, name, "y", GAP)), row("sector identity", walk(t, sect, "y", GAP)),
                         *[row(k + " on the words months", v) for k, v in fixed_ics(rest, "y").items()], row("control: words on filing reaction", walk(t, words, "reaction", 15))])
    res.to_csv(D / "filing_change_ic.csv", index=False)
    base.to_csv(D / "filing_change_baselines.csv", index=False)
    print("\nsignals\n", res.to_string(index=False), "\n\nbaselines and control\n", base.to_string(index=False), flush=True)

    books = pd.DataFrame([book(t, "raw_" + k, o, c, member) for k in ("stable", "little_new", "steady_length")]
                         + [book(t[t.form == f], "raw_stable", o, c, member) | {"signal": "raw_stable " + f} for f in ("10-K", "10-Q")]
                         + [book(rest, "words", o, c, member)])
    books.to_csv(D / "filing_change_books.csv", index=False)
    print("\nbooks, 10 bps a side\n", books.T.to_string(), flush=True)

    best = res.t.max()
    rng = np.random.default_rng(0)
    null = []
    for i in range(RUNS):
        s = t.assign(y=t.y.groupby(t.month).transform(lambda x: rng.permutation(x.values)))
        tt = [nw_tstat(v, 3) for v in fixed_ics(s, "y").values()] + [nw_tstat(walk(s, words, "y", GAP), 3)]
        null.append(max(tt))
        if (i + 1) % 10 == 0:
            print("luck", i + 1, "of", RUNS, flush=True)
    null = np.array(null)
    print(f"\nbest t {best}; shuffled outcomes reached it in {(null >= best).sum()} of {RUNS} runs; shuffled best t: median {np.median(null):.2f}, 95th {np.quantile(null, .95):.2f}", flush=True)
    pd.Series(null).to_csv(D / "filing_change_luck.csv", index=False)


if __name__ == "__main__":
    {"build": build, "test": test}[sys.argv[1]]()
