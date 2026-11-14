"""Cheap filter before any 9B run on smaller stocks: the call test of call_qa_check.py, moved to names
with less attention. In large names the text was priced by the first close after the call; does that
hold where fewer people are looking? Train period only: bars are cut at 2025-01-01.

  PYTHONPATH=src nice -n 10 .venv/bin/python -W ignore scripts/small_call_check.py build
  PYTHONPATH=src nice -n 10 .venv/bin/python -W ignore scripts/small_call_check.py test
  PYTHONPATH=src nice -n 10 .venv/bin/python -W ignore scripts/small_call_check.py luck [runs]
  PYTHONPATH=src nice -n 10 .venv/bin/python -W ignore scripts/small_call_check.py follow

Everything comes from defeatbeta/yahoo-finance-data, in data/raw/defeatbeta and data/raw/transcripts.
Timing and outcome as in call_qa_check.py: decision at the close of the first session after the call
date, entry at the next open, exit 60 sessions after the decision, return minus sector mean over the
stock's own volatility. Dividends are added back from the dividend table.

Universe, set each day from past data only: common stock outside the 545 names already tested, not a
shell company, a sector on file, price at least $3 before later splits, and median dollar volume of
the last 60 sessions between $1M and $50M.

Cleaning, added after the first IC table and before any book result: a day above +500% is a break in
the series (real moves stop near +470%), so the stock's history before it is dropped; a call filed
under two tickers counts once.

Eight signals, fixed before looking. Six as in call_qa_check.py; two need no text:
  reaction     move from the close before the call to the decision close, in volatility units
  eps_change   EPS of the quarter against a year earlier, over price

Bar, written before any result. One signal must do all of this:
  1. IC positive with t at least 2
  2. IC above company identity and above sector identity on the same events
  3. best t over the eight signals reached by shuffled outcomes in at most 5% of runs
  4. IC positive in 2016-2021, in 2022-2024 and in 2023-2024
  5. BUY book after 40 bps a side beats owning everything, t at least 2, and the edge is positive
     in 2023-2024

Failed companies are mostly missing from this source, so early years see survivors only. A 2024 event
only had to survive into 2025, so 2023-2024 is the check that matters. A pass on reaction or eps_change
alone would be a lead for a plain rule, not for reading text. IC by size and by number of analysts on
the call is printed for reading only and counts toward nothing.
"""
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

import filing_change_check as fc
from call_qa_check import FIXED as TEXT, RAW, one
from filing_change_check import CUT, D, GAP, H, RUNS, book, buys, ic_of, place, row
from raw_text_check import walk, walk_scores
from tagent.data import ROOT, load_bars
from tagent.evaluate import nw_tstat
from tagent.risk_agent import adjusted

SRC = ROOT / "data" / "raw" / "defeatbeta"
FEAT = ROOT / "data" / "features" / "small_call_features.parquet"
TEXTS = ROOT / "data" / "features" / "small_call_text.parquet"
WORDS = ROOT / "data" / "features" / "small_call_words.npz"
LOW, HIGH, MIN_PRICE, BREAK = 1e6, 50e6, 3, 5
PLAIN = ["reaction", "eps_change"]
CONTROLS = ["size", "past", "risk", "reaction"]
FIXED = TEXT + PLAIN
fc.COST = .004


def build():
    tested = set(pd.read_parquet(ROOT / "data" / "cases" / "train.parquet", columns=["symbol"]).symbol)
    rows, texts, seen, writer = [], [], 0, None

    def flush():
        nonlocal writer, texts
        table = pa.Table.from_pylist(texts)
        writer = writer or pq.ParquetWriter(TEXTS, table.schema)
        writer.write_table(table)
        texts = []

    for b in pq.ParquetFile(RAW).iter_batches(batch_size=200, columns=["symbol", "fiscal_year", "fiscal_quarter", "report_date", "transcripts"]):
        for r in b.to_pylist():
            if r["symbol"].isalpha() and r["symbol"] not in tested and "2016" <= r["report_date"] < "2025":
                seen += 1
                x = one(r)
                if x:
                    n = len(rows)
                    texts.append({"row": n, "answers": x.pop("answers"), "questions": x.pop("questions")})
                    rows.append({"row": n, **x})
                    if len(texts) == 2000:
                        flush()
                if seen % 20000 == 0:
                    print(seen, "calls", flush=True)
    flush()
    writer.close()
    d = pd.DataFrame(rows)
    d.to_parquet(FEAT)
    print(seen, "calls,", len(d), "with a Q&A found,", d.symbol.nunique(), "names", flush=True)
    print(d.groupby(d.call.str[:4])[["prepared_words", "answer_words", "question_words", "askers", "answer_tone", "question_tone", "dodges"]].median().round(3).to_string(), flush=True)


def wide(d, field, days):
    return d.pivot(index="date", columns="symbol", values=field).reindex(days)


def market(symbols):
    """Price tables with dividends added back, the universe and each stock's 60-session outcome."""
    p = pq.read_table(SRC / "stock_prices.parquet", filters=[("report_date", ">=", "2015-01-01"), ("report_date", "<", str(CUT.date())), ("symbol", "in", list(symbols))])
    for k in ("open", "close"):
        p = p.set_column(p.schema.get_field_index(k), k, pc.cast(p[k], pa.float64()))
    p = p.select(["symbol", "report_date", "open", "close", "volume"]).to_pandas()
    p["date"] = pd.to_datetime(p.report_date)
    p = p[(p.close > 0) & (p.open > 0)].drop_duplicates(["symbol", "date"])
    b = load_bars()
    so, _, sc = adjusted(b[(b.symbol == "SPY") & (b.date < CUT)])
    days = sc.index[sc.index >= "2015-01-01"]
    o, c, v = wide(p, "open", days), wide(p, "close", days), wide(p, "volume", days).astype(float)
    del p

    prof = pd.read_parquet(SRC / "stock_profile.parquet", columns=["symbol", "sector", "industry"]).drop_duplicates("symbol").set_index("symbol")
    prof = prof[(prof.sector.fillna("") != "") & (prof.industry != "Shell Companies")]
    keep = c.columns.intersection(prof.index)
    o, c, v = o[keep], c[keep], v[keep]
    sector = prof.sector.reindex(keep)
    broken = c.pct_change(fill_method=None).gt(BREAK)
    print("series breaks", {k: str(x.index[x].max().date()) for k, x in broken.items() if x.any()}, flush=True)
    whole = broken.iloc[::-1].cummax().iloc[::-1].shift(-1, fill_value=False)
    o, c, v = o.mask(whole), c.mask(whole), v.mask(whole)

    later = pd.DataFrame(1.0, index=days, columns=keep)  # the price then, before splits made since
    s = pd.read_parquet(SRC / "stock_split_events.parquet")
    s = s[s.symbol.isin(keep)]
    for r in s.itertuples():
        a, z = r.split_factor.split(":")
        later.loc[later.index < r.report_date, r.symbol] *= float(a) / float(z)
    paid = c * later
    dollars = (c * v).rolling(60, min_periods=40).median()
    member = dollars.ge(LOW) & dollars.le(HIGH) & paid.ge(MIN_PRICE)

    d = pd.read_parquet(SRC / "stock_dividend_events.parquet")
    d = d[d.symbol.isin(keep)].assign(date=lambda x: pd.to_datetime(x.report_date), amount=lambda x: x.amount.astype(float))
    d = d.groupby(["date", "symbol"]).amount.sum().unstack().reindex(index=days, columns=keep).fillna(0)
    k = (1 + (d / c.shift(1)).clip(upper=.5).fillna(0)).cumprod()
    o, c = o * k, c * k

    ret = (c.shift(-H) / o.shift(-1) - 1).clip(-.9, 3).where(member)
    vol = c.pct_change(fill_method=None).rolling(60).std().replace(0, np.nan)
    return o, c, member, sector, ret, ret.T.groupby(sector).mean().T, vol, dollars, paid, so["SPY"], sc["SPY"]


def eps_change(t):
    """EPS of the last quarter ended at least 5 days before the call, against a year earlier."""
    e = pd.read_parquet(SRC / "stock_tailing_eps.parquet", columns=["symbol", "report_date", "eps"])
    e = e[e.symbol.isin(t.symbol.unique())].assign(end=lambda x: pd.to_datetime(x.report_date), eps=lambda x: pd.to_numeric(x.eps, errors="coerce"))
    e = e.dropna(subset=["eps"]).drop_duplicates(["symbol", "end"]).sort_values(["symbol", "end"])
    year = (e.end - e.groupby("symbol").end.shift(4)).dt.days
    e["yoy"] = (e.eps - e.groupby("symbol").eps.shift(4)).where(year.between(350, 380))
    x = t.assign(seen=pd.to_datetime(t.call) - pd.Timedelta(days=5)).sort_values("seen")
    x = pd.merge_asof(x, e[["symbol", "end", "yoy"]].sort_values("end"), left_on="seen", right_on="end", by="symbol", direction="backward", tolerance=pd.Timedelta(days=100))
    return x.set_index("row").yoy.reindex(t.row).values


def events():
    t = pd.read_parquet(FEAT)
    o, c, member, sector, ret, peers, vol, dollars, paid, so, sc = market(t.symbol.unique())
    t["pos"] = c.index.searchsorted(pd.to_datetime(t.call) + pd.Timedelta(days=1))
    t = t.sort_values(["symbol", "call"])
    t["tone_change"] = t.answer_tone - t.groupby("symbol").answer_tone.transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean())
    t = place(t, c, sector, ret, peers, vol)
    print(len(t), "calls on a session;", int(t.y.notna().sum()), "with an outcome in the universe", flush=True)
    t = t[t.y.notna() & np.isfinite(t.y) & (t.answer_words >= 300) & (t.question_words >= 100)].sort_values(["date", "symbol"])
    n = len(t)
    t = t.drop_duplicates(["call", "prepared_words", "answer_words", "question_words"]).reset_index(drop=True)
    print(n - len(t), "calls filed under a second ticker dropped", flush=True)
    t["eps_change"] = eps_change(t) / paid.values[t.pos, t.col]
    t["dollars"] = dollars.values[t.pos, t.col]
    t["spy"] = (sc.shift(-H) / so.shift(-1) - 1).reindex(c.index).values[t.pos]
    t["good"] = (t.ret > 0) & (t.ret > t.spy)
    t["month"] = t.date.dt.to_period("M").astype(str)
    t["form"] = "call"
    t["tone_gap"] = t.answer_tone - t.prepared_tone
    t["straight"] = -t.dodges / t.answer_words * 1000
    return t, o, c, member


def text_matrix(t):
    """Word counts for the kept calls, in t's order. Text is streamed: it does not fit in memory beside the 9B."""
    keep = set(t.row)

    def stream(col):
        for b in pq.ParquetFile(TEXTS).iter_batches(batch_size=500, columns=["row", col]):
            for n, x in zip(b["row"].to_pylist(), b[col].to_pylist()):
                if n in keep:
                    yield x

    vec = lambda: TfidfVectorizer(min_df=15, max_df=.6, max_features=30000, sublinear_tf=True, stop_words="english", dtype=np.float32)
    X = sp.hstack([vec().fit_transform(stream("answers")), vec().fit_transform(stream("questions"))]).tocsr()
    return X[np.searchsorted(np.sort(t.row.values), t.row.values)]


def full_row(name, ic):
    late = ic[ic.index >= "2023"]
    return {**row(name, ic), "ic_2023_on": round(late.mean(), 4), "t_2023_on": round(nw_tstat(late, 3), 2)}


def fixed_ics(t, y):
    return {k: ic_of(t, k, y) for k in FIXED}


def test():
    pd.set_option("display.width", 240, "display.max_columns", 30)
    t, o, c, member = events()
    print(t.groupby(t.date.dt.year).size().to_dict(), t.symbol.nunique(), "names; all stocks good", round(t.good.mean(), 3), flush=True)
    print("universe per day", member.sum(1).iloc[[300, 800, 1300, 1800, 2300]].to_dict(), flush=True)
    day = c.pct_change(fill_method=None).where(member)
    print("member days", int(day.notna().sum().sum()), "above +100%", int((day > 1).sum().sum()), "below -50%", int((day < -.5).sum().sum()), "worst", round(float(day.min().min()), 3), "best", round(float(day.max().max()), 3), flush=True)
    print(t[["dollars", "askers", "ret", "excess", "y"]].describe().round(3).to_string(), flush=True)
    print("with eps_change", round(t.eps_change.notna().mean(), 3), "with tone_change", round(t.tone_change.notna().mean(), 3), flush=True)
    print(t[FIXED].corrwith(t.reaction).round(3).to_dict(), "correlation with the reaction", flush=True)
    t.to_parquet(D / "small_call_events.parquet")

    words = text_matrix(t)
    sp.save_npz(WORDS, words)
    name = sp.csr_matrix(pd.get_dummies(t.symbol).values.astype(np.float32))
    sect = sp.csr_matrix(pd.get_dummies(t.sector).values.astype(np.float32))
    print("features", words.shape, flush=True)

    t["words"] = walk_scores(t, words, "y", GAP).score.reindex(t.index)
    t.to_parquet(D / "small_call_events.parquet")
    ics = {**fixed_ics(t, "y"), "words": ic_of(t, "words", "y")}
    res = pd.DataFrame([full_row(k, v) for k, v in ics.items()])
    res.to_csv(D / "small_call_ic.csv", index=False)
    print("\nsignals\n", res.to_string(index=False), flush=True)

    rest = t[t.words.notna()]
    base = pd.DataFrame([full_row("company identity", walk(t, name, "y", GAP)), full_row("sector identity", walk(t, sect, "y", GAP)),
                         *[full_row(k + " on the words months", v) for k, v in fixed_ics(rest, "y").items()], full_row("control: words on the reaction", walk(t, words, "reaction", 15))])
    base.to_csv(D / "small_call_baselines.csv", index=False)
    print("\nbaselines and control\n", base.to_string(index=False), flush=True)

    rows = []
    for k in FIXED + ["words"]:
        d = rest if k == "words" else t.dropna(subset=[k])
        buy, _, ok = buys(d, k)
        if not buy.any():  # straight: over 30% of calls tie at the top, so nothing clears the line
            print(k, "has no BUY", flush=True)
            continue
        b = book(d, k, o, c, member)
        rows.append({**b, "buy_price_up_and_beat_spy": round(d[buy & ok].good.mean(), 3), "all_price_up_and_beat_spy": round(d.good.mean(), 3)})
    books = pd.DataFrame(rows)
    books.to_csv(D / "small_call_books.csv", index=False)
    print(f"\nbooks, {fc.COST * 1e4:.0f} bps a side\n", books.T.to_string(), flush=True)

    few = t.askers <= t.askers.median()
    groups = {"under $5M": t.dollars < 5e6, "$5M to $20M": t.dollars.between(5e6, 20e6), "over $20M": t.dollars > 20e6, "fewer analysts": few, "more analysts": ~few}
    tiers = pd.DataFrame([{"group": g, "calls": int(m.sum()), **{k: round(ic_of(t[m], k, "y").mean(), 4) for k in FIXED + ["words"]}} for g, m in groups.items()])
    tiers.to_csv(D / "small_call_tiers.csv", index=False)
    print("\nIC by group, for reading only\n", tiers.to_string(index=False), flush=True)


def hold(d, o, c, cost):
    """Daily return of a book holding each row of d from the next open for 60 sessions, as in fc.book."""
    held = np.zeros(c.shape, bool)
    for r in d.itertuples():
        held[r.pos + 1:r.pos + H + 1, r.col] = True
    was = np.vstack([np.zeros((1, c.shape[1]), bool), held[:-1]])
    enter, leave = held & ~was, ~held & was
    day = np.where(enter, (c / o - 1).values, c.pct_change(fill_method=None).values)
    n = held.sum(1)
    live = n > 0
    net = np.nansum(np.where(held, day, 0), 1)[live] / n[live] - (enter.sum(1) + leave.sum(1))[live] / n[live] * cost
    return pd.Series(net, index=c.index[live])


def with_controls(t, c):
    pos, col = t.pos.values, t.col.values
    return t.assign(past=c.values[pos, col] / c.values[np.maximum(pos - 60, 0), col] - 1,
                    risk=c.pct_change(fill_method=None).rolling(60).std().values[pos, col], size=np.log(t.dollars))


def month_coefs(d, cols, least=50):
    """Each month, the ranked outcome on the ranked inputs. One row of coefficients per month."""
    r = d.groupby("month")[cols + ["y"]].rank(pct=True) - .5
    return pd.DataFrame({m: np.linalg.lstsq(g[cols].values, g.y.values, rcond=None)[0] for m, g in r.groupby(d.month) if len(g) >= least}, index=cols).T


def follow():
    """Written after the book results, so nothing here counts toward the bar. Every book lost to owning
    everything by about the same amount, so: is that the signal, or the benchmark? Compares each BUY book
    with a book that buys after every call on the same terms, then asks whether tone is more than size,
    recent return, volatility and the price reaction."""
    pd.set_option("display.width", 240, "display.max_columns", 30)
    t, o, c, member = events()
    every = c.pct_change(fill_method=None).where(member).mean(axis=1)
    last = t.date.max()
    points = lambda x: round(x.mean() * 25200, 2)
    rows = []
    for cost in (0, fc.COST):
        calls = hold(t, o, c, cost)
        rows.append({"signal": "every call", "cost_bps": cost * 1e4, "yearly": points(calls[calls.index <= last])})
        for k in ("answer_tone", "question_tone", "tone_change", "reaction"):
            d = t.dropna(subset=[k])
            buy, sell, ok = buys(d, k)
            x, z = hold(d[buy & ok], o, c, cost), hold(d[sell & ok], o, c, cost)
            g = pd.concat([x, calls, z], axis=1, keys=["buy", "calls", "sell"]).dropna()
            g = g[g.index <= last]
            pick, spread, late = g.buy - g.calls, g.buy - g.sell, g[g.index >= "2023"]
            rows.append({"signal": k, "cost_bps": cost * 1e4, "yearly": points(g.buy), "vs_every_call": points(pick), "t": round(nw_tstat(pick, 20), 2),
                         "vs_call_2023_on": points(late.buy - late.calls), "t_2023_on": round(nw_tstat(late.buy - late.calls, 20), 2),
                         "buy_minus_sell": points(spread), "spread_t": round(nw_tstat(spread, 20), 2)})
    rows.append({"signal": "owning everything", "cost_bps": 0, "yearly": points(every[every.index <= last])})
    res = pd.DataFrame(rows)
    res.to_csv(D / "small_call_follow_books.csv", index=False)
    print("\nBUY book against buying after every call\n", res.to_string(index=False), flush=True)

    t = with_controls(t, c)
    rows = []
    for k in ("answer_tone", "question_tone", "tone_change"):
        cols = [k, *CONTROLS]
        d = t.dropna(subset=cols + ["y"])
        b = month_coefs(d, cols)
        late = b[b.index >= "2023"]
        rows.append({"signal": k, "months": len(b), **{f"{x}_coef": round(b[x].mean(), 4) for x in cols[1:]}, "coef": round(b[k].mean(), 4), "t": round(nw_tstat(b[k], 3), 2),
                     "coef_2023_on": round(late[k].mean(), 4), "t_2023_on": round(nw_tstat(late[k], 3), 2),
                     "corr_size": round(d[k].corr(d["size"], method="spearman"), 3), "corr_past": round(d[k].corr(d.past, method="spearman"), 3), "corr_risk": round(d[k].corr(d.risk, method="spearman"), 3)})
    res = pd.DataFrame(rows)
    res.to_csv(D / "small_call_follow_controls.csv", index=False)
    print("\ntone after size, past return, risk and reaction\n", res.T.to_string(), flush=True)

    x = t.dropna(subset=["answer_tone"]).copy()
    x["fifth"] = x.groupby("month").answer_tone.rank(pct=True).mul(5).clip(upper=4.999).astype(int) + 1
    five = x.groupby("fifth").agg(calls=("y", "size"), y=("y", "mean"), excess=("excess", "mean"), median_excess=("excess", "median"), beat_sector=("excess", lambda s: (s > 0).mean()),
                                  up=("ret", lambda s: (s > 0).mean()), good=("good", "mean"), big_loss=("excess", lambda s: (s < -.3).mean())).round(4)
    five.to_csv(D / "small_call_follow_fifths.csv")
    print("\nanswer tone, by fifth within the month\n", five.to_string(), flush=True)


def luck(runs):
    t = pd.read_parquet(D / "small_call_events.parquet")
    words = sp.load_npz(WORDS)
    best = pd.read_csv(D / "small_call_ic.csv").t.max()
    rng = np.random.default_rng(0)
    null = []
    for i in range(runs):
        s = t.assign(y=t.y.groupby(t.month).transform(lambda x: rng.permutation(x.values)))
        null.append(max([nw_tstat(v, 3) for v in fixed_ics(s, "y").values()] + [nw_tstat(walk(s, words, "y", GAP), 3)]))
        pd.Series(null).to_csv(D / "small_call_luck.csv", index=False)
        print("luck", i + 1, "of", runs, flush=True)
    null = np.array(null)
    print(f"\nbest t {best}; shuffled outcomes reached it in {(null >= best).sum()} of {runs} runs; shuffled best t: median {np.median(null):.2f}, 95th {np.quantile(null, .95):.2f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "luck":
        luck(int(sys.argv[2]) if len(sys.argv) > 2 else RUNS)
    else:
        {"build": build, "test": test, "follow": follow}[sys.argv[1]]()
