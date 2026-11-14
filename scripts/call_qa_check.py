"""Cheap filter before any 9B run on earnings calls: does what is said in the Q&A say anything about
the next 60 sessions? Train period only: bars are cut at 2025-01-01, so no outcome reaches validation.

  PYTHONPATH=src nice -n 10 .venv/bin/python -W ignore scripts/call_qa_check.py build
  PYTHONPATH=src nice -n 10 .venv/bin/python -W ignore scripts/call_qa_check.py test

Transcripts: defeatbeta/yahoo-finance-data, in data/raw/transcripts/defeatbeta. It gives the call date
but no time, so the decision is at the close of the first session after that date. Entry at the next
open, exit at the close 60 sessions after the decision. Outcome: return minus sector mean, over the
stock's own volatility. A good case, as agreed with the user, is one where the price rose and rose
more than SPY's. Six signals, fixed before looking:
  answer_tone    good against bad words in management's answers
  question_tone  the same in the analysts' questions
  tone_gap       answers against prepared remarks: does management sound worse unscripted?
  straight       few dodges such as "not going to" or "too early" in the answers
  tone_change    answer tone against the company's last four calls
  words          word model on answers and questions, fit on earlier months

Bar, written before any result. One signal must do all of this:
  1. IC positive with t at least 2
  2. IC above company identity and above sector identity on the same events
  3. best t over the six signals reached by shuffled outcomes in at most 5% of runs
  4. IC positive in both 2017-2021 and 2022-2024
  5. BUY book after 10 bps a side beats owning everything, t at least 2
"""
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from filing_change_check import CUT, D, GAP, H, RUNS, book, buys, ic_of, market, place, row
from raw_text_check import walk, walk_scores
from tagent.data import ROOT, load_bars
from tagent.evaluate import nw_tstat
from tagent.risk_agent import adjusted

RAW = ROOT / "data" / "raw" / "transcripts" / "defeatbeta" / "transcripts.parquet"
FEAT = ROOT / "data" / "features" / "call_text.parquet"
FIXED = ["answer_tone", "question_tone", "tone_gap", "straight", "tone_change"]
START = re.compile(r"first question|next question|question (today |this morning |this afternoon )?(comes|is coming|will come|is from)|first caller", re.I)
WORDS = re.compile(r"[a-z]+")
GOOD = set("strong strength growth record improve improved improving improvement momentum tailwind tailwinds exceeded beat confident robust solid "
           "accelerate accelerating outperform favorable opportunity pleased".split())
BAD = set("decline declines declined weak weakness headwind headwinds challenging challenge pressure pressures softness soft delay delayed loss losses "
          "difficult uncertain uncertainty concern concerned disappointing shortfall slowdown".split())
DODGE = re.compile(r"not going to|can't comment|won't comment|don't want to|do not want to|not able to|too early|don't disclose|do not disclose|don't provide|"
                   r"do not provide|don't break out|wouldn't want to|hard to say|difficult to say|can't give|not prepared to|don't have (that|the) ", re.I)


def tone(text):
    w = WORDS.findall(text.lower())
    good, bad = sum(x in GOOD for x in w), sum(x in BAD for x in w)
    return (good - bad) / max(good + bad, 1), len(w)


def split(parts):
    """Prepared remarks, management's answers and the analysts' questions. The Q&A starts at the
    operator's first call for a question; anyone who spoke before it counts as management."""
    is_op = lambda s: (s or "").lower().startswith("operator")
    start = next((i for i, (s, x) in enumerate(parts) if i > 1 and is_op(s) and START.search(x)), None)
    if start is None:
        start = next((i for i, (s, x) in enumerate(parts) if i > 1 and START.search(x)), None)
    if start is None:
        return None
    team = {s for s, _ in parts[:start] if not is_op(s)}
    ask = [x for s, x in parts[start:] if not is_op(s) and s not in team]
    say = [x for s, x in parts[start:] if not is_op(s) and s in team]
    askers = len({s for s, _ in parts[start:] if not is_op(s) and s not in team})
    return " ".join(x for s, x in parts[:start] if not is_op(s)), " ".join(say), " ".join(ask), askers


def one(r):
    parts = [(p["speaker"], p["content"] or "") for p in r["transcripts"] or []]
    x = split(parts)
    if x is None:
        return None
    prepared, answers, questions, askers = x
    (a, an), (q, qn), (m, mn) = tone(answers), tone(questions), tone(prepared)
    return {"symbol": r["symbol"], "call": r["report_date"], "fy": r["fiscal_year"], "fq": r["fiscal_quarter"], "answer_tone": a, "question_tone": q, "prepared_tone": m,
            "answer_words": an, "question_words": qn, "prepared_words": mn, "dodges": len(DODGE.findall(answers)), "askers": askers, "answers": answers, "questions": questions}


def build():
    universe = set(pd.read_parquet(ROOT / "data" / "cases" / "train.parquet", columns=["symbol"]).symbol)
    rows, seen = [], 0
    for b in pq.ParquetFile(RAW).iter_batches(batch_size=200, columns=["symbol", "fiscal_year", "fiscal_quarter", "report_date", "transcripts"]):
        for r in b.to_pylist():
            if r["symbol"] in universe and "2016" <= r["report_date"] < "2025":
                seen += 1
                x = one(r)
                if x:
                    rows.append(x)
    d = pd.DataFrame(rows)
    d.to_parquet(FEAT)
    print(seen, "calls,", len(d), "with a Q&A found", flush=True)
    d["year"] = d.call.str[:4]
    d["share"] = d.answer_words / (d.answer_words + d.question_words + d.prepared_words)
    print(d.groupby("year")[["prepared_words", "answer_words", "question_words", "share", "answer_tone", "question_tone", "prepared_tone", "dodges"]].median().round(3).to_string(), flush=True)


def events():
    t = pd.read_parquet(FEAT)
    o, c, member, sector, ret, peers, vol = market()
    t["pos"] = c.index.searchsorted(pd.to_datetime(t.call) + pd.Timedelta(days=1))
    t = t.sort_values(["symbol", "call"])
    t["tone_change"] = t.answer_tone - t.groupby("symbol").answer_tone.transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean())
    t = place(t, c, sector, ret, peers, vol)
    print(len(t), "calls on a session;", int(t.y.notna().sum()), "with an outcome", flush=True)
    t = t[t.y.notna() & (t.answer_words >= 300) & (t.question_words >= 100)].sort_values("date").reset_index(drop=True)
    b = load_bars()
    so, _, sc = adjusted(b[(b.symbol == "SPY") & (b.date < CUT)])
    t["spy"] = (sc.shift(-H) / so.shift(-1) - 1)["SPY"].reindex(c.index).values[t.pos]
    t["good"] = (t.ret > 0) & (t.ret > t.spy)
    t["month"] = t.date.dt.to_period("M").astype(str)
    t["form"] = "call"
    t["tone_gap"] = t.answer_tone - t.prepared_tone
    t["straight"] = -t.dodges / t.answer_words * 1000
    return t, o, c, member


def fixed_ics(t, y):
    return {k: ic_of(t, k, y) for k in FIXED}


def test():
    pd.set_option("display.width", 240, "display.max_columns", 30)
    t, o, c, member = events()
    print(t.groupby(t.date.dt.year).size().to_dict(), t.symbol.nunique(), "names; all stocks good", round(t.good.mean(), 3), flush=True)
    print("sessions from call to decision", (t.date - pd.to_datetime(t.call)).dt.days.describe().round(1).to_dict(), flush=True)
    print(t[FIXED].corrwith(t.reaction).round(3).to_dict(), "correlation with the reaction", flush=True)
    t.drop(columns=["answers", "questions"]).to_parquet(D / "call_qa_events.parquet")

    vec = lambda: TfidfVectorizer(min_df=15, max_df=.6, max_features=30000, sublinear_tf=True, stop_words="english")
    words = sp.hstack([vec().fit_transform(t.answers), vec().fit_transform(t.questions)]).tocsr()
    t = t.drop(columns=["answers", "questions"])
    name = sp.csr_matrix(pd.get_dummies(t.symbol).values.astype(float))
    sect = sp.csr_matrix(pd.get_dummies(t.sector).values.astype(float))
    print("features", words.shape, flush=True)

    t["words"] = walk_scores(t, words, "y", GAP).score.reindex(t.index)
    ics = {**fixed_ics(t, "y"), "words": ic_of(t, "words", "y")}
    res = pd.DataFrame([row(k, v) for k, v in ics.items()])
    rest = t[t.words.notna()]
    base = pd.DataFrame([row("company identity", walk(t, name, "y", GAP)), row("sector identity", walk(t, sect, "y", GAP)),
                         *[row(k + " on the words months", v) for k, v in fixed_ics(rest, "y").items()], row("control: words on the reaction", walk(t, words, "reaction", 15))])
    res.to_csv(D / "call_qa_ic.csv", index=False)
    base.to_csv(D / "call_qa_baselines.csv", index=False)
    print("\nsignals\n", res.to_string(index=False), "\n\nbaselines and control\n", base.to_string(index=False), flush=True)

    rows = []
    for k in FIXED + ["words"]:
        d = rest if k == "words" else t.dropna(subset=[k])
        b = book(d, k, o, c, member)
        buy, _, ok = buys(d, k)
        rows.append({**b, "buy_price_up_and_beat_spy": round(d[buy & ok].good.mean(), 3), "all_price_up_and_beat_spy": round(d.good.mean(), 3)})
    books = pd.DataFrame(rows)
    books.to_csv(D / "call_qa_books.csv", index=False)
    print("\nbooks, 10 bps a side\n", books.T.to_string(), flush=True)

    best = res.t.max()
    rng = np.random.default_rng(0)
    null = []
    for i in range(RUNS):
        s = t.assign(y=t.y.groupby(t.month).transform(lambda x: rng.permutation(x.values)))
        null.append(max([nw_tstat(v, 3) for v in fixed_ics(s, "y").values()] + [nw_tstat(walk(s, words, "y", GAP), 3)]))
        if (i + 1) % 20 == 0:
            print("luck", i + 1, "of", RUNS, flush=True)
    null = np.array(null)
    print(f"\nbest t {best}; shuffled outcomes reached it in {(null >= best).sum()} of {RUNS} runs; shuffled best t: median {np.median(null):.2f}, 95th {np.quantile(null, .95):.2f}", flush=True)
    pd.Series(null).to_csv(D / "call_qa_luck.csv", index=False)


if __name__ == "__main__":
    {"build": build, "test": test}[sys.argv[1]]()
