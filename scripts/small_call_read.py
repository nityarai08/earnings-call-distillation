"""Can the 9B read earnings calls better than a list of 40 words? No training: one forward pass per call.
Follows small_call_check.py, where answer tone had a real but small link to the next 60 sessions.
Train period only: outcomes end before 2025.

  PYTHONPATH=src nice -n 10 .venv/bin/python -W ignore scripts/small_call_read.py sample
  nohup scripts/run_small_call_read.sh > logs/small_call_read_chain.log 2>&1 &
  PYTHONPATH=src nice -n 10 .venv/bin/python -W ignore scripts/small_call_read.py test

Sample: 2,000 calls from 2023-2024, the years least affected by missing failed companies, spread by
month. The reader gets the same text as the word list, management's answers, up to 2,500 words, with
the company name, ticker and years masked. It rates the outlook from 1 to 5, and the score is the
expected rating from its probabilities on the five digits. One question, so one signal.

Months run from 15 calls to 219, so every monthly number here is weighted by calls, and t comes from
the spread of months around the weighted mean. Unweighted, random scores gave IC 0.043 in a dry run.
Weighted, the error on IC is about 0.022, so this cannot confirm an IC of 0.024. It can see one of
about 0.05, and that is about what the word list's 1.1% top-to-bottom spread would need to clear a
0.8% round trip with room to spare. A reader only as good as the word list counts as a fail here.

Bar, written before any result. The reader must do all of this:
  1. IC at least 0.05, with t at least 2
  2. coefficient positive with t at least 2 after word tone, size, past return, risk and reaction
  3. its t reached by shuffled outcomes in at most 5% of 1,000 runs
  4. IC positive in 2023 and in 2024
  5. before costs, top fifth at least 0.8% above all calls and at least 2.0% above the bottom fifth
Control: IC against the reaction around the call at least 0.10. Below that the reader did not work
and the test says nothing either way.

Pass: score the other 2023-2024 calls and run the real book test before any SFT. Fail: stop on
transcripts. Fifths are set within the month, so a pass here is not yet a tradable result. The model
may have seen these years in training, and masking does not hide a company from it fully (it named 76%
of masked filings), so a pass still needs checking on later data.
"""
import json
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from filing_change_check import D, ic_of
from small_call_check import CONTROLS, TEXTS, events, month_coefs, with_controls
from tagent.data import ROOT
from tagent.earnings_text import mask
from tagent.evaluate import nw_tstat

SAMPLE = ROOT / "data" / "features" / "small_call_read_sample.parquet"
OUT = D / "small_call_read.csv"
MODEL = "mlx-community/Qwen3.5-9B-4bit"
N, MAX_WORDS, RUNS = 2000, 2500, 1000
SYSTEM = "You are an experienced stock analyst. You read what a company's managers said when answering analysts on a quarterly earnings call. The company's name is masked."
ASK = ("Based only on these answers, rate how this company's business is likely to do over the next two quarters. "
       "1 = much worse than now, 2 = somewhat worse, 3 = about the same, 4 = somewhat better, 5 = much better. "
       "Managers sound positive by habit, so weigh specifics: orders, pricing, margins, customers, and questions they avoided. Reply with one digit.")
PREFIX = "Rating: "


def names(title):
    n = re.sub(r"[,.]", " ", title).strip()
    core = re.sub(r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|trust|/de|/new)\b", "", n, flags=re.I).strip()
    first = core.split()[0] if core.split() else ""
    return sorted({x for x in (n, core, first) if len(x) >= 4}, key=len, reverse=True)


def sample():
    t = pd.read_parquet(D / "small_call_events.parquet", columns=["row", "symbol", "month", "date"])
    t = t[t.date >= "2023"]
    t = t[t.groupby("month").row.transform("size") >= 15]
    share = t.groupby("month").size() / len(t)
    take = (share * N).round().astype(int).clip(lower=15)
    s = pd.concat(g.sample(n=min(take[m], len(g)), random_state=0) for m, g in t.groupby("month"))
    titles = {x["ticker"]: x["title"] for x in json.loads((ROOT / "data" / "raw" / "defeatbeta" / "company_tickers.json").read_text()).values()}
    keep = set(s.row)
    text = {}
    for b in pq.ParquetFile(TEXTS).iter_batches(batch_size=500, columns=["row", "answers"]):
        for n, x in zip(b["row"].to_pylist(), b["answers"].to_pylist()):
            if n in keep:
                text[n] = x
    s["text"] = [re.sub(r"\b(19|20)\d\d\b|[‘’']\d\d\b", "[year]", mask(" ".join(text[r.row].split()[:MAX_WORDS]), names(titles.get(r.symbol, "")), [r.symbol])) for r in s.itertuples()]
    s["named"] = s.symbol.isin(titles)
    s = s.sample(frac=1, random_state=0).reset_index(drop=True)  # any stop leaves an unbiased part
    s.to_parquet(SAMPLE)
    print(len(s), "calls,", s.symbol.nunique(), "names, with a name to mask", round(s.named.mean(), 3), "median words", int(s.text.str.split().str.len().median()), flush=True)
    print(s.groupby("month").size().to_dict(), flush=True)


def read(limit):
    import mlx.core as mx
    from mlx_lm import load

    s = pd.read_parquet(SAMPLE)
    seen = set(pd.read_csv(OUT).row) if OUT.exists() else set()
    if not OUT.exists():
        OUT.write_text("row,tokens,seconds,p1,p2,p3,p4,p5\n")
    todo = s[~s.row.isin(seen)].head(limit)
    if todo.empty:
        print("nothing to read", flush=True)
        return
    model, tok = load(MODEL)
    head = tok.encode(PREFIX)
    ids = [tok.encode(PREFIX + str(k))[len(head)] for k in range(1, 6)]
    with OUT.open("a") as f:
        for r in todo.itertuples():
            m = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Management's answers:\n\n{r.text}\n\n{ASK}"}]
            x = tok.encode(tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False) + PREFIX)
            start = time.time()
            p = mx.softmax(model(mx.array([x]))[0, -1].astype(mx.float32), axis=-1)
            mx.eval(p)
            f.write(f"{r.row},{len(x)},{time.time() - start:.1f}," + ",".join(str(float(p[i])) for i in ids) + "\n")
            f.flush()
            mx.clear_cache()
    print(len(seen) + len(todo), "of", len(s), flush=True)


def weighted(x, calls):
    """Mean of monthly numbers weighted by calls, and its t from the months' spread around that mean."""
    w = calls.reindex(x.index)
    m = (w * x).sum() / w.sum()
    return m, m * w.sum() / np.sqrt((w ** 2 * (x - m) ** 2).sum())


def test():
    pd.set_option("display.width", 200, "display.max_columns", 30)
    r = pd.read_csv(OUT)
    p = r[[f"p{k}" for k in range(1, 6)]]
    r["reader"] = (p.values * np.arange(1, 6)).sum(1) / p.sum(1)
    print(len(r), "calls; mass on the digits", round(p.sum(1).mean(), 3), "mean tokens", int(r.tokens.mean()), "seconds", round(r.seconds.mean(), 1), flush=True)
    print("top digit", (p.idxmax(axis=1).value_counts(normalize=True).round(3)).to_dict(), "score sd", round(r.reader.std(), 3), flush=True)
    t, o, c, member = events()
    t = with_controls(t, c).merge(r[["row", "reader"]], on="row")
    print(len(t), "calls matched", flush=True)
    print("reader against word tone", round(t.reader.corr(t.answer_tone, method="spearman"), 3), "against reaction", round(t.reader.corr(t.reaction, method="spearman"), 3), flush=True)

    calls = t.groupby("month").size()
    ic = ic_of(t, "reader", "y")
    level, tval = weighted(ic, calls)
    words, words_t = weighted(ic_of(t, "answer_tone", "y"), calls)
    control, control_t = weighted(ic_of(t, "reader", "reaction"), calls)
    cols = ["reader", "answer_tone", *CONTROLS]
    coef, coef_t = weighted(month_coefs(t.dropna(subset=cols + ["y"]), cols, 15).reader, calls)
    rng = np.random.default_rng(0)
    null = np.array([weighted(ic_of(t.assign(y=t.y.groupby(t.month).transform(lambda x: rng.permutation(x.values))), "reader", "y"), calls) for _ in range(RUNS)])
    y23, y24 = weighted(ic[ic.index < "2024"], calls)[0], weighted(ic[ic.index >= "2024"], calls)[0]
    t["fifth"] = t.groupby("month").reader.rank(pct=True).mul(5).clip(upper=4.999).astype(int) + 1
    five = t.groupby("fifth").agg(calls=("y", "size"), excess=("excess", "mean"), median_excess=("excess", "median"), beat_sector=("excess", lambda s: (s > 0).mean()), up=("ret", lambda s: (s > 0).mean()), good=("good", "mean")).round(4)
    five.to_csv(D / "small_call_read_fifths.csv")
    print("\nby fifth within the month\n", five.to_string(), flush=True)
    above_all, above_low = five.excess[5] - t.excess.mean(), five.excess[5] - five.excess[1]

    rows = [("1. IC at least 0.05, t at least 2", f"IC {level:.4f}, t {tval:.2f}, word tone on these calls {words:.4f} (t {words_t:.2f})", level >= .05 and tval >= 2),
            ("2. adds to word tone and controls, t at least 2", f"coefficient {coef:.4f}, t {coef_t:.2f}", coef > 0 and coef_t >= 2),
            ("3. luck at most 5%", f"{(null[:, 1] >= tval).mean():.1%} of {RUNS}; shuffled IC sd {null[:, 0].std():.4f}, t 95th {np.quantile(null[:, 1], .95):.2f}", (null[:, 1] >= tval).mean() <= .05),
            ("4. IC positive in both years", f"2023 {y23:.4f}, 2024 {y24:.4f}", y23 > 0 and y24 > 0),
            ("5. top fifth 0.8% above all, 2.0% above bottom", f"{above_all:.2%} and {above_low:.2%}", above_all >= .008 and above_low >= .02),
            ("control: IC on the reaction at least 0.10", f"{control:.4f} (t {control_t:.2f})", control >= .1)]
    res = pd.DataFrame(rows, columns=["test", "result", "passed"])
    res.to_csv(D / "small_call_read_bar.csv", index=False)
    print("\n", res.to_string(index=False), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "read":
        read(int(sys.argv[2]))
    else:
        {"sample": sample, "test": test}[mode]()
