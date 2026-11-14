"""Datasets for the paper runs: one row per call, with sets fixed and hashed before any experiment.

  PYTHONPATH=src:scripts nice -n 10 .venv/bin/python -W ignore scripts/paper/build_data.py parse
  PYTHONPATH=src:scripts nice -n 10 .venv/bin/python -W ignore scripts/paper/build_data.py market
  PYTHONPATH=src:scripts nice -n 10 .venv/bin/python -W ignore scripts/paper/build_data.py sets

parse   reads 2025-2026 calls the same way small_call_check.py read 2016-2024. Rows start at 200,000.
market  builds calls.parquet, which holds what was knowable at the decision, and outcomes, which do not.
        The universe uses past data only and no filter uses an outcome, so a stock that later failed
        stays in with a missing outcome. Rows up to 2024 must match the checked small_call_events.
sets    assigns every call to one set. Sets are nested, so the mini run's calls are the first part of
        the week's and the week's of the full run's. Test-S is split at random within each month.

Test outcomes go to data/paper/sealed/<part>.parquet and are not read here or by anything but score.py.
"""
import json
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
from common import DATA, FREEZE, ROOT, SEALED, sha, write_atomic  # noqa: E402
import small_call_check as sc  # noqa: E402
from call_qa_check import RAW, one  # noqa: E402

FIRST_NEW = 200_000
END = pd.Timestamp("2026-11-06")
NEW_TEXT = DATA / "text_new.parquet"
NEW_FEAT = DATA / "feat_new.parquet"
CALLS = DATA / "calls.parquet"
OUTCOMES = DATA / "outcomes.parquet"
ALL_OUT = DATA / ".outcomes_all.parquet"
VAL = ("2025-01-02", "2025-11-28")
TEST = ("2026-03-02", "2026-08-06")  # last decision with 60 sessions of prices
TRAIN = ("2021-01-01", "2024-10-04")
TEST_D, TEST_W = 100, 800
SFT_N, SFT_VAL, RL_N, VAL_N, CHECK_N = 3000, 300, 800, 3000, 1000


def parse():
    tested = set(pd.read_parquet(ROOT / "data" / "cases" / "train.parquet", columns=["symbol"]).symbol)
    rows, texts = [], []
    for b in pq.ParquetFile(RAW).iter_batches(batch_size=200, columns=["symbol", "fiscal_year", "fiscal_quarter", "report_date", "transcripts"]):
        for r in b.to_pylist():
            if r["symbol"].isalpha() and r["symbol"] not in tested and "2025" <= r["report_date"] < str(END.date()):
                x = one(r)
                if x:
                    n = FIRST_NEW + len(rows)
                    texts.append({"row": n, "answers": x.pop("answers"), "questions": x.pop("questions")})
                    rows.append({"row": n, **x})
    pq.write_table(pa.Table.from_pylist(texts), NEW_TEXT)
    pd.DataFrame(rows).to_parquet(NEW_FEAT)
    print(len(rows), "calls with a Q&A found", flush=True)


def next_eps(t):
    """EPS against a year earlier: for the quarter just reported (known) and the next one (outcome)."""
    e = pd.read_parquet(sc.SRC / "stock_tailing_eps.parquet", columns=["symbol", "report_date", "eps"])
    e = e[e.symbol.isin(t.symbol.unique())].assign(end=lambda x: pd.to_datetime(x.report_date), eps=lambda x: pd.to_numeric(x.eps, errors="coerce"))
    e = e.dropna(subset=["eps"]).drop_duplicates(["symbol", "end"]).sort_values(["symbol", "end"])
    g = e.groupby("symbol")
    e["yoy"] = (e.eps - g.eps.shift(4)).where((e.end - g.end.shift(4)).dt.days.between(350, 380))
    e["next_yoy"] = g.yoy.shift(-1).where((g.end.shift(-1) - e.end).dt.days.between(80, 100))
    x = t[["row", "symbol", "call"]].assign(seen=pd.to_datetime(t.call) - pd.Timedelta(days=5)).sort_values("seen")
    x = pd.merge_asof(x, e[["symbol", "end", "yoy", "next_yoy"]].sort_values("end"), left_on="seen", right_on="end", by="symbol", direction="backward", tolerance=pd.Timedelta(days=100))
    return x.set_index("row")[["yoy", "next_yoy"]].reindex(t.row)


def market():
    sc.CUT = END
    t = pd.concat([pd.read_parquet(sc.FEAT), pd.read_parquet(NEW_FEAT)], ignore_index=True)
    o, c, member, sector, ret, peers, vol, dollars, paid, so, s = sc.market(t.symbol.unique())
    print("sessions", c.index[0].date(), c.index[-1].date(), c.shape, flush=True)
    t["pos"] = c.index.searchsorted(pd.to_datetime(t.call) + pd.Timedelta(days=1))
    t = sc.place(t, c, sector, ret, peers, vol)
    pos, col = t.pos.values, t.col.values
    t["member"] = member.values[pos, col]
    t = t[t.member & (t.answer_words >= 300) & (t.question_words >= 100) & t.sector.notna()].sort_values(["date", "symbol"])
    t = t.drop_duplicates(["call", "prepared_words", "answer_words", "question_words"]).reset_index(drop=True)
    pos, col = t.pos.values, t.col.values
    t["dollars"] = dollars.values[pos, col]
    t["past"] = c.values[pos, col] / c.values[np.maximum(pos - 60, 0), col] - 1
    t["risk"] = c.pct_change(fill_method=None).rolling(60).std().values[pos, col]
    t["month"] = t.date.dt.to_period("M").astype(str)
    eps = next_eps(t)
    t["eps_yoy"] = eps.yoy.values
    t["eps_change"] = t.eps_yoy / paid.values[pos, col]
    t["next_yoy"] = eps.next_yoy.values
    t["exit"] = c.index[np.minimum(pos + sc.H, len(c.index) - 1)]
    outcome = ["row", "symbol", "date", "exit", "ret", "excess", "y", "next_yoy"]
    t[outcome].to_parquet(ALL_OUT)
    keep = ["row", "symbol", "call", "fy", "fq", "date", "month", "sector", "dollars", "past", "risk", "reaction", "eps_yoy", "eps_change",
            "answer_tone", "question_tone", "prepared_tone", "answer_words", "question_words", "dodges", "askers"]
    t[keep].to_parquet(CALLS)
    print(len(t), "calls", t.groupby(t.date.dt.year).size().to_dict(), flush=True)
    old = pd.read_parquet(sc.D / "small_call_events.parquet", columns=["row", "y", "reaction"]).merge(t[["row", "y", "reaction"]], on="row", suffixes=("_old", ""))
    print("match with checked events:", len(old), "rows, y corr", round(old.y_old.corr(old.y), 6), "max diff", round((old.y_old - old.y).abs().max(), 4), "reaction max diff", round((old.reaction_old - old.reaction).abs().max(), 4), flush=True)


def sets():
    t = pd.read_parquet(CALLS)
    rng = np.random.default_rng(20261106)
    t["set"], t["rank"] = "", -1
    read = pd.read_csv(sc.D / "small_call_read.csv", usecols=["row"]).row
    dev = t[t.row.isin(read)].sample(frac=1, random_state=1)
    t.loc[dev.index[:CHECK_N], "set"] = "student_check"
    t.loc[dev.index[CHECK_N:], "set"] = "dev"
    pool = t[(t.date >= TRAIN[0]) & (t.date <= TRAIN[1]) & (t.set == "")]
    even = pool.date.dt.quarter % 2 == 0
    sft = pool[even].sample(frac=1, random_state=2)
    rl = pool[~even].sample(frac=1, random_state=3)
    t.loc[sft.index[:SFT_N], "set"] = "sft"
    t.loc[sft.index[SFT_N:SFT_N + SFT_VAL], "set"] = "sft_val"
    t.loc[rl.index[:RL_N], "set"] = "rl"
    val = t[(t.date >= VAL[0]) & (t.date <= VAL[1])].sample(frac=1, random_state=4)
    t.loc[val.index[:VAL_N], "set"] = "validation"
    test = t[(t.date >= TEST[0]) & (t.date <= TEST[1])]
    share = test.groupby("month").size() / len(test)
    d_n, w_n = (share * TEST_D).round().astype(int), (share * TEST_W).round().astype(int)
    for m, g in test.groupby("month"):
        i = rng.permutation(g.index.values)
        t.loc[i[:d_n[m]], "set"] = "test_d"
        t.loc[i[d_n[m]:d_n[m] + w_n[m]], "set"] = "test_w"
        t.loc[i[d_n[m] + w_n[m]:], "set"] = "test_f"
    for s, g in t[t.set != ""].groupby("set"):
        t.loc[g.sample(frac=1, random_state=5).index, "rank"] = np.arange(len(g))  # nested sizes: take rank < n
    t = t[t.set != ""].sort_values(["set", "rank"]).reset_index(drop=True)
    t.to_parquet(DATA / "sets.parquet")
    every = pd.read_parquet(ALL_OUT).merge(t[["row", "set"]], on="row", how="left")
    every[~every.set.fillna("").str.startswith("test")].drop(columns="set").to_parquet(DATA / "kb_outcomes.parquet")  # past outcomes the knowledge base may show
    o = every.dropna(subset=["set"])
    SEALED.mkdir(exist_ok=True)
    for part in ("test_d", "test_w", "test_f"):
        o[o.set == part].drop(columns="set").to_parquet(SEALED / f"{part}.parquet")
    o[~o.set.str.startswith("test")].drop(columns="set").to_parquet(OUTCOMES)
    ALL_OUT.unlink()
    freeze = {"sets": sha(DATA / "sets.parquet"), "outcomes": sha(OUTCOMES), **{p: sha(SEALED / f"{p}.parquet") for p in ("test_d", "test_w", "test_f")},
              "counts": t.groupby("set").size().to_dict(), "test_rows": {p: t[t.set == p].row.tolist() for p in ("test_d", "test_w", "test_f")}}
    write_atomic(FREEZE, json.dumps(freeze))
    print(freeze["counts"], flush=True)
    print(t.groupby(["set", t.date.dt.year]).size().to_string(), flush=True)


def refresh_eps():
    """Fetches the free EPS table again and refills next-quarter EPS in the outcome files, so calls whose next
    quarter was reported after the build get their label. Calls and every other column stay as frozen."""
    import os
    import subprocess
    f = sc.SRC / "stock_tailing_eps.parquet"
    if os.environ.get("PAPER_PINNED_EPS") != "1":  # set it to 1 to keep the EPS table already on disk (the published 2026-11-12 copy)
        tmp = f.with_suffix(".part")
        subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(tmp), "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/US/stock_tailing_eps.parquet"], check=True)
        pq.ParquetFile(tmp)  # refuses a broken download before it replaces the old table
        tmp.replace(f)
    calls = pd.read_parquet(CALLS)
    eps = next_eps(calls).next_yoy
    eps.index = calls.row.values
    for path in [OUTCOMES, *(SEALED / f"{p}.parquet" for p in ("test_d", "test_w", "test_f"))]:
        d = pd.read_parquet(path)
        before = int(d.next_yoy.notna().sum())
        d["next_yoy"] = d.next_yoy.fillna(d.row.map(eps))
        d.to_parquet(path)
        print(path.name, "next EPS known", before, "->", int(d.next_yoy.notna().sum()), "of", len(d), flush=True)


if __name__ == "__main__":
    {"parse": parse, "market": market, "sets": sets, "refresh_eps": refresh_eps}[sys.argv[1]]()
