"""Scores every read the same way. The only script that opens sealed test outcomes, and only by name.

  PYTHONPATH=src:scripts .venv/bin/python -W ignore scripts/paper/score.py selftest
  ... score.py baselines --stage mini --set dev --n 100
  ... score.py pick --stage mini --group sft --arms score,read,read_eps,hard --set dev
  ... score.py memo --stage mini --arm memo
  ... score.py final --stage mini --open test_d

IC is the correlation of ranks within each month, so market swings between months do not count. A step
between two setups is their paired IC difference, with a 95% interval from a bootstrap of whole weeks.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import prompt as P  # noqa: E402
from common import DATA, FREEZE, RUNS, SEALED, sha, write_atomic  # noqa: E402

ROUND_TRIP = .008
SETUPS = {"No call": "memory", "Call only": "raw", "RAG": "rag", "Pi": "pi", "SFT": "sft_best", "RL": "rl_best"}
BASES = {"Buy every call": "base_all", "Word list": "base_words", "Known-EPS rule": "base_eps", "Numbers only": "base_numbers"}
BOOT = 2000


def outcomes(part=None):
    """Development outcomes, or one sealed test part. A sealed part must match its frozen hash."""
    if part is None:
        return pd.read_parquet(DATA / "outcomes.parquet")
    f = SEALED / f"{part}.parquet"
    frozen = json.loads(FREEZE.read_text())
    d = pd.read_parquet(f)
    assert sorted(d.row) == sorted(frozen["test_rows"][part]), f"{part} calls changed since they were frozen"
    return d


def join(reads, o):
    s = pd.read_parquet(DATA / "sets.parquet", columns=["row", "month", "set"])
    d = reads.merge(o, on="row").merge(s, on="row")
    d["week"] = pd.to_datetime(d.date).dt.to_period("W").astype(str)
    return d


def month_rank(d, col):
    return d.groupby("month")[col].rank(pct=True)


def ic(d, score, outcome):
    x = d.dropna(subset=[score, outcome])
    return float(month_rank(x, score).corr(month_rank(x, outcome))) if len(x) > 2 and x[score].nunique() > 1 else None


def metrics(d):
    m = {"calls": len(d)}
    if d.p_eps_up.notna().any():
        x = d.dropna(subset=["p_eps_up", "next_yoy"])
        m["eps_acc"] = float(((x.p_eps_up > .5) == (x.next_yoy > 0)).mean())
        m["eps_ic"] = ic(d, "p_eps_up", "next_yoy")
    x = d.dropna(subset=["p_pos", "excess"])
    m["stock_acc"] = float(((x.p_pos > .5) == (x.excess > 0)).mean())
    m["stock_ic"] = ic(d, "p_pos", "y")
    buys = x[x.p_pos > .5]
    m["book_net"] = float(buys.excess.mean() - ROUND_TRIP) if len(buys) else None
    return m


def step_test(a, b):
    """Paired IC difference of b over a on shared calls, with a week bootstrap."""
    d = a.merge(b[["row", "p_pos"]], on="row", suffixes=("_a", "_b")).dropna(subset=["y"])
    if len(d) < 10:
        return None
    rng = np.random.default_rng(0)
    weeks = {w: g for w, g in d.groupby("week")}
    names = list(weeks)
    base = ic(d, "p_pos_b", "y") - ic(d, "p_pos_a", "y")
    got = []
    for _ in range(BOOT):
        x = pd.concat([weeks[w] for w in rng.choice(names, len(names))])
        i, j = ic(x, "p_pos_b", "y"), ic(x, "p_pos_a", "y")
        if i is not None and j is not None:
            got.append(i - j)
    lo, hi = np.quantile(got, [.025, .975])
    return base, lo, hi


def reads(stage, arm):
    f = RUNS / stage / "reads" / f"{arm}.csv"
    return pd.read_csv(f) if f.exists() else None


def selftest(a):
    o = outcomes()
    rng = np.random.default_rng(0)
    r = o[["row"]].assign(p_pos=rng.random(len(o)), p_eps_up=rng.random(len(o)))
    m = metrics(join(r, o))
    print("random scores:", {k: round(v, 4) for k, v in m.items()})
    assert abs(m["stock_ic"]) < .05 and abs(m["eps_acc"] - .5) < .05
    print("passed", flush=True)


def baselines(a):
    """Reads with no LLM: same columns, so they score the same way."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    s = pd.read_parquet(DATA / "sets.parquet")
    t = s[s.set == a.set].sort_values("rank").head(a.n) if a.set else s[s.set.str.startswith(("dev", "validation", "test"))]
    o = outcomes()
    fit = s[s.set.isin(["sft", "rl"])].merge(o[["row", "y"]], on="row").dropna(subset=["y"])
    cols = ["dollars", "past", "risk", "reaction", "eps_change"]
    model = HistGradientBoostingRegressor(max_depth=3, max_iter=100, random_state=0).fit(fit[cols], fit.y)
    rank = lambda x: x.rank(pct=True)
    out = {"base_all": pd.DataFrame({"row": t.row, "p_pos": .9, "p_eps_up": np.nan}),
           "base_words": pd.DataFrame({"row": t.row, "p_pos": rank(t.answer_tone), "p_eps_up": rank(t.answer_tone)}),
           "base_eps": pd.DataFrame({"row": t.row, "p_pos": rank(t.eps_change), "p_eps_up": (t.eps_yoy > 0).astype(float)}),
           "base_numbers": pd.DataFrame({"row": t.row, "p_pos": rank(pd.Series(model.predict(t[cols]), index=t.index)), "p_eps_up": np.nan})}
    folder = RUNS / a.stage / "reads"
    folder.mkdir(parents=True, exist_ok=True)
    for name, x in out.items():
        x.to_csv(folder / f"{name}{a.suffix}.csv", index=False)


def pick(a):
    """The best arm on development calls, by IC with rows from outcomes(); ties go to the first listed."""
    o = outcomes()
    best, table = None, []
    for arm in a.arms.split(","):
        r = reads(a.stage, arm)
        if r is None:
            continue
        d = join(r, o)
        d = d[d.set == a.set] if a.set else d
        m = metrics(d)
        table.append({"arm": arm, **m})
        v = (m.get("eps_ic") or 0) + (m.get("stock_ic") or 0)
        if best is None or v > best[1]:
            best = (arm, v)
    t = pd.DataFrame(table)
    print(t.round(4).to_string(), flush=True)
    t.to_csv(RUNS / a.stage / f"pick_{a.group}.csv", index=False)
    write_atomic(RUNS / a.stage / f"best_{a.group}.txt", best[0])
    print("best", best[0], flush=True)


def memo(a):
    """Did a small training run memorize its targets? Compares a read of the trained calls with them."""
    from train_soft import targets
    s = pd.read_parquet(DATA / "sets.parquet")
    r = reads(a.stage, a.arm)
    t = targets("read_eps", s[s.row.isin(r.row)])
    d = r.merge(t, on="row", suffixes=("", "_t"))
    names = [n.lower() for n, _ in P.LINES[:5]]
    exact = np.mean([(d[n] == d[f"{n}_t"]).mean() for n in names])
    rank = d.p_pos.corr(d.p_pos_t, method="spearman")
    print(f"read lines exact {exact:.1%}; action rank agreement {rank:.3f}", flush=True)
    write_atomic(RUNS / a.stage / f"memo_{a.arm}.json", json.dumps({"exact": exact, "rank": rank}))


def faith(a):
    """How far the action moves when one read line is flipped, before and after RL."""
    rows = []
    for tag in a.arms.split(","):
        up, down = reads(a.stage, f"{tag}_demand_up"), reads(a.stage, f"{tag}_demand_down")
        d = up.merge(down, on="row", suffixes=("_u", "_d"))
        rows.append({"model": tag, "move": float((d.p_pos_u - d.p_pos_d).mean()), "share_right_way": float((d.p_pos_u > d.p_pos_d).mean()), "calls": len(d)})
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(), flush=True)
    t.to_csv(RUNS / a.stage / "faithfulness.csv", index=False)


def read_match(stage, arm):
    """Share of the five read lines that match Opus, on the student-check calls."""
    r = reads(stage, arm)
    from teacher import LABELS as f
    if r is None or not f.exists():
        return None
    t = pd.DataFrame(map(json.loads, f.open()))
    t = t[(t.model == "claude-opus-5-5") & (t.lens == "neutral")].drop_duplicates("row")
    d = r.merge(t, on="row", suffixes=("", "_t"))
    names = [n.lower() for n, _ in P.LINES[:5]]
    return float(np.mean([(d[n] == d[f"{n}_t"]).mean() for n in names])) if len(d) else None


def final(a):
    o = outcomes(a.open)
    table, loaded = [], {}
    for name, arm in {**SETUPS, **BASES}.items():
        r = reads(a.stage, f"{arm}_test")
        if r is None:
            table.append({"setup": name})
            continue
        if "p_eps_up" not in r:
            r["p_eps_up"] = np.nan
        d = join(r, o)
        loaded[name] = d
        table.append({"setup": name, **metrics(d), "read_match": read_match(a.stage, f"{arm}_check")})
    cols = ["setup", "calls", "read_match", "eps_acc", "eps_ic", "stock_acc", "stock_ic", "book_net"]
    res = pd.DataFrame(table).reindex(columns=cols)
    res.round(4).to_csv(RUNS / a.stage / "results.csv", index=False)
    print(res.round(4).to_string(), flush=True)
    rows = []
    for key, x, y in (("rag_pi", "RAG", "Pi"), ("distill", "RAG", "SFT"), ("kb", "Call only", "RAG"), ("reward", "SFT", "RL")):
        if x in loaded and y in loaded:
            t = step_test(loaded[x], loaded[y])
            if t:
                rows.append({"question": key, "result": f"{y} minus {x}: IC {t[0]:+.3f} (95% {t[1]:+.3f} to {t[2]:+.3f})", "verdict": "Clear" if t[1] > 0 or t[2] < 0 else "Not distinguishable at this size"})
    pd.DataFrame(rows, columns=["question", "result", "verdict"]).to_csv(RUNS / a.stage / "findings.csv", index=False)
    print(rows, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["selftest", "baselines", "pick", "memo", "faith", "final"])
    p.add_argument("--stage")
    p.add_argument("--set")
    p.add_argument("--n", type=int, default=10 ** 9)
    p.add_argument("--group")
    p.add_argument("--suffix", default="")
    p.add_argument("--arm")
    p.add_argument("--arms")
    p.add_argument("--open", choices=["test_d", "test_w", "test_f"])
    a = p.parse_args()
    {"selftest": selftest, "baselines": baselines, "pick": pick, "memo": memo, "faith": faith, "final": final}[a.mode](a)


if __name__ == "__main__":
    main()
