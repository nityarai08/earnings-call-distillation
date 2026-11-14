"""Early results while a stage runs: every finished read scored on development calls, training curves, how
pi searched, and how Opus compares with the 9B. Never opens a sealed test part.

  PYTHONPATH=src:scripts .venv/bin/python -W ignore scripts/paper/interim.py mini

Development calls are from 2023-2024, years the model may remember, and the mini run reads 20 to 100 of
them, so these numbers show direction and plumbing, not findings. Intervals are a bootstrap over calls.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import prompt as P  # noqa: E402
from common import DATA, ROOT, RUNS, write_atomic  # noqa: E402
from score import ic, join, outcomes  # noqa: E402

OUT = ROOT / "reports" / "paper" / "tracker"
BOOT = 300
ARMS = {  # read file: (group, name shown)
    "memory": ("Setups", "Memory only"), "raw": ("Setups", "Raw"), "rag": ("Setups", "RAG"), "pi": ("Setups", "Pi search"),
    "pi_rag_docs": ("Controls", "Pi loop, RAG's documents"), "rag_random": ("Controls", "RAG, random documents"), "rag_shuffled": ("Controls", "RAG, outcomes shuffled"),
    "prompt_plain": ("Prompts", "Prompt: plain"), "prompt_habit": ("Prompts", "Prompt: habit"), "prompt_change": ("Prompts", "Prompt: change"),
    "sft_score": ("SFT arms", "SFT: score only"), "sft_read": ("SFT arms", "SFT: read"), "sft_read_eps": ("SFT arms", "SFT: read + EPS"),
    "sft_hard": ("SFT arms", "SFT: hard labels"), "sft_placebo": ("SFT arms", "SFT: placebo"), "sft_grown": ("SFT arms", "SFT: grown"),
    "rl_business": ("RL arms", "RL: business"), "rl_trade": ("RL arms", "RL: trade"), "rl_both": ("RL arms", "RL: both"), "rl_shuffled": ("RL arms", "RL: shuffled"),
    "base_words": ("Baselines", "Word list"), "base_eps": ("Baselines", "Known-EPS rule"), "base_numbers": ("Baselines", "Numbers only")}


def boot(d, score, outcome, rng):
    got = [ic(d.iloc[rng.integers(0, len(d), len(d))], score, outcome) for _ in range(BOOT)]
    got = [x for x in got if x is not None]
    return (float(np.quantile(got, .025)), float(np.quantile(got, .975))) if got else (None, None)


def reads_table(stage):
    o = outcomes()
    sets = pd.read_parquet(DATA / "sets.parquet", columns=["row", "set"])
    rng = np.random.default_rng(0)
    rows = []
    for arm, (group, name) in ARMS.items():
        f = RUNS / stage / "reads" / f"{arm}.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f)
        if arm.startswith("base_"):  # baselines are scored on the calls the raw read covered, once it exists
            raw = RUNS / stage / "reads" / "raw.csv"
            if not raw.exists():
                continue
            r = r[r.row.isin(pd.read_csv(raw).row)]
        if "p_eps_up" not in r:
            r["p_eps_up"] = np.nan
        d = join(r, o)
        if not len(d):
            continue
        x = d.dropna(subset=["next_yoy"])
        eps = x.p_eps_up.notna().any()
        row = {"key": f"{stage}:{arm}", "stage": stage, "group": group, "arm": name, "calls": len(d),
               "eps_acc": float(((x.p_eps_up > .5) == (x.next_yoy > 0)).mean()) if eps else None,
               "eps_ic": ic(d, "p_eps_up", "next_yoy") if eps else None,
               "stock_acc": float(((d.p_pos > .5) == (d.excess > 0)).mean()), "stock_ic": ic(d, "p_pos", "y"),
               "buy_share": float((d.p_pos > .5).mean()), "mean_p": float(d.p_pos.mean())}
        row["eps_lo"], row["eps_hi"] = boot(d, "p_eps_up", "next_yoy", rng) if eps else (None, None)
        row["stock_lo"], row["stock_hi"] = boot(d, "p_pos", "y", rng)
        rows.append(row)
    return rows


def training_table(stage):
    rows = []
    for f in sorted((RUNS / stage / "train").glob("*.csv")):
        t = pd.read_csv(f)
        if "loss_read" in t:
            for r in t.itertuples():
                rows.append({"key": f"{stage}:{f.stem}:{r.update}", "stage": stage, "kind": "sft", "run": f.stem, "update": int(r.update),
                             "loss": round(r.loss_read + r.loss_eps + r.loss_action, 4), "loss_action": round(r.loss_action, 4)})
        elif "reward" in t:  # RL: mean rollout reward, KL from the starting model, and positive-action share
            for r in t.itertuples():
                rows.append({"key": f"{stage}:{f.stem}:{r.update}", "stage": stage, "kind": "rl", "run": f.stem, "update": int(r.update),
                             "reward": round(r.reward, 4), "kl": round(r.kl, 4), "p_pos": round(r.p_pos, 4)})
    return rows


NAME = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b")


def agent_table(stage):
    rows = []
    for f in sorted((RUNS / stage / "agent").glob("*.jsonl")):
        x = pd.DataFrame(map(json.loads, f.open()))
        q = [s for qs in x.queries for s in qs]
        rows.append({"key": f"{stage}:{f.stem}", "stage": stage, "arm": {"pi": "Pi search", "pi_rag_docs": "Pi loop, RAG's documents"}.get(f.stem, f.stem),
                     "cases": len(x), "finished": float((x.code == 0).mean()), "seconds": float(x.seconds.mean()), "tool_calls": float(x.calls.clip(upper=4).mean()),
                     "docs_read": float(x.docs.map(len).mean()), "read_none": float((x.docs.map(len) == 0).mean()), "tokens_read": float(x.tokens.mean()),
                     "queries": len(q), "repeated": float(np.mean([len(qs) != len(set(qs)) for qs in x.queries if qs])) if q else None,
                     "named": float(np.mean([bool(NAME.search(s)) for s in q])) if q else None})
    return rows


def teacher_table(stage):
    from teacher import LABELS
    f = DATA / "teacher" / "labels.jsonl" if stage == "mini" else LABELS  # the mini run used the 1-5 labels
    r = RUNS / stage / "reads" / "student_rag.csv"
    if not f.exists() or not r.exists():
        return []
    lab = pd.DataFrame(map(json.loads, f.open()))
    nine = pd.read_csv(r)
    o = outcomes()
    names = [n.lower() for n, _ in P.LINES[:5]]
    opus = lab[(lab.model == "claude-opus-5-5") & (lab.lens == "neutral") & lab.row.isin(nine.row)].drop_duplicates("row")
    rows = []
    for model, label in (("claude-opus-5-5", "Opus 5.5"), ("claude-sonnet-5-5", "Sonnet 5.5")):
        t = lab[(lab.model == model) & (lab.lens == "neutral") & lab.row.isin(nine.row)].drop_duplicates("row")
        d = t.merge(o, on="row").assign(p_eps_up=lambda z: (z.next_eps == "up").astype(float), p_pos=lambda z: z["score" if "score" in z else "rating"].astype(float))
        d = join(d[["row", "p_eps_up", "p_pos"]], o)
        x = d.dropna(subset=["next_yoy"])
        agree = t.merge(opus, on="row", suffixes=("", "_o"))
        rows.append({"key": f"{stage}:{model}", "stage": stage, "reader": label, "calls": len(d), "eps_acc": float(((x.p_eps_up > .5) == (x.next_yoy > 0)).mean()),
                     "stock_ic": ic(d, "p_pos", "y"), "match_opus": float(np.mean([(agree[n] == agree[f"{n}_o"]).mean() for n in names])) if model != "claude-opus-5-5" else 1.0})
    d = join(nine, o)
    x = d.dropna(subset=["next_yoy"])
    agree = nine.merge(opus, on="row", suffixes=("", "_o"))
    rows.append({"key": f"{stage}:9b", "stage": stage, "reader": "Qwen 9B, untrained", "calls": len(d), "eps_acc": float(((x.p_eps_up > .5) == (x.next_yoy > 0)).mean()),
                 "stock_ic": ic(d, "p_pos", "y"), "match_opus": float(np.mean([(agree[n] == agree[f"{n}_o"]).mean() for n in names]))})
    return rows


def main():
    """Every stage that has run, so the page keeps earlier stages when a new one starts. The argument is kept
    for older commands and is not needed."""
    stages = [s for s in ("mini", "week", "full") if (RUNS / s).exists()]
    tables = {"interim": reads_table, "training": training_table, "agent": agent_table, "teacher": teacher_table}
    for name, make in tables.items():
        rows = []
        for stage in stages:
            got = make(stage)
            for r in got:  # the page names stages as the tracker does: Mini, Week, Full
                r["stage"] = stage.title()
            rows += got
        write_atomic(OUT / f"{name}.json", json.dumps(rows))
        print(name, len(rows), flush=True)


if __name__ == "__main__":
    main()
