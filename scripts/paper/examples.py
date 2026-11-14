"""Writes examples/: the exact inputs, targets and outputs for three real calls, so a reader can see what the
models saw and wrote without running anything.

  PYTHONPATH=src:scripts .venv/bin/python -W ignore scripts/paper/examples.py [--model]

1_test_call         the first Test-W call: the prompt each setup saw, every setup's answer, and what happened
2_sft_training_call the first SFT training call: Opus's two labels, the training target, the training text
3_rl_training_call  an RL training call: the prompt, the reward arithmetic and the target RL trains toward

--model also runs the SFT adapter on the RL call, to show the starting probabilities RL tilts (Apple Silicon).
Needs the knowledge base (data/paper/kb.sqlite), since prompts are built from it.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import blocks  # noqa: E402
import kb  # noqa: E402
import prompt as P  # noqa: E402
import train_soft as T  # noqa: E402
from common import DATA, ROOT, RUNS  # noqa: E402
from read import MODEL, masked_text  # noqa: E402

OUT = ROOT / "examples"
READS = RUNS / "week" / "reads"
LINE_COLS = [n.lower() for n, _ in P.LINES[:5]]


def chat_text(tok, msgs, lines="all"):
    """The exact text the model read: chat template, then the first answer label it continues from."""
    first = P.KEEP[lines][0] if P.KEEP[lines] else "Action"
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False) + f"{first}:"


def answer(read, p_eps_up, p_pos, position):
    """A saved read as the seven lines the model would write, most likely value on each line."""
    pos, neg = P.ACTIONS[position]
    lines = [f"{n}: {read[c]}" for (n, _), c in zip(P.LINES[:5], LINE_COLS)]
    if p_eps_up == p_eps_up:
        lines.append(f"Next EPS: {'up' if p_eps_up > .5 else 'down'}")
    lines.append(f"Action: {pos if p_pos > .5 else neg}")
    return "\n".join(lines)


def write(folder, name, text):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text if text.endswith("\n") else text + "\n")


def test_call(db, tok, s):
    """Test-W was sealed until the final step; this is its first call by the fixed random order."""
    folder = OUT / "1_test_call"
    r = next(s[s.set == "test_w"].sort_values("rank").head(1).itertuples())
    text = masked_text(db, r.row)
    pi = {x["row"]: x for x in map(json.loads, (RUNS / "week" / "agent" / "pi_test.jsonl").open())}[r.row]
    prompts = {"memory": P.messages(r, "", "", "flat"),  # numbers only, no transcript, no knowledge
               "raw": P.messages(r, text, "", "flat"),
               "rag": P.messages(r, text, blocks.rag(db, r, text, tok), "flat"),  # also what SFT and RL read
               "pi": P.messages(r, text, pi["block"], "flat")}
    for k, m in prompts.items():
        write(folder, f"prompt_{k}.txt", chat_text(tok, m))
    rows = []
    for setup, arm, prompt_file in (("Memory only", "memory", "memory"), ("Raw", "raw", "raw"), ("RAG", "rag", "rag"), ("Pi", "pi", "pi"),
                                    ("SFT", "sft_best", "rag"), ("RL", "rl_best", "rag")):
        d = pd.read_csv(READS / f"{arm}_test.csv").set_index("row").loc[r.row]
        rows.append({"setup": setup, "prompt": f"prompt_{prompt_file}.txt", **{c: d[c] for c in LINE_COLS},
                     "p_next_eps_up": d.p_eps_up, "p_buy": d.p_pos, "answer": answer(d, d.p_eps_up, d.p_pos, "flat").replace("\n", " | ")})
    pd.DataFrame(rows).to_csv(folder / "answers.csv", index=False)
    o = pd.read_parquet(DATA / "sealed" / "test_w.parquet").set_index("row").loc[r.row]
    vol = o.excess / (o.y * math.sqrt(60))
    write(folder, "outcome.json", json.dumps({
        "row": int(r.row), "call_date": str(r.call), "decision_session": str(o.date.date()), "exit_session": str(o.exit.date()),
        "return": round(float(o.ret), 4), "sector_average_return": round(float(o.ret - o.excess), 4), "excess_return": round(float(o.excess), 4),
        "daily_volatility": round(float(vol), 4), "y = excess / (daily_volatility * sqrt(60))": round(float(o.y), 4),
        "next_quarter_eps_minus_year_earlier": round(float(o.next_yoy), 4), "next_eps_direction": "up" if o.next_yoy > 0 else "down"}, indent=1))
    return r


def sft_call(db, tok, s):
    """The first call of the SFT training set, with the target the read_eps arm learned from."""
    folder = OUT / "2_sft_training_call"
    calls = s[s.set == "sft"].sort_values("rank").iloc[0:400]  # the read_eps arm's 400 calls
    t = T.targets("read_eps", calls)
    x = t.sort_values("rank").iloc[0]
    r = next(calls[calls.row == x.row].itertuples())
    labels = [l for l in map(json.loads, (DATA / "teacher" / "labels_v2.jsonl").open()) if l["row"] == x.row]
    write(folder, "opus_labels.jsonl", "\n".join(json.dumps(l) for l in labels))
    month = t[t.month == x.month]
    target = {"row": int(x.row), "position": x.position,
              "read_lines (from Opus's first label)": {n: x[c] for (n, _), c in zip(P.LINES[:5], LINE_COLS)},
              "next_eps (the real outcome, not Opus)": x.eps,
              "opus_scores": [l["score"] for l in labels], "mean_opus_score": float(np.mean([l["score"] for l in labels])),
              "calls_in_same_month": int(len(month)),
              "action_target p(BUY) = percentile of mean score within the month - 0.5 / calls": round(float(x.p_pos), 4)}
    write(folder, "target.json", json.dumps(target, indent=1))
    read = {n: x[c] for (n, _), c in zip(P.LINES[:5], LINE_COLS)}
    pos, neg = P.ACTIONS[x.position]
    full = (chat_text(tok, P.messages(r, masked_text(db, r.row), blocks.rag(db, r, masked_text(db, r.row), tok), x.position))
            + "".join(f" {read[n]}\n{m}:" for (n, _), (m, _) in zip(P.LINES[:5], P.LINES[1:])) + f" {x.eps}\nAction: {pos}")
    write(folder, "training_text.txt", full)
    return x


def rl_call(db, tok, s, model):
    """The first trade-reward training call in a flat position with a clear outcome, to show the arithmetic."""
    folder = OUT / "3_rl_training_call"
    rl = s[s.set == "rl"].sort_values("rank").head(400).merge(pd.read_parquet(DATA / "outcomes.parquet", columns=["row", "y", "next_yoy"]), on="row", how="left")
    rl["eps"] = rl.next_yoy.map(lambda v: None if pd.isna(v) else "up" if v > 0 else "down")
    rl["position"] = np.where(rl.row % 2 == 0, "flat", "long")
    w = T.rewards(rl, argparse.Namespace(arm="trade", control=False))
    x = w[(w.position == "flat") & (w.z.abs() > .5)].sort_values("rank").iloc[0]
    r = next(rl[rl.row == x.row].itertuples())
    text = masked_text(db, r.row)
    msgs = P.messages(r, text, blocks.rag(db, r, text, tok), "flat")
    write(folder, "prompt.txt", chat_text(tok, msgs))
    unit = x.risk * math.sqrt(60)
    out = {"row": int(x.row), "position": "flat (BUY or AVOID)", "rank_in_rl_set": int(x["rank"]),
           "y (clean 60-session outcome)": round(float(x.y), 4),
           "month_average_y (the 400 RL calls)": round(float(rl[rl.month == x.month].y.mean()), 4),
           "z = y - month average, clipped to [-3, 3]": round(float(x.z), 4),
           "unit = daily risk * sqrt(60)": round(float(unit), 4),
           "reward if BUY: z - 0.008 / unit": round(float(x.r_pos), 4),
           "reward if AVOID": 0.0}
    if model:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        from read import step, walk
        m, mtok = load(MODEL, adapter_path=str(ROOT / "adapters" / "paper" / "week" / "sft_grown"))
        cache = make_prompt_cache(m)
        logits, _ = step(mx, m, P.rendered(mtok, msgs), cache)
        read, probs, _ = walk(mx, m, P.options(mtok, "flat"), logits, cache)
        p = float(probs["Action"]["BUY"])
        tilted = p * math.exp(x.r_pos) / (p * math.exp(x.r_pos) + (1 - p))
        out.update({"starting model (SFT) read": read, "starting p(BUY)": round(p, 4),
                    "expected trade reward = p(BUY) * reward_if_BUY + p(AVOID) * 0": round(p * float(x.r_pos), 4),
                    "RL target p(BUY) = p * e^reward / (p * e^reward + (1 - p) * e^0)": round(tilted, 4)})
    write(folder, "reward.json", json.dumps(out, indent=1, default=str))
    return x


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", action="store_true")
    a = a.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    db = kb.connect()
    s = pd.read_parquet(DATA / "sets.parquet")
    print("test call", test_call(db, tok, s).row)
    print("sft call", sft_call(db, tok, s).row)
    print("rl call", rl_call(db, tok, s, a.model).row)


if __name__ == "__main__":
    main()
