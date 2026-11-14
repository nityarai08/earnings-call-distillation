"""Every LLM read in the paper runs: all setups, controls and hidden states. Saves one row per call, and
skips rows already saved, so a stop loses one call. Holds the 9B lock one chunk at a time.

  PYTHONPATH=src:scripts .venv/bin/python -W ignore scripts/paper/read.py --stage mini --step raw --arm raw \
      --set dev --n 100 [--lines all] [--block rag] [--style plain] [--adapter adapters/x] [--position flat] [--no-text] [--no-react] [--hidden]
"""
import argparse
import fcntl
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import blocks  # noqa: E402
import kb  # noqa: E402
import prompt as P  # noqa: E402
from common import DATA, ROOT, RUNS, Status, done_rows, write_atomic  # noqa: E402

MODEL = "mlx-community/Qwen3.5-9B-4bit"
MAX_WORDS = 2500
CHUNK = 100
COLS = ["row", "position", "tokens", "seconds"] + [n.lower().replace(" ", "_") for n, _ in P.LINES[:5]] + ["p_eps_up", "p_pos", "p_mass"]


def masked_text(db, row):
    parts = db.execute("select text from passages where call_row = ? and kind = 'call' order by id", [row]).fetchall()
    return " ".join(" ".join(x[0] for x in parts).split()[:MAX_WORDS])


def lm_parts(model):
    lm = model.language_model
    return lm, lm.model


def head(lm, h):
    return lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)


def step(mx, model, ids, cache):
    """Runs tokens through the model and returns the last token's logits and hidden state."""
    lm, tm = lm_parts(model)
    h = tm(mx.array([ids]), cache)[:, -1:]
    return head(lm, h)[0, 0].astype(mx.float32), h[0, 0]


def walk(mx, model, opts, logits, cache, choose=None):
    """Picks each line's value. choose(name, values, probs) picks a value; the default takes the most likely.
    Returns the read and each line's probabilities over its allowed values."""
    read, probs, mass = {}, {}, 1.0
    for i, (name, ids, after) in enumerate(opts):
        vals = list(ids)
        p = np.array(mx.softmax(logits, axis=-1)[mx.array([ids[v][0] for v in vals])].tolist())
        total = float(p.sum())
        p = p / total
        probs[name] = dict(zip(vals, p))
        read[name] = choose(name, vals, p) if choose else vals[int(p.argmax())]
        mass = min(mass, total)
        if i + 1 < len(opts):
            logits, _ = step(mx, model, ids[read[name]] + after, cache)
    return read, probs, mass


def forced(spec):
    """--force Demand=down writes that line as given, so we can see whether the action follows the read."""
    if not spec:
        return None
    line, value = spec.split("=")
    return lambda name, vals, p: value if name == line else vals[int(p.argmax())]


def run(a):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    s = pd.read_parquet(DATA / "sets.parquet")
    s = s[s.set == a.set].sort_values("rank").head(a.n)
    out = RUNS / a.stage / "reads" / f"{a.arm}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = {int(x) for x in done_rows(out)}
    if not out.exists():
        write_atomic(out, ",".join(COLS) + "\n")
    status = Status(a.stage, a.step, None)
    make_block = blocks.KINDS.get(a.block) or blocks.saved(ROOT / a.block)
    todo = s[~s.row.isin(seen)]
    db = kb.connect()
    while len(todo):
        part, todo = todo.head(CHUNK), todo.iloc[CHUNK:]
        lock = (ROOT / "logs" / "llm.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        model, tok = load(MODEL, adapter_path=a.adapter)
        opts = P.options(tok, a.position, a.lines)
        hidden = []
        with out.open("a") as f:
            for r in part.itertuples():
                start = time.time()
                text = "" if a.no_text else masked_text(db, r.row)
                block = make_block(db, r, text or masked_text(db, r.row), tok)
                ids = P.rendered(tok, P.messages(r, text, block, a.position, a.style, not a.no_react), a.lines)
                cache = make_prompt_cache(model)
                logits, h = step(mx, model, ids, cache)
                if a.hidden:
                    hidden.append(np.array(h.astype(mx.float32).tolist(), dtype=np.float32))
                read, probs, mass = walk(mx, model, opts, logits, cache, forced(a.force))
                pos = P.ACTIONS[a.position][0]
                f.write(",".join(str(x) for x in [r.row, a.position, len(ids), round(time.time() - start, 2), *[read.get(n, "") for n, _ in P.LINES[:5]],
                                                  round(probs["Next EPS"]["up"], 5) if "Next EPS" in probs else "", round(probs["Action"][pos], 5), round(mass, 4)]) + "\n")
                f.flush()
                mx.clear_cache()
        if a.hidden:
            np.savez(out.with_name(f"{a.arm}_hidden_{part.row.iloc[0]}.npz"), row=part.row.values, h=np.stack(hidden))
        status.save()
        del model
        fcntl.flock(lock, fcntl.LOCK_UN)
    return len(s)


def main():
    p = argparse.ArgumentParser()
    for k in ("stage", "step", "arm", "set"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--n", type=int, default=10 ** 9)
    p.add_argument("--block", default="none", help="none, rag, random, shuffled, or a pi log path")
    p.add_argument("--style", default="plain", choices=list(P.STYLES))
    p.add_argument("--position", default="flat", choices=list(P.ACTIONS))
    p.add_argument("--lines", default="all", choices=list(P.KEEP))
    p.add_argument("--adapter")
    p.add_argument("--force", help="Line=value, for the faithfulness test")
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--no-react", action="store_true")
    p.add_argument("--hidden", action="store_true")
    a = p.parse_args()
    print(run(a), "rows", flush=True)


if __name__ == "__main__":
    main()
