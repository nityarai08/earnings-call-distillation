"""Claude reads calls: the same prompt and knowledge block the student gets, no web tools, no outcomes.

  PYTHONPATH=src:scripts .venv/bin/python -W ignore scripts/paper/teacher.py handbook
  ... teacher.py probe  --model claude-opus-5-5 --set dev --n 5
  ... teacher.py submit --stage week --step opus_labels --model claude-opus-5-5 --set sft --n 800 --lenses both
  ... teacher.py fetch  --stage week --step opus_labels

submit saves the batch id before anything else and records the estimated cost in the ledger first, and the
ledger refuses anything over the cap. Rows already labelled are never sent again. fetch can run any number of
times: results are keyed by row and lens, so nothing is paid twice.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
import blocks  # noqa: E402
import kb  # noqa: E402
import prompt as P  # noqa: E402
from common import DATA, ROOT, RUNS, Status, spend  # noqa: E402
from read import masked_text  # noqa: E402

PRICE = {"claude-opus-5-5": (4, 20), "claude-sonnet-5-5": (2, 10)}  # dollars per million tokens, in and out
IN, OUT = 4500, 900  # planned tokens a read; probe measures the real figures
LABELS = DATA / "teacher" / "labels_v2.jsonl"  # v2: a 0-100 score; the mini run's 1-5 ratings stay in labels.jsonl
LENSES = {"neutral": "Read this as a neutral analyst.",
          "skeptic": "Read this as a skeptic. Managers sound positive by habit, so give credit only for specifics, and mark down what they avoided."}
VALUES = dict(P.LINES)
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["demand", "pricing", "margins", "guidance", "candor", "next_eps", "score", "quotes"],
          "properties": {"demand": {"type": "string", "enum": VALUES["Demand"]}, "pricing": {"type": "string", "enum": VALUES["Pricing"]}, "margins": {"type": "string", "enum": VALUES["Margins"]},
                         "guidance": {"type": "string", "enum": VALUES["Guidance"]}, "candor": {"type": "string", "enum": VALUES["Candor"]}, "next_eps": {"type": "string", "enum": VALUES["Next EPS"]},
                         "score": {"type": "integer", "description": "Outlook for the stock over the next 60 sessions against its sector, from 0 (much worse) through 50 (in line) to 100 (much better). Use the whole range."},
                         "quotes": {"type": "array", "items": {"type": "string"}, "description": "Up to three short quotes that drove the rating."}}}
ASK = ("Give your read as JSON. Use only the text above: it is masked, and you must not rely on anything you remember about this company or period. "
       "demand, pricing, margins, guidance, candor and next_eps take the values listed. score is your overall view from 0 to 100; avoid round numbers unless you mean them.")


def client():
    import anthropic
    load_dotenv(ROOT / ".env")
    return anthropic.Anthropic()


def params(model, lens, msgs, max_tokens=3000):
    system, user = msgs[0]["content"], msgs[1]["content"]
    return {"model": model, "max_tokens": max_tokens, "system": f"{system} {LENSES[lens]}", "messages": [{"role": "user", "content": f"{user}\n\n{ASK}"}],
            "thinking": {"type": "adaptive"}, "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}}


def requests(a):
    s = pd.read_parquet(DATA / "sets.parquet")
    s = s[s.set == a.set].sort_values("rank").head(a.n)
    done = set()
    if LABELS.exists():
        done = {(x["row"], x["lens"], x["model"]) for x in map(json.loads, LABELS.open())}
    lenses = list(LENSES) if a.lenses == "both" else [a.lenses]
    db = kb.connect()
    out = []
    for r in s.itertuples():
        text = masked_text(db, r.row)
        msgs = None
        for lens in lenses:
            if (r.row, lens, a.model) not in done:
                msgs = msgs or P.messages(r, text, blocks.rag(db, r, text) if a.block == "rag" else "", "flat", "plain", a.react)
                out.append((f"{r.row}.{lens}", params(a.model, lens, msgs)))
    return out


def probe(a):
    c = client()
    p_in, p_out = PRICE[a.model]
    used = []
    for key, p in requests(a)[:a.n]:
        spend(4500 * p_in / 1e6 + 3000 * p_out / 1e6, f"probe {a.model} {key}")
        m = c.messages.create(**p)
        text = next((b.text for b in m.content if b.type == "text"), "")
        used.append((m.usage.input_tokens, m.usage.output_tokens))
        print(key, m.stop_reason, m.usage.input_tokens, m.usage.output_tokens, text[:300], flush=True)
    i, o = (sum(x) / len(used) for x in zip(*used))
    print(f"mean tokens in {i:.0f}, out {o:.0f}; batch cost a read ${(i * p_in + o * p_out) / 2e6:.4f}", flush=True)


def submit(a):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    todo = requests(a)
    status = Status(a.stage, a.step, len(todo))
    if not todo:
        print("nothing to send", flush=True)
        return
    p_in, p_out = PRICE[a.model]
    cost = len(todo) * (a.tokens_in * p_in + a.tokens_out * p_out) / 2e6
    spend(cost, f"{a.stage} {a.step} {len(todo)} reads {a.model}")
    status.spent(cost)
    c = client()
    batch = c.messages.batches.create(requests=[Request(custom_id=k.replace(".", "-"), params=MessageCreateParamsNonStreaming(**p)) for k, p in todo])
    f = RUNS / a.stage / a.step / "batches.jsonl"
    with f.open("a") as w:
        w.write(json.dumps({"id": batch.id, "model": a.model, "n": len(todo)}) + "\n")
    print(batch.id, len(todo), f"${cost:.2f}", flush=True)


def fetch(a):
    c = client()
    f = RUNS / a.stage / a.step / "batches.jsonl"
    status = Status(a.stage, a.step)
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    have = {(x["row"], x["lens"], x["model"]) for x in map(json.loads, LABELS.open())} if LABELS.exists() else set()
    open_batches = 0
    for b in map(json.loads, f.open()):
        info = c.messages.batches.retrieve(b["id"])
        if info.processing_status != "ended":
            open_batches += 1
            print(b["id"], info.processing_status, info.request_counts, flush=True)
            continue
        with LABELS.open("a") as w:
            for res in c.messages.batches.results(b["id"]):
                row, lens = res.custom_id.split("-")
                if (int(row), lens, b["model"]) in have:
                    continue
                if res.result.type != "succeeded":
                    print("failed", res.custom_id, res.result.type, flush=True)
                    continue
                m = res.result.message
                text = next((x.text for x in m.content if x.type == "text"), "")
                if m.stop_reason != "end_turn" or not text:
                    print("no answer", res.custom_id, m.stop_reason, flush=True)
                    continue
                w.write(json.dumps({"row": int(row), "lens": lens, "model": b["model"], "tokens_in": m.usage.input_tokens, "tokens_out": m.usage.output_tokens, **json.loads(text)}) + "\n")
    status.rows(sum(1 for _ in LABELS.open()) if LABELS.exists() else 0)
    if not open_batches:
        status.done()
    print("open batches", open_batches, flush=True)
    return open_batches


def wait(a):
    while fetch(a):
        time.sleep(120)


HANDBOOK = {"type": "object", "additionalProperties": False, "required": ["notes"], "properties": {"notes": {"type": "array", "items": {"type": "string"}}}}


def handbook(a):
    spend(1.5, "handbook")
    m = client().messages.create(
        model="claude-opus-5-5", max_tokens=12000, thinking={"type": "adaptive"}, output_config={"effort": "medium", "format": {"type": "json_schema", "schema": HANDBOOK}},
        messages=[{"role": "user", "content": "Write 50 short notes for an analyst reading earnings-call answers from small and mid-sized US public companies. Each note is 40 to 80 words on one general idea: what a guidance cut usually means, how to read an inventory build, signs of an evasive answer, what backlog or pricing language tells you, working capital, customer concentration, one-off items, and similar. General ideas only: name no company, person, year, date or market event, and give no specific figures from any real period."}])
    notes = json.loads(next(b.text for b in m.content if b.type == "text"))["notes"]
    print(len(notes), m.usage.input_tokens, m.usage.output_tokens, flush=True)
    (DATA / "handbook.json").write_text(json.dumps(notes, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["handbook", "probe", "submit", "fetch", "wait"])
    p.add_argument("--stage")
    p.add_argument("--step")
    p.add_argument("--model", default="claude-opus-5-5", choices=list(PRICE))
    p.add_argument("--set")
    p.add_argument("--n", type=int, default=10 ** 9)
    p.add_argument("--lenses", default="both", choices=["both", *LENSES])
    p.add_argument("--block", default="rag", choices=["rag", "none"])
    p.add_argument("--react", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tokens-in", type=int, default=IN)
    p.add_argument("--tokens-out", type=int, default=OUT)
    a = p.parse_args()
    {"handbook": handbook, "probe": probe, "submit": submit, "fetch": fetch, "wait": wait}[a.mode](a)


if __name__ == "__main__":
    main()
