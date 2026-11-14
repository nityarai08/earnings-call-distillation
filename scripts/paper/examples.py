"""Writes examples/: real inputs and outputs for every experiment and every test of the paper runs, and
data/paper/training/: the full SFT targets and RL reward inputs of every training call.

  PYTHONPATH=src:scripts .venv/bin/python -W ignore scripts/paper/examples.py [--model]

Each experiment folder holds its first 10 calls, in the fixed order of the call set it used (test folders: the
first 10 test calls of March 2026, so that one month can be ranked). Prompts are written once to
examples/prompts/ and the experiment files point to them. examples/README.md lists every experiment.

--model runs the model again (Apple Silicon): the start probabilities that RL changes, and for the first call of each
experiment a sample_output.md with what the model wrote and how the script turned it into numbers. The prompts need the knowledge base (data/paper/kb.sqlite).
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
PROMPTS = OUT / "prompts"
N = 10
GH = "https://github.com/nityarai08/earnings-call-distillation/blob/main/"
LINE_COLS = [n.lower() for n, _ in P.LINES[:5]]

# step, experiment, read file, call set, calls used, what it tests, prompt options
READS = [
    ("raw", "prompt_plain", "prompt_plain", "dev", 100, "Prompt style: the plain instruction", {}),
    ("raw", "prompt_habit", "prompt_habit", "dev", 100, "Prompt style: an instruction to weigh specifics over positive tone", {"style": "habit"}),
    ("raw", "prompt_change", "prompt_change", "dev", 100, "Prompt style: an instruction to compare with the last quarter", {"style": "change"}),
    ("raw", "raw", "raw", "validation", 500, "Raw setup: the call text, no knowledge block", {}),
    ("raw", "memory", "memory", "validation", 500, "Memory only: the facts line, no call text", {"text": False}),
    ("raw", "raw_flat", "raw_flat", "dev", 200, "Position check: the model holds no stock and chooses BUY or AVOID", {}),
    ("raw", "raw_long", "raw_long", "dev", 200, "Position check: the model holds the stock and chooses HOLD or SELL", {"position": "long"}),
    ("raw", "probe", "probe", "sft", 800, "Probe: the raw model's answers on SFT calls. Its hidden states are in reads/probe_hidden_*.npz", {}),
    ("rag", "rag", "rag", "validation", 500, "RAG setup: a fixed search adds a knowledge block", {"block": "rag"}),
    ("rag", "rag_random", "rag_random", "validation", 200, "Control: random passages of the same length instead of the search results", {"block": "random"}),
    ("rag", "rag_shuffled", "rag_shuffled", "validation", 200, "Control: the search results, with the past outcomes in them shuffled", {"block": "shuffled"}),
    ("opus_check", "student_rag", "student_rag", "student_check", 300, "The small model (RAG) on the check calls that Opus and Sonnet also read", {"block": "rag", "react": False}),
    ("sft", "memo", "memo", "sft", 50, "Memorization check: 50 training calls, trained 10 times, then read back", {"block": "rag"}),
    ("sft", "sft_read_eps", "sft_read_eps", "validation", 300, "SFT variant read + EPS (the variant kept)", {"block": "rag"}),
    ("sft", "sft_score", "sft_score", "validation", 300, "SFT variant score only: the model writes only the Action line", {"block": "rag", "lines": "action"}),
    ("sft", "sft_read", "sft_read", "validation", 300, "SFT variant read: read lines and Action, no Next EPS", {"block": "rag", "lines": "read"}),
    ("sft", "sft_hard", "sft_hard", "validation", 300, "SFT variant hard: Action trained on the real outcome", {"block": "rag", "lines": "action"}),
    ("sft", "sft_placebo", "sft_placebo", "validation", 300, "Control: SFT on Opus answers shuffled among calls of the same month", {"block": "rag"}),
    ("sft", "sft_grown", "sft_grown", "validation", 500, "SFT model after 800 calls (the SFT setup)", {"block": "rag"}),
    ("rl", "rl_business", "rl_business", "validation", 300, "RL with the business reward (EPS forecast)", {"block": "rag"}),
    ("rl", "rl_both", "rl_both", "validation", 300, "RL with the business and trade rewards together", {"block": "rag"}),
    ("rl", "rl_trade", "rl_trade", "validation", 300, "RL with the trade reward (the RL setup)", {"block": "rag"}),
    ("rl", "rl_shuffled", "rl_shuffled", "validation", 300, "Control: RL with outcomes shuffled within each month", {"block": "rag"}),
    ("rl", "sft_demand_up", "sft_demand_up", "validation", 200, "Forced-Demand test: SFT model with Demand forced to up", {"block": "rag", "force": "up"}),
    ("rl", "sft_demand_down", "sft_demand_down", "validation", 200, "Forced-Demand test: SFT model with Demand forced to down", {"block": "rag", "force": "down"}),
    ("rl", "rl_demand_up", "rl_demand_up", "validation", 200, "Forced-Demand test: RL model with Demand forced to up", {"block": "rag", "force": "up"}),
    ("rl", "rl_demand_down", "rl_demand_down", "validation", 200, "Forced-Demand test: RL model with Demand forced to down", {"block": "rag", "force": "down"}),
    ("pi", "pi", "pi", "validation", 500, "Pi setup: the model searches the library itself", {"block": "pi"}),
    ("pi", "pi_rag_docs", "pi_rag_docs", "validation", 200, "Control: the Pi tool loop, given RAG's documents", {"block": "pi_rag_docs"}),
    ("check", "raw_check", "raw_check", "student_check", 300, "Raw setup on the check calls", {"react": False}),
    ("check", "rag_check", "rag_check", "student_check", 300, "RAG setup on the check calls", {"block": "rag", "react": False}),
    ("check", "sft_best_check", "sft_best_check", "student_check", 300, "SFT setup on the check calls", {"block": "rag", "react": False}),
    ("check", "rl_best_check", "rl_best_check", "student_check", 300, "RL setup on the check calls", {"block": "rag", "react": False}),
]
TESTS = [  # the final test of each stage: the six setups
    ("memory_test", "Memory only", {"text": False}), ("raw_test", "Raw", {}), ("rag_test", "RAG", {"block": "rag"}),
    ("pi_test", "Pi", {"block": "pi_test"}), ("sft_best_test", "SFT", {"block": "rag"}), ("rl_best_test", "RL", {"block": "rag"})]
STAGE_TEST = {"week": ("test_w", 800, "2026-03"), "mini": ("test_d", 100, "2026-03")}
SFT_ARMS = [  # arm, name, rows of the SFT set, passes, what it tests
    ("read_eps", "read_eps", (0, 400), 2, "Read + EPS: read lines, Next EPS and Action"),
    ("score", "score", (0, 400), 2, "Score only: the Action line alone"),
    ("read", "read", (0, 400), 2, "Read: read lines and Action, no Next EPS"),
    ("hard", "hard", (0, 400), 2, "Hard: Action = 1 if the stock beat its month's median outcome"),
    ("placebo", "placebo", (0, 200), 2, "Control: Opus read lines and ratings shuffled within the month"),
    ("read_eps", "sft_grown", (400, 800), 2, "Grown: 400 more calls after read_eps, at a lower learning rate"),
    ("read_eps", "memo", (0, 50), 10, "Memorization check: 50 calls, 10 passes")]


def link(path):
    """The link text is the whole path, so two different files never share one link text."""
    return f"[{path}]({GH}{path})"


def chat_text(tok, msgs, lines="all"):
    """The exact text the model read: chat template, then the first answer label it continues from."""
    first = P.KEEP[lines][0] if P.KEEP[lines] else "Action"
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False) + f"{first}:"


class Prompts:
    """Builds each prompt once and writes it to examples/prompts/<row>_<variant>.txt."""

    def __init__(self, db, tok, stage="week"):
        self.db, self.tok, self.done = db, tok, {}
        self.logs = {}
        for st in ("week", "mini"):
            for name in ("pi", "pi_rag_docs", "pi_test"):
                f = RUNS / st / "agent" / f"{name}.jsonl"
                self.logs[(st, name)] = {x["row"]: x for x in map(json.loads, f.open())} if f.exists() else {}

    def block(self, r, text, kind, stage):
        if kind in ("rag", "random", "shuffled"):
            return blocks.KINDS[kind](self.db, r, text, self.tok)
        if kind in ("pi", "pi_rag_docs", "pi_test"):
            return self.logs[(stage, kind)][r.row]["block"]
        return ""

    def msgs(self, r, text=True, block="none", style="plain", position="flat", react=True, stage="week", **_):
        words = masked_text(self.db, r.row)
        return P.messages(r, words if text else "", self.block(r, words, block, stage), position, style, react)

    def get(self, r, text=True, block="none", style="plain", position="flat", react=True, lines="all", stage="week", **_):
        kind = "pi" if block in ("pi", "pi_test") else block
        parts = [str(r.row), "nocall" if not text else "", kind if kind != "none" else "noblock", style if style != "plain" else "",
                 "holds" if position == "long" else "", "noreaction" if not react else "", "actiononly" if lines == "action" else ""]
        name = "_".join(p for p in parts if p) + ".txt"
        if name not in self.done:
            msgs = self.msgs(r, text, block, style, position, react, stage)
            (PROMPTS / name).write_text(chat_text(self.tok, msgs, lines))
            self.done[name] = True
        return f"examples/prompts/{name}"

    def teacher(self, r, lens, react):
        import teacher
        name = f"{r.row}_opus_{lens}{'' if react else '_noreaction'}.txt"
        if name not in self.done:
            words = masked_text(self.db, r.row)
            p = teacher.params("claude-opus-5-5", lens, P.messages(r, words, blocks.rag(self.db, r, words), "flat", "plain", react))
            (PROMPTS / name).write_text(f"[system]\n{p['system']}\n\n[user]\n{p['messages'][0]['content']}\n\n"
                                        f"[output format: JSON with the fields {', '.join(teacher.SCHEMA['required'])}]\n")
            self.done[name] = True
        return f"examples/prompts/{name}"


def first(s, setname, rows_used=None, month=None):
    t = s[s.set == setname].sort_values("rank")
    if month:
        t = t[t.month == month]
    if rows_used is not None:
        t = t[t.row.isin(rows_used)]
    return t.head(N)


def outcomes(setname):
    if setname.startswith("test_"):
        return pd.read_parquet(DATA / "sealed" / f"{setname}.parquet")
    return pd.read_parquet(DATA / "outcomes.parquet")


def answers(calls, reads, pr, opts, setname, stage="week"):
    """One row per call: the prompt file, the answer, the probabilities and the real outcome."""
    o = outcomes(setname).set_index("row")
    rows = []
    for r in calls.itertuples():
        d = reads.loc[r.row]
        pos = opts.get("position", "flat")
        act = P.ACTIONS[pos]
        has_eps = d.p_eps_up == d.p_eps_up
        line = {c: d[c] if isinstance(d[c], str) else "" for c in LINE_COLS}
        if opts.get("force"):
            line["demand"] = opts["force"]
        rows.append({"row": r.row, "month": r.month, "prompt": pr.get(r, stage=stage, **opts), **line,
                     "next_eps": ("up" if d.p_eps_up > .5 else "down") if has_eps else "", "action": act[0] if d.p_pos > .5 else act[1],
                     "p_next_eps_up": d.p_eps_up if has_eps else "", f"p_{act[0].lower()}": d.p_pos, "p_mass": d.p_mass,
                     "excess_return": o.excess.get(r.row), "outcome_y": o.y.get(r.row), "next_eps_change": o.next_yoy.get(r.row)})
    return pd.DataFrame(rows)


def write(df, folder, name="answers.csv"):
    folder.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(folder / name, index=False)
    return str((folder / name).relative_to(ROOT))


def read_examples(s, pr, index):
    for step, exp, arm, setname, used, what, opts in READS:
        reads = pd.read_csv(RUNS / "week" / "reads" / f"{arm}.csv").set_index("row")
        calls = first(s, setname, reads.index)
        f = write(answers(calls, reads, pr, opts, setname), OUT / "week" / step / exp)
        index.append(("week", step, exp, what, f"{setname}, first {N} of {used}", f, f"runs/paper/week/reads/{arm}.csv"))


def test_examples(s, pr, index):
    for stage, (setname, used, month) in STAGE_TEST.items():
        calls = first(s, setname, month=month)
        for arm, setup, opts in TESTS:
            reads = pd.read_csv(RUNS / stage / "reads" / f"{arm}.csv").set_index("row")
            f = write(answers(calls, reads, pr, opts, setname, stage), OUT / stage / "test" / arm)
            index.append((stage, "test", arm, f"Final test of the {setup} setup", f"{setname}, the first {N} calls of March 2026 (of {used})", f, f"runs/paper/{stage}/reads/{arm}.csv"))
        base = []
        for rule, name in (("base_all", "BUY on every call"), ("base_words", "Word list"), ("base_eps", "Known-EPS rule"), ("base_numbers", "Numbers only")):
            b = pd.read_csv(RUNS / stage / "reads" / f"{rule}_test.csv").set_index("row")
            for r in calls.itertuples():
                base.append({"rule": name, "row": r.row, "month": r.month, "answer_tone": r.answer_tone, "eps_yoy": r.eps_yoy, "eps_change": r.eps_change,
                             "past": r.past, "risk": r.risk, "reaction": r.reaction, "dollars": r.dollars, "stock_score": b.p_pos.get(r.row), "p_next_eps_up": b.p_eps_up.get(r.row)})
        f = write(pd.DataFrame(base), OUT / stage / "test" / "simple_rules")
        index.append((stage, "test", "simple_rules", "Final test of the four simple rules (no AI), with the inputs each rule uses", f"{setname}, the same {N} calls", f, f"runs/paper/{stage}/reads/base_*_test.csv"))


def search_examples(s, db, index):
    """The real searches behind the RAG knowledge block: the query terms and every passage returned."""
    rows = []
    for r in first(s, "test_w", month="2026-03").itertuples():
        text = masked_text(db, r.row)
        q = kb.key_terms(db, text)
        date = str(r.date.date())
        for scope, k in (("own", 2), ("similar", 3), ("handbook", 1)):
            for x in kb.search(db, date, r.symbol, r.row, q, scope, k):
                when = db.execute("select available from passages where id = ?", [x["id"]]).fetchone()[0]
                rows.append({"row": r.row, "decision_date": date, "query": q, "scope": scope, "passage_id": x["id"], "source_call_row": x["call"],
                             "passage_date": when if scope != "handbook" else "", "outcome_card": x["after"] or "", "first_words": " ".join(x["text"].split()[:25])})
        rows.append({"row": r.row, "decision_date": date, "query": q, "scope": "facts", "first_words": kb.facts(db, date, r.symbol)})
    f = write(pd.DataFrame(rows), OUT / "week" / "rag", "searches.csv")
    index.append(("week", "rag", "searches", "The RAG search for each test call: query words, passages returned, their dates and outcome cards", f"test_w, the first {N} calls of March 2026", f, "data/paper/kb.sqlite (built by reproduce/build_data.sh)"))
    agent = []
    for st, name in (("week", "pi"), ("week", "pi_test")):
        log = {x["row"]: x for x in map(json.loads, (RUNS / st / "agent" / f"{name}.jsonl").open())}
        calls = first(s, "validation" if name == "pi" else "test_w", list(log), None if name == "pi" else "2026-03")
        for r in calls.itertuples():
            x = log[r.row]
            agent.append({"run": name, "row": r.row, "finished": x["code"] == 0, "seconds": x["seconds"], "tool_calls": x["calls"], "tokens_written": x["tokens"],
                          "queries": " | ".join(x["queries"]), "documents_kept": " ".join(map(str, x["docs"]))})
    f = write(pd.DataFrame(agent), OUT / "week" / "pi", "agent_runs.csv")
    index.append(("week", "pi", "agent_runs", "What the Pi agent searched for and kept, on validation and test calls", f"validation and test_w, first {N} each", f, "runs/paper/week/agent/*.jsonl"))


def teacher_examples(s, pr, index):
    lab = pd.DataFrame(map(json.loads, (DATA / "teacher" / "labels_v2.jsonl").open()))
    for step, setname, models, react, what in (("opus_labels", "sft", ["claude-opus-5-5"], True, "Opus answers used for SFT: neutral and skeptical instruction"),
                                               ("opus_check", "student_check", ["claude-opus-5-5", "claude-sonnet-5-5"], False, "Opus and Sonnet on the check calls")):
        calls = first(s, setname, lab[lab.model.isin(models)].row)
        out = []
        for r in calls.itertuples():
            for x in lab[(lab.row == r.row) & lab.model.isin(models)].itertuples():
                out.append({"row": r.row, "model": x.model, "instruction": x.lens, "prompt": pr.teacher(r, x.lens, react), **{c: getattr(x, c) for c in LINE_COLS},
                            "next_eps": x.next_eps, "rating": x.score, "quotes": " | ".join(x.quotes), "tokens_in": x.tokens_in, "tokens_out": x.tokens_out})
        f = write(pd.DataFrame(out), OUT / "week" / step)
        index.append(("week", step, "claude", what, f"{setname}, first {N}", f, "data/paper/teacher/labels_v2.jsonl"))


def sft_examples(s, pr, index):
    sft = s[s.set == "sft"].sort_values("rank")
    for arm, name, (a, b), passes, what in SFT_ARMS:
        t = T.targets(arm, sft.iloc[a:b]).sort_values("rank").head(N)
        lines = T.ARM_LINES[arm]
        out = []
        for x in t.itertuples():
            r = next(sft[sft.row == x.row].itertuples())
            act = P.ACTIONS[x.position]
            target = [f"{n}: {getattr(x, c)}" for (n, _), c in zip(P.LINES[:5], LINE_COLS)] if lines in ("all", "read") else []
            target += [f"Next EPS: {x.eps}"] if lines == "all" and isinstance(x.eps, str) else []
            target += [f"Action: {act[0]} with probability {x.p_pos:.2f}, {act[1]} with {1 - x.p_pos:.2f}"]
            out.append({"row": x.row, "month": x.month, "prompt": pr.get(r, block="rag", position=x.position, lines=lines), "position": x.position,
                        "target_answer": " | ".join(target), "first_action_target": x.p_pos, "passes": passes})
        f = write(pd.DataFrame(out), OUT / "week" / "sft_training" / name, "targets.csv")
        index.append(("week", "sft_training", name, what, f"sft rows {a} to {b}, first {N}", f, f"runs/paper/week/train/{name}.csv"))


def rl_examples(s, pr, index, model):
    rl = s[s.set == "rl"].sort_values("rank").merge(pd.read_parquet(DATA / "outcomes.parquet", columns=["row", "y", "next_yoy"]), on="row", how="left")
    rl["eps"] = rl.next_yoy.map(lambda v: None if pd.isna(v) else "up" if v > 0 else "down")
    rl["position"] = np.where(rl.row % 2 == 0, "flat", "long")
    start = {}
    if model:
        start = start_probs(rl.head(N), pr)
    for arm, n, what in (("trade", 400, "Trade reward (the RL setup)"), ("business", 400, "Business reward: the real Next EPS"), ("both", 400, "Business and trade rewards"),
                         ("shuffled", 200, "Control: trade reward with outcomes shuffled within the month"), ("control", 240, "Number-rule control: reward +3 if the past return is positive, else -3. The model does not see the call text")):
        w = T.rewards(rl.head(n), argparse.Namespace(arm="trade" if arm == "control" else arm, control=arm == "control")).head(N)
        w["month_mean_y"] = rl.head(n).groupby("month").y.transform("mean").head(N).values
        out = []
        for x in w.itertuples():
            act = P.ACTIONS[x.position]
            row = {"row": x.row, "month": x.month, "position": x.position, "prompt": pr.get(x, text=arm != "control", block="none" if arm == "control" else "rag", position=x.position),
                   "actions": " or ".join(act), "y": x.y, "month_mean_y": x.month_mean_y, "z": x.z, "daily_volatility": x.risk, "reward_first_action": x.r_pos, "reward_second_action": x.r_neg, "real_next_eps": x.eps}
            p = start.get(("base" if arm == "control" else "sft", x.row))
            if p:
                pa, pe = p["p_first"], p["p_eps"].get(x.eps) if isinstance(x.eps, str) else None
                trade = pa * x.r_pos + (1 - pa) * x.r_neg
                business = math.log(max(pe, 1e-6)) + math.log(2) if pe else 0.0
                reward = {"trade": trade, "business": business, "both": business / .7 + trade, "shuffled": trade, "control": trade}[arm]
                tilt = pa if arm == "business" else pa * math.exp(x.r_pos) / (pa * math.exp(x.r_pos) + (1 - pa) * math.exp(x.r_neg))
                row.update({"start_answer": p["read"], "start_p_first_action": pa, "start_p_real_eps": pe, "reward": reward, "target_p_first_action": tilt})
                if arm in ("business", "both") and pe:
                    row["target_p_real_eps"] = pe * math.exp(T.EPS_GAIN) / (pe * math.exp(T.EPS_GAIN) + (1 - pe))
            out.append(row)
        f = write(pd.DataFrame(out), OUT / "week" / "rl_training" / arm, "rewards.csv")
        index.append(("week", "rl_training", arm, what, f"rl, first {N} of {n}", f, f"runs/paper/week/train/{arm}.csv"))


def start_probs(calls, pr):
    """The start model's answer for each RL example call: the SFT adapter on the RAG prompt, and the base model
    on the no-text prompt that the number-rule control used."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from read import step, walk
    got = {}
    for tag, adapter, opts in (("sft", str(ROOT / "adapters/paper/week/sft_grown"), {"block": "rag"}), ("base", None, {"text": False, "block": "none"})):
        m, mtok = load(MODEL, adapter_path=adapter)
        for x in calls.itertuples():
            text = masked_text(pr.db, x.row) if opts.get("text", True) else ""
            block = pr.block(x, masked_text(pr.db, x.row), opts["block"], "week")
            ids = P.rendered(mtok, P.messages(x, text, block, x.position))
            cache = make_prompt_cache(m)
            logits, _ = step(mx, m, ids, cache)
            read, probs, _ = walk(mx, m, P.options(mtok, x.position), logits, cache)
            got[(tag, x.row)] = {"read": " | ".join(f"{k}: {v}" for k, v in read.items()), "p_first": float(probs["Action"][P.ACTIONS[x.position][0]]),
                                 "p_eps": {k: float(v) for k, v in probs["Next EPS"].items()}}
        del m
    return got


def training_tables(s):
    """The real training data, one row per call, without the transcript text: the SFT targets and the RL reward inputs."""
    folder = DATA / "training"
    folder.mkdir(exist_ok=True)
    lab = T.labels()
    rating = lab[lab.model == "claude-opus-5-5"].pivot_table(index="row", columns="lens", values="score").rename(columns=lambda c: f"opus_rating_{c}")
    sft = s[s.set == "sft"].sort_values("rank")
    parts = []
    for part, arm, rows in (("first 400 (read_eps)", "read_eps", sft.iloc[0:400]), ("next 400 (sft_grown)", "read_eps", sft.iloc[400:800]),
                            ("placebo 200", "placebo", sft.iloc[0:200])):
        t = T.targets(arm, rows).merge(rating, left_on="row", right_index=True, how="left")
        if arm == "placebo":  # its targets came from other calls of the month, so this call's own ratings would mislead
            t[["view", *rating.columns]] = np.nan
        t["part"] = part
        t["calls_in_month"] = t.groupby("month").row.transform("size")
        t["actions"] = t.position.map(lambda p: " or ".join(P.ACTIONS[p]))
        parts.append(t)
    cols = ["part", "row", "month", "position", "actions"] + LINE_COLS + ["eps", "opus_rating_neutral", "opus_rating_skeptic", "view", "calls_in_month", "p_pos"]
    names = {"eps": "next_eps_target", "view": "mean_rating", "p_pos": "first_action_target"}
    t = pd.concat(parts)[cols].rename(columns=names)
    t.round(4).to_csv(folder / "sft_targets.csv", index=False)
    rl = s[s.set == "rl"].sort_values("rank").merge(pd.read_parquet(DATA / "outcomes.parquet", columns=["row", "y", "next_yoy"]), on="row", how="left")
    rl["eps"] = rl.next_yoy.map(lambda v: None if pd.isna(v) else "up" if v > 0 else "down")
    rl["position"] = np.where(rl.row % 2 == 0, "flat", "long")
    out = []
    for arm, n in (("trade", 400), ("shuffled", 200)):  # business and both use the same 400 calls and rewards as trade
        w = T.rewards(rl.head(n), argparse.Namespace(arm=arm, control=False))
        w["reward_arm"] = "trade, business, both" if arm == "trade" else "shuffled (control)"
        w["month_mean_y"] = w.groupby("month").y.transform("mean")
        w["unit"] = w.risk * math.sqrt(60)
        w["actions"] = w.position.map(lambda p: " or ".join(P.ACTIONS[p]))
        out.append(w)
    w = pd.concat(out)[["reward_arm", "row", "month", "position", "actions", "y", "month_mean_y", "z", "risk", "unit", "r_pos", "r_neg", "eps"]]
    w.rename(columns={"risk": "daily_volatility", "r_pos": "reward_first_action", "r_neg": "reward_second_action", "eps": "next_eps"}).round(4).to_csv(folder / "rl_rewards.csv", index=False)
    return len(t), len(w)


ADAPTER = {"memo": "memo", "sft_read_eps": "read_eps", "sft_score": "score", "sft_read": "read", "sft_hard": "hard", "sft_placebo": "placebo",
           "sft_grown": "sft_grown", "sft_demand_up": "sft_grown", "sft_demand_down": "sft_grown", "sft_best_check": "sft_grown", "sft_best_test": "sft_grown",
           "rl_business": "business", "rl_both": "both", "rl_trade": "trade", "rl_shuffled": "shuffled", "rl_demand_up": "trade", "rl_demand_down": "trade",
           "rl_best_check": "trade", "rl_best_test": "trade"}  # the adapter each read used (configs/paper/commands.yaml)


def walk_detail(mx, model, tok, opts, logits, cache, force=None):
    """read.walk, line by line, keeping what the script saw: the probability of each allowed value's first token out of
    the whole vocabulary, their sum, the same values rescaled to sum to 1, and the five tokens the model liked most."""
    from read import step
    lines = []
    for i, (name, ids, after) in enumerate(opts):
        vals = list(ids)
        full = np.array(mx.softmax(logits.astype(mx.float32), axis=-1).tolist())
        raw = np.array([full[ids[v][0]] for v in vals])
        norm = raw / raw.sum()
        chosen = force if force and name == "Demand" else vals[int(norm.argmax())]
        top = np.argsort(-full)[:5]
        lines.append({"line": name, "values": vals, "raw": raw.tolist(), "sum": float(raw.sum()), "norm": norm.tolist(), "chosen": chosen,
                      "forced": bool(force and name == "Demand"), "top": [(tok.decode([int(t)]), float(full[t])) for t in top]})
        if i + 1 < len(opts):
            logits, _ = step(mx, model, ids[chosen] + after, cache)
    return lines


def fmt(x):
    """Four decimals, or scientific notation for values below 0.001, so that tiny probabilities do not show as 0.0000."""
    return f"{x:.4f}" if x >= 1e-3 or x == 0 else f"{x:.2e}"


def sample_page(folder, title, what, call, prompt, lines, position, stored):
    """sample_output.md: what the model wrote for one call, and how the script turned it into numbers."""
    (folder / "trace.json").write_text(json.dumps({"call": int(call), "prompt": prompt, "position": position, "lines": lines}, indent=1))
    act = P.ACTIONS[position]
    out = [f"# Sample output: {title}", "", what, "",
           f"Call {call}. The model read the prompt in [{prompt}]({GH}{prompt}). The prompt ends with the label `{lines[0]['line']}:`.", "",
           "## What the model wrote", "", "```"] + [f"{x['line']}: {x['chosen']}" for x in lines] + ["```", "",
           "The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.", "",
           "## How the script read each line", "",
           "Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.", "",
           "| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |", "| --- | --- | --- | --- | --- |"]
    for x in lines:
        for v, raw, norm in zip(x["values"], x["raw"], x["norm"]):
            mark = ("forced" if x["forced"] else "yes") if v == x["chosen"] else ""
            out.append(f"| {x['line']} | {v} | {fmt(raw)} | {norm:.4f} | {mark} |")
    by = {x["line"]: x for x in lines}
    a = by["Action"]
    out += ["", "## The two numbers that the scores use", ""]
    if "Next EPS" in by:
        e = by["Next EPS"]
        out.append(f"- **p(Next EPS up)** = {fmt(e['raw'][0])} ÷ ({fmt(e['raw'][0])} + {fmt(e['raw'][1])}) = **{e['norm'][0]:.4f}**")
    out.append(f"- **p({act[0]})** = {fmt(a['raw'][0])} ÷ ({fmt(a['raw'][0])} + {fmt(a['raw'][1])}) = **{a['norm'][0]:.4f}**. This is the stock score.")
    out += ["", f"The published file has p({act[0]}) = {stored:.4f} for this call. The values agree to the rounding of the published file." if abs(stored - a["norm"][0]) < 1e-3
            else f"The published file has p({act[0]}) = {stored:.4f} for this call.", "",
            "## The five tokens that the model liked most at each line", "",
            "These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example \" AVOID\" or \" evasive\", counts by its first token (\" A\", \" ev\"). The sign � is a part of a character that takes more than one token.", "",
            "| Line | Top five tokens (probability out of the whole vocabulary) |", "| --- | --- |"]
    for x in lines:
        out.append(f"| {x['line']} | " + ", ".join(f"`{t.strip() or repr(t)}` {fmt(p)}" for t, p in x["top"]) + " |")
    (folder / "sample_output.md").write_text("\n".join(out) + "\n")


def traces(s, pr):
    """Runs the model again on the first example call of each experiment and test, and writes sample_output.md."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from read import step
    jobs = []
    for step_, exp, arm, setname, used, what, opts in READS:
        reads = pd.read_csv(RUNS / "week" / "reads" / f"{arm}.csv").set_index("row")
        jobs.append(("week", ADAPTER.get(exp), OUT / "week" / step_ / exp, exp, what, next(first(s, setname, reads.index).itertuples()), opts, reads))
    for stage, (setname, used, month) in STAGE_TEST.items():
        for arm, setup, opts in TESTS:
            reads = pd.read_csv(RUNS / stage / "reads" / f"{arm}.csv").set_index("row")
            jobs.append((stage, ADAPTER.get(arm), OUT / stage / "test" / arm, f"{arm} ({stage} stage)", f"Final test of the {setup} setup.", next(first(s, setname, month=month).itertuples()), opts, reads))
    jobs.sort(key=lambda j: (j[0], j[1] or ""))
    loaded, model, mtok = "nothing loaded yet", None, None
    for stage, adapter, folder, title, what, r, opts, reads in jobs:
        path = str(ROOT / "adapters" / "paper" / stage / adapter) if adapter else None
        if path != loaded:
            model = None
            model, mtok = load(MODEL, adapter_path=path)
            loaded = path
        lines = opts.get("lines", "all")
        position = opts.get("position", "flat")
        msgs = pr.msgs(r, stage=stage, **opts)
        cache = make_prompt_cache(model)
        logits, _ = step(mx, model, P.rendered(mtok, msgs, lines), cache)
        detail = walk_detail(mx, model, mtok, P.options(mtok, position, lines), logits, cache, opts.get("force"))
        sample_page(folder, title, what, r.row, pr.get(r, stage=stage, **opts), detail, position, float(reads.loc[r.row].p_pos))
        print(title, r.row, flush=True)


def index_page(index):
    """examples/README.md: every experiment and test, with its example file and its full output file."""
    head = ["# Real examples of every experiment and test", "",
            "Each folder below holds real calls from one experiment or test. The calls are the first 10 of the call set that the experiment used, in the set's fixed random order. The test folders use the first 10 test calls of March 2026, so that the calls can be ranked inside one month.", "",
            "Each sample_output.md shows, for one call, what the model wrote and how the script turned the model's probabilities into the numbers that the scores use. Each answers file links to the exact prompt that the model read, in `prompts/`. Each targets or rewards file shows what training moved the model toward. `outcome_y` is the stock outcome and `next_eps_change` is the real EPS change.", "",
            "The full tables of training data are in `data/paper/training/`. `scripts/paper/examples.py` writes this folder.", ""]
    rows, last = [], None
    for stage, step, exp, what, calls, f, full in index:
        if (stage, step) != last:
            rows += ["", f"## {stage} stage: {step}", "", "| Experiment | What it tests | Calls | Sample output | Example | Full file |", "| --- | --- | --- | --- | --- | --- |"]
            last = (stage, step)
        full_link = link(full) if "*" not in full and "(" not in full else f"`{full}`"
        sample = Path(f).parent / "sample_output.md"
        sample_link = link(str(sample)) if (ROOT / sample).exists() else ""
        rows.append(f"| {exp} | {what} | {calls} | {sample_link} | {link(f)} | {full_link} |")
    (OUT / "README.md").write_text("\n".join(head + rows) + "\n")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", action="store_true")
    a = a.parse_args()
    from transformers import AutoTokenizer
    s = pd.read_parquet(DATA / "sets.parquet")
    print("training tables", training_tables(s))
    PROMPTS.mkdir(parents=True, exist_ok=True)
    db = kb.connect()
    pr = Prompts(db, AutoTokenizer.from_pretrained(MODEL))
    index = []
    read_examples(s, pr, index)
    search_examples(s, db, index)
    teacher_examples(s, pr, index)
    sft_examples(s, pr, index)
    rl_examples(s, pr, index, a.model)
    test_examples(s, pr, index)
    index.sort(key=lambda x: (["week", "mini"].index(x[0]), ["raw", "rag", "opus_check", "opus_labels", "sft_training", "sft", "rl_training", "rl", "pi", "check", "test"].index(x[1])))
    if a.model:
        traces(s, pr)
    index_page(index)
    print(len(index), "experiments,", len(pr.done), "prompts")


if __name__ == "__main__":
    main()
