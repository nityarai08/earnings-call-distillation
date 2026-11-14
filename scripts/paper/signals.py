"""Every possible positive signal of the week run, checked on the test calls and again on the 2025 validation calls.

  PYTHONPATH=src:scripts .venv/bin/python -W ignore scripts/paper/signals.py

Writes runs/paper/week/signals.csv: one row per check and call set, with the value, its 95% range from a bootstrap
of whole weeks (2,000 draws, seed 0, as in score.py) and the verdict. The checks:
  ic          stock IC of one setup or rule, against zero
  spread      mean return of the top third by p(BUY) minus the bottom third, within each month
  book        return after costs of the calls where p(BUY) > 0.5
  book_gain   difference in return after costs between two setups, on the same calls
  eps_ic      EPS IC of one setup, against zero
  eps_gain    paired EPS IC difference between two setups
  ic_gain     paired stock IC difference between two setups
  mix         IC of the mean month rank of RL and the two word and EPS rules, against the two rules alone
  eps_acc     share of calls where the EPS direction was right (no range)
  teacher     on the 300 check calls (2023-2024): Opus (rating / 100 as p(BUY)) against the small model's setups
Opus also read the 800 test calls (step opus_test: same prompt and RAG block as the RAG setup, neutral instruction).
Its knowledge runs to about June 2026, so it may know what happened after the March to May calls. The set
test_jul_aug repeats every test check on the July and August calls only, which came after that date.
Run only after score.py final has opened Test-W. These checks were chosen after the results were seen, so a range
that excludes zero here is a lead for Test-F, not a finding.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import score as S  # noqa: E402
from common import RUNS  # noqa: E402

BOOT = 2000
# name shown: (test read file, validation read file). RL is the arm best_rl.txt picked (trade), SFT the grown arm.
# Validation: the 500 calls of the first five setups. The RL, placebo and shuffled arms read the first 300 of them.
ARMS = {"No call": ("memory_test", "memory"), "Call only": ("raw_test", "raw"), "RAG": ("rag_test", "rag"), "Pi": ("pi_test", "pi"),
        "SFT": ("sft_best_test", "sft_grown"), "RL": ("rl_best_test", "rl_trade"), "Placebo SFT": (None, "sft_placebo"),
        "Shuffled RL": (None, "rl_shuffled"), "Known-EPS rule": ("base_eps_test", "base_eps"), "Word list": ("base_words_test", "base_words"),
        "Buy every call": ("base_all_test", None)}
PAIRS = [("No call", "Call only"), ("Call only", "RAG"), ("RAG", "SFT"), ("SFT", "RL"), ("Call only", "RL"), ("Placebo SFT", "RL"),
         ("Shuffled RL", "RL"), ("Known-EPS rule", "RL"), ("Word list", "RL"), ("Buy every call", "RL"),
         ("Call only", "Opus"), ("RAG", "Opus"), ("SFT", "Opus"), ("RL", "Opus"), ("Known-EPS rule", "Opus"), ("Word list", "Opus")]
CLEAN_FROM = "2026-07"  # first month after Opus's knowledge


def opus(rows):
    """Opus's neutral answers for these rows as reads: p(BUY) is its 0-100 rating / 100, p(EPS up) is 1 or 0."""
    import json
    from teacher import LABELS
    lab = pd.DataFrame(map(json.loads, LABELS.open()))
    t = lab[(lab.model == "claude-opus-5-5") & (lab.lens == "neutral") & lab.row.isin(rows)].drop_duplicates("row")
    return t.assign(p_pos=t.score.astype(float) / 100, p_eps_up=(t.next_eps == "up").astype(float))[["row", "p_pos", "p_eps_up"]]


def load(which):
    o = S.outcomes("test_w") if which == "test" else S.outcomes()
    sets = pd.read_parquet(S.DATA / "sets.parquet", columns=["row", "set"])
    keep = set(sets[sets.set == ("test_w" if which == "test" else "validation")].row)
    if which == "validation":  # the 500 validation calls the setups read; the rules scored all 3,000
        keep &= set(S.reads("week", "raw").row)
    out = {}
    for name, files in ARMS.items():
        f = files[0] if which == "test" else files[1]
        r = S.reads("week", f) if f else None
        if r is None:
            continue
        r = r[r.row.isin(keep)]
        if "p_eps_up" not in r:
            r["p_eps_up"] = np.nan
        out[name] = S.join(r, o).dropna(subset=["excess"])
    if which == "test":
        r = opus(keep)
        if len(r):
            out["Opus"] = S.join(r, o).dropna(subset=["excess"])
    return out


def check_calls():
    """Opus and four setups of the small model on the same 300 check calls. Opus's p(BUY) is its 0-100 rating / 100."""
    o = S.outcomes()
    out = {}
    for name, f in (("Call only", "raw_check"), ("RAG", "rag_check"), ("SFT", "sft_best_check"), ("RL", "rl_best_check")):
        out[name] = S.join(S.reads("week", f), o).dropna(subset=["excess"])
    out["Opus"] = S.join(opus(set(out["RL"].row)), o).dropna(subset=["excess"])
    return out


def thirds(d):
    r = d.groupby("month").p_pos.rank(pct=True, method="first")
    return d[r > 2 / 3].excess.mean() - d[r <= 1 / 3].excess.mean()


def book(d):
    b = d[d.p_pos > .5]
    return b.excess.mean() - S.ROUND_TRIP if len(b) else np.nan


def boot(d, f):
    """Value of f on d and its 95% range, resampling whole weeks."""
    rng = np.random.default_rng(0)
    weeks = {w: g for w, g in d.groupby("week")}
    names = list(weeks)
    got = [f(pd.concat([weeks[w] for w in rng.choice(names, len(names))])) for _ in range(BOOT)]
    got = [g for g in got if g is not None and not np.isnan(g)]
    return f(d), *np.quantile(got, [.025, .975])


def paired(a, b, cols):
    """The calls both setups scored, with a's columns as *_a and b's as *_b."""
    return a.merge(b[["row", *cols]], on="row", suffixes=("_a", "_b"))


def main():
    rows = []

    def add(check, which, what, n, v, lo, hi):
        verdict = "above zero" if lo > 0 else "below zero" if hi < 0 else "range includes zero"
        rows.append({"check": check, "calls_set": which, "what": what, "calls": n, "value": round(v, 4), "lo": round(lo, 4), "hi": round(hi, 4), "verdict": verdict})

    test = load("test")
    for which, D in (("test", test), ("validation", load("validation")), ("test_jul_aug", {n: d[d.month >= CLEAN_FROM] for n, d in test.items()})):
        for n, d in D.items():
            if d.p_pos.nunique() > 1:
                y = d.dropna(subset=["y"])
                add("ic", which, n, len(y), *boot(y, lambda x: S.ic(x, "p_pos", "y")))
                add("spread", which, n, len(d), *boot(d, thirds))
            if (d.p_pos > .5).any():
                add("book", which, n, int((d.p_pos > .5).sum()), *boot(d, book))
            if d.p_eps_up.notna().any():
                e = d.dropna(subset=["p_eps_up", "next_yoy"])
                add("eps_ic", which, n, len(e), *boot(e, lambda x: S.ic(x, "p_eps_up", "next_yoy")))
                rows.append({"check": "eps_acc", "calls_set": which, "what": n, "calls": len(e), "value": round(float(((e.p_eps_up > .5) == (e.next_yoy > 0)).mean()), 4)})
        for a, b in PAIRS:
            if a not in D or b not in D:
                continue
            d = paired(D[a], D[b], ["p_pos", "p_eps_up"]).dropna(subset=["y"])
            what = f"{b} minus {a}"
            if d.p_pos_a.nunique() > 1:
                add("ic_gain", which, what, len(d), *boot(d, lambda x: S.ic(x, "p_pos_b", "y") - S.ic(x, "p_pos_a", "y")))
            book_ab = lambda x: book(x.assign(p_pos=x.p_pos_b)) - book(x.assign(p_pos=x.p_pos_a))  # noqa: E731
            if (d.p_pos_a > .5).any() and (d.p_pos_b > .5).any():  # the placebo never says BUY, so it has no return
                add("book_gain", which, what, len(d), *boot(d, book_ab))
            if d.p_eps_up_a.notna().any() and d.p_eps_up_b.notna().any():
                e = d.dropna(subset=["p_eps_up_a", "p_eps_up_b", "next_yoy"])
                add("eps_gain", which, what, len(e), *boot(e, lambda x: S.ic(x, "p_eps_up_b", "next_yoy") - S.ic(x, "p_eps_up_a", "next_yoy")))
        m = D["RL"][["row", "month", "week", "y", "p_pos"]]
        for n, c in (("Word list", "words"), ("Known-EPS rule", "eps")):
            m = m.merge(D[n][["row", "p_pos"]].rename(columns={"p_pos": c}), on="row")
        for c in ("p_pos", "words", "eps"):
            m[c + "_r"] = m.groupby("month")[c].rank(pct=True)
        m = m.assign(mix=m[["p_pos_r", "words_r", "eps_r"]].mean(axis=1), rules=m[["words_r", "eps_r"]].mean(axis=1)).dropna(subset=["y"])
        add("mix", which, "RL + both rules", len(m), *boot(m, lambda x: S.ic(x, "mix", "y")))
        add("mix", which, "both rules", len(m), *boot(m, lambda x: S.ic(x, "rules", "y")))
        add("mix", which, "RL + both rules minus both rules", len(m), *boot(m, lambda x: S.ic(x, "mix", "y") - S.ic(x, "rules", "y")))
        for c in ("words", "eps"):
            rows.append({"check": "rank_corr", "calls_set": which, "what": f"RL with {'Word list' if c == 'words' else 'Known-EPS rule'}",
                         "calls": len(m), "value": round(float(m.p_pos_r.corr(m[c + "_r"])), 4)})
        if which == "test":
            for mo, g in D["RL"].groupby("month"):
                b = g[g.p_pos > .5]
                rows.append({"check": "book_month", "calls_set": which, "what": f"RL {mo}", "calls": len(b), "value": round(book(g), 4)})
    C = check_calls()
    for n, d in C.items():
        y = d.dropna(subset=["y"])
        add("teacher_ic", "check", n, len(y), *boot(y, lambda x: S.ic(x, "p_pos", "y")))
        add("teacher_book", "check", n, int((d.p_pos > .5).sum()), *boot(d, book))
    for n in ("Call only", "RAG", "SFT", "RL"):
        d = paired(C[n], C["Opus"], ["p_pos"]).dropna(subset=["y"])
        add("teacher_ic_gain", "check", f"Opus minus {n}", len(d), *boot(d, lambda x: S.ic(x, "p_pos_b", "y") - S.ic(x, "p_pos_a", "y")))
        add("teacher_book_gain", "check", f"Opus minus {n}", len(d), *boot(d, lambda x: book(x.assign(p_pos=x.p_pos_b)) - book(x.assign(p_pos=x.p_pos_a))))
    out = pd.DataFrame(rows)
    out.to_csv(RUNS / "week" / "signals.csv", index=False)
    print(out.to_string(), flush=True)


if __name__ == "__main__":
    main()
