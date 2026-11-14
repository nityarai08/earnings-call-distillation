"""Writes configs/paper/commands.yaml: the same steps for every stage, with that stage's sizes.
Only sizes and the final test part differ, so nothing can be dropped from one stage by accident."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PY = "PYTHONPATH=src:scripts .venv/bin/python -W ignore scripts/paper"
SIZES = {"mini": dict(val=100, arm_val=30, control=50, check=50, prompts=30, flatlong=20, probe=100, memo=20, sft=40, placebo=20, grown=80, rl=16, rl_shuffled=8, labels=80, faith=30, test="test_d", tset="test_d", dev_set="dev"),
         "week": dict(val=500, arm_val=300, control=200, check=300, prompts=100, flatlong=200, probe=800, memo=50, sft=400, placebo=200, grown=800, rl=400, rl_shuffled=200, labels=800, faith=200, test="test_w", tset="test_w", dev_set="validation"),
         "full": dict(val=3000, arm_val=1000, control=1000, check=1000, prompts=300, flatlong=200, probe=3000, memo=100, sft=1000, placebo=1000, grown=3000, rl=800, rl_shuffled=400, labels=3000, faith=500, test="test_f", tset="test_f", dev_set="validation")}


def read(stage, step, arm, s, n, extra=""):
    return f"{PY}/read.py --stage {stage} --step {step} --arm {arm} --set {s} --n {n} {extra}".strip()


def steps(stage, z):
    v = "dev" if stage == "mini" else "validation"
    ad = f"adapters/paper/{stage}"
    pi = f"runs/paper/{stage}/agent/pi.jsonl"
    forced = f"runs/paper/{stage}/agent/pi_rag_docs.jsonl"
    train = "PYTHONPATH=src:scripts .venv/bin/python scripts/llm_locked.py .venv/bin/python -W ignore scripts/paper/train_soft.py"  # holds the 9B lock for the whole run
    out = {}
    out["baselines"] = [f"{PY}/score.py selftest", f"{PY}/score.py baselines --stage {stage}"]
    out["raw"] = [read(stage, "raw", f"prompt_{x}", "dev", z["prompts"], f"--style {x}") for x in ("plain", "habit", "change")] + [
        read(stage, "raw", "raw", v, z["val"]), read(stage, "raw", "memory", v, z["val"], "--no-text"),
        read(stage, "raw", "raw_flat", "dev", z["flatlong"]), read(stage, "raw", "raw_long", "dev", z["flatlong"], "--position long"),
        read(stage, "raw", "probe", "sft", z["probe"], "--hidden")]
    out["rag"] = [read(stage, "rag", "rag", v, z["val"], "--block rag"), read(stage, "rag", "rag_random", v, z["control"], "--block random"), read(stage, "rag", "rag_shuffled", v, z["control"], "--block shuffled")]
    out["pi"] = [f"{PY}/agent_run.py --stage {stage} --step pi --arm pi --set {v} --n {z['val']}", read(stage, "pi", "pi", v, z["val"], f"--block {pi}"),
                 f"{PY}/agent_run.py --stage {stage} --step pi --arm pi_rag_docs --set {v} --n {z['control']} --forced", read(stage, "pi", "pi_rag_docs", v, z["control"], f"--block {forced}"),
                 f"{PY}/score.py pick --stage {stage} --group block --arms rag,pi"]
    out["opus_check"] = [f"{PY}/teacher.py submit --stage {stage} --step opus_check --model {m} --set student_check --n {z['check']} --lenses neutral --no-react" for m in ("claude-opus-5-5", "claude-sonnet-5-5")] + [
        f"{PY}/teacher.py wait --stage {stage} --step opus_check", read(stage, "opus_check", "student_rag", "student_check", z["check"], "--block rag --no-react")]
    out["opus_labels"] = [f"{PY}/teacher.py submit --stage {stage} --step opus_labels --model claude-opus-5-5 --set sft --n {z['labels']}", f"{PY}/teacher.py wait --stage {stage} --step opus_labels"]
    arms = [("read_eps", "all", z["sft"]), ("score", "action", z["sft"]), ("read", "read", z["sft"]), ("hard", "action", z["sft"]), ("placebo", "all", z["placebo"])]
    out["sft"] = [f"{train} sft --stage {stage} --step sft --arm read_eps --name memo --calls {z['memo']} --passes 10 --block rag",
                  read(stage, "sft", "memo", "sft", z["memo"], f"--block rag --adapter {ad}/memo"), f"{PY}/score.py memo --stage {stage} --arm memo"]
    grown = [f"{train} sft --stage {stage} --step sft --arm read_eps --name sft_grown --calls {z['grown'] - z['sft']} --skip {z['sft']} --passes 2 --block rag --start {ad}/read_eps --lr 3e-5,3e-6",
             read(stage, "sft", "sft_grown", v, z["val"], f"--block rag --adapter {ad}/sft_grown")]
    for i, (arm, ln, n) in enumerate(arms):
        lr = " --lr 3e-5,3e-6" if ln == "action" else ""  # one target token at 1e-4 diverged on the week run
        out["sft"] += [f"{train} sft --stage {stage} --step sft --arm {arm} --calls {n} --passes 2 --block rag{lr}", read(stage, "sft", f"sft_{arm}", v, z["arm_val"], f"--block rag --lines {ln} --adapter {ad}/{arm}")]
        out["sft"] += grown if i == 0 else []
    out["sft"] += [f"{PY}/score.py pick --stage {stage} --group sft --arms sft_score,sft_read,sft_read_eps,sft_hard"]
    out["rl"] = [f"{train} rl --stage {stage} --step rl --arm trade --name control --calls 240 --no-text --control --block none"]
    for arm, n in (("business", z["rl"]), ("both", z["rl"]), ("trade", z["rl"]), ("shuffled", z["rl_shuffled"])):
        out["rl"] += [f"{train} rl --stage {stage} --step rl --arm {arm} --calls {n} --block rag --start {ad}/sft_grown", read(stage, "rl", f"rl_{arm}", v, z["arm_val"], f"--block rag --adapter {ad}/{arm}")]
    out["rl"] += [f"{PY}/score.py pick --stage {stage} --group rl --arms rl_business,rl_trade,rl_both"]
    best = f"$(sed s/rl_// runs/paper/{stage}/best_rl.txt)"
    for tag, folder in (("sft", "sft_grown"), ("rl", best)):
        out["rl"] += [read(stage, "rl", f"{tag}_demand_{x}", v, z["faith"], f"--block rag --adapter {ad}/{folder} --force Demand={x}") for x in ("up", "down")]
    out["rl"] += [f"{PY}/score.py faith --stage {stage} --arms sft,rl"]
    t, n = z["tset"], 10 ** 6
    check = [("raw", ""), ("rag", "--block rag"), ("pi", "--block rag"), ("sft_best", f"--block rag --adapter {ad}/sft_grown"), ("rl_best", f"--block rag --adapter {ad}/{best}")]
    out["test"] = [f"{PY}/build_data.py refresh_eps", f"{PY}/score.py baselines --stage {stage} --set {t} --suffix _test", read(stage, "test", "memory_test", t, n, "--no-text"), read(stage, "test", "raw_test", t, n), read(stage, "test", "rag_test", t, n, "--block rag"),
                   f"{PY}/agent_run.py --stage {stage} --step test --arm pi_test --set {t}", read(stage, "test", "pi_test", t, n, f"--block runs/paper/{stage}/agent/pi_test.jsonl"),
                   read(stage, "test", "sft_best_test", t, n, f"--block rag --adapter {ad}/sft_grown"), read(stage, "test", "rl_best_test", t, n, f"--block rag --adapter {ad}/{best}")]
    out["test"] += [read(stage, "test", f"{name}_check", "student_check", z["check"], f"{extra} --no-react") for name, extra in check]
    out["report"] = [f"{PY}/score.py final --stage {stage} --open {t}"]
    if stage != "mini":  # most promising first: distillation and RL before pi and the comparison arms
        order = ["baselines", "raw", "rag", "opus_check", "opus_labels", "sft", "rl", "pi", "test", "report"]
        out = {k: out[k] for k in order}
    return out


def main():
    plan = {stage: steps(stage, z) for stage, z in SIZES.items()}
    (ROOT / "configs" / "paper" / "commands.yaml").write_text(yaml.safe_dump(plan, sort_keys=False, width=1000))
    print({k: {s: len(c) for s, c in v.items()} for k, v in plan.items()})


if __name__ == "__main__":
    main()
