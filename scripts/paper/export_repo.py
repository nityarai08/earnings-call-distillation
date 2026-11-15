"""Copies the paper pipeline, its own data and its results into a clean clone of the public repo.

  .venv/bin/python scripts/paper/export_repo.py ../earnings-call-distillation

Files keep their paths, so the scripts run unchanged in the clone. Three files are cut down on the way:
the price bars keep only SPY (the only symbol the paper reads), the old case set keeps only its symbol
column, and the read sample keeps only its row column. Rows of the sealed Test-F part are dropped from
every run file, the retrieved transcript text is dropped from the pi agent logs, and the export stops if anything from Test-F, a key or a private email would leave.
The three example calls in examples/ keep their full prompts on purpose, as illustrations.
Files written by hand in the repo (README, docs, reproduce/) are never touched.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]).resolve()

CODE = ["scripts/paper/" + f for f in ("common.py", "build_data.py", "build_kb.py", "kb.py", "kb_service.py", "blocks.py", "prompt.py", "read.py",
                                         "agent_run.py", "teacher.py", "score.py", "train_soft.py", "run_step.py", "make_commands.py", "tracker.py",
                                         "interim.py", "export_repo.py", "examples.py", "signals.py")] + \
       ["scripts/" + f for f in ("small_call_check.py", "call_qa_check.py", "filing_change_check.py", "raw_text_check.py", "small_call_read.py", "llm_locked.py")] + \
       ["src/tagent/" + f for f in ("__init__.py", "data.py", "evaluate.py", "risk_agent.py", "earnings_text.py", "edgar.py", "edgar_features.py",
                                     "filing_changes.py", "train_lora.py")] + \
       ["configs/paper/commands.yaml", "configs/paper/steps.yaml", "configs/paper/freeze.json", "configs/universe.yaml",
        "tools/pi/kb_tools.ts", "tools/pi/no_network.sb", "tools/pi/package.json", "tools/pi/package-lock.json", "tools/pi/agent/models.json"]
DATA = ["data/paper/" + f for f in ("sets.parquet", "calls.parquet", "outcomes.parquet", "kb_outcomes.parquet", "feat_new.parquet", "handbook.json",
                                     "teacher/labels.jsonl", "teacher/labels_v2.jsonl", "sealed/test_d.parquet", "sealed/test_w.parquet",
                                     "training/sft_targets.csv", "training/rl_rewards.csv")] + \
       ["data/raw/defeatbeta/" + f for f in ("spec.json", "company_tickers.json", "stock_profile.parquet", "stock_dividend_events.parquet",
                                              "stock_split_events.parquet", "stock_tailing_eps.parquet")] + \
       ["data/features/small_call_features.parquet", "reports/diagnostics/small_call_events.parquet", "runs/paper/spend.csv"] + \
       [f"adapters/paper/week/{a}/{f}" for a in ("sft_grown", "trade") for f in ("adapters.safetensors", "adapter_config.json")]
# The two adapters scored on Test-W (SFT, and the RL arm best_rl.txt picked). Weights only: without the optimizer
# state and state.json the trainer starts fresh instead of trying to resume from them.
PAPER = ["reports/paper/conference/" + f for f in ("main.tex", "refs.bib", "main.pdf")]  # the LaTeX paper and its built PDF
RUNS = ["runs/paper/mini", "runs/paper/week"]
SKIP = re.compile(r"\.bak$|__pycache__|\.lock$|\.part$")


def copy(rel):
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rel, dst)


def row_of(line):
    try:
        x = json.loads(line)
    except ValueError:
        return None
    return x.get("row") if isinstance(x, dict) else None


def runs(sealed):
    """Run folders, with any Test-F row taken out of tables and line files."""
    for top in RUNS:
        for src in sorted((ROOT / top).rglob("*")):
            if not src.is_file() or SKIP.search(str(src)):
                continue
            rel = src.relative_to(ROOT)
            dst = OUT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix == ".csv":
                try:
                    d = pd.read_csv(src)
                except pd.errors.EmptyDataError:
                    d = None
                if d is not None and "row" in d.columns:
                    d[~d.row.isin(sealed)].to_csv(dst, index=False)
                    continue
            elif src.suffix == ".jsonl":
                keep = [ln for ln in src.read_text().splitlines() if row_of(ln) not in sealed]
                if src.parent.name == "agent":  # the retrieved transcript text is not ours to share; queries and document ids stay
                    keep = [json.dumps({k: v for k, v in json.loads(ln).items() if k != "block"}) for ln in keep if ln.strip()]
                dst.write_text("".join(ln + "\n" for ln in keep))
                continue
            elif src.suffix in (".log", ".json", ".txt", ".yaml"):
                dst.write_text(src.read_text(errors="replace").replace(str(ROOT) + "/", ""))  # paths in logs become repo-relative
                continue
            shutil.copy2(src, dst)


def trimmed():
    """The three inputs published smaller than they are on disk."""
    b = pd.read_parquet(ROOT / "data/raw/daily_bars.parquet")
    (OUT / "data/raw").mkdir(parents=True, exist_ok=True)
    b[b.symbol == "SPY"].reset_index(drop=True).to_parquet(OUT / "data/raw/daily_bars.parquet", index=False)
    (OUT / "data/cases").mkdir(parents=True, exist_ok=True)
    pd.read_parquet(ROOT / "data/cases/train.parquet", columns=["symbol"]).drop_duplicates().to_parquet(OUT / "data/cases/train.parquet", index=False)
    pd.read_csv(ROOT / "reports/diagnostics/small_call_read.csv", usecols=["row"]).to_csv(OUT / "reports/diagnostics/small_call_read.csv", index=False)


def check(sealed):
    """Stops the export if Test-F rows, keys or private emails made it into the clone."""
    bad = []
    for f in OUT.rglob("*"):
        if not f.is_file() or ".git" in f.parts:
            continue
        rel = str(f.relative_to(OUT))
        if "test_f" in f.name or f.name in (".env", "auth.json") or f.name == "kb.sqlite" or "text_new" in f.name:
            bad.append(rel + ": not allowed")
        if f.suffix in (".csv", ".parquet") and "configs/" not in rel and "data/paper/sets" not in rel and "data/paper/calls" not in rel and "feat_new" not in rel:
            d = pd.read_csv(f) if f.suffix == ".csv" else pd.read_parquet(f)
            if "row" in d.columns and d.row.isin(sealed).any() and f.parent.name != "sealed" and "outcomes" not in f.name:
                bad.append(rel + ": holds Test-F rows")
            if f.suffix == ".parquet" and "outcomes" in f.name and d.row.isin(sealed).any():
                bad.append(rel + ": Test-F outcomes")
        if f.suffix in (".py", ".md", ".json", ".jsonl", ".yaml", ".csv", ".ts", ".sh", ".txt", ".sb"):
            t = f.read_text(errors="ignore")
            if re.search(r"sk-ant-[A-Za-z0-9]", t) or re.search(r"ANTHROPIC_API_KEY\s*=\s*['\"]?sk-", t):
                bad.append(rel + ": looks like an API key")
            for m in re.findall(r"[\w.+-]+@[\w-]+\.(?:com|org|net|io)", t):
                personal = m.split("@")[1].lower() in ("gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "me.com", "aol.com")
                if rel.startswith("examples/") and not personal:
                    continue  # a company's own investor-relations address, read out in its call
                if personal or not m.endswith(("@example.com", "@anthropic.com", "users.noreply.github.com")):
                    bad.append(f"{rel}: email {m}")
    if bad:
        sys.exit("export stopped:\n" + "\n".join(sorted(set(bad))))


def main():
    sealed = set(json.loads((ROOT / "configs/paper/freeze.json").read_text())["test_rows"]["test_f"])
    for rel in CODE + DATA + PAPER:
        copy(rel)
    trimmed()
    runs(sealed)
    (OUT / "docs").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "logs/paper_fixes.md", OUT / "docs/FIXES.md")  # what broke during the runs and how it was fixed
    shutil.copytree(ROOT / "examples", OUT / "examples", dirs_exist_ok=True)  # three real calls, written by examples.py
    shutil.copytree(ROOT / "blog", OUT / "blog", dirs_exist_ok=True)
    check(sealed)
    files = [f for f in OUT.rglob("*") if f.is_file() and ".git" not in f.parts]
    print(len(files), "files,", round(sum(f.stat().st_size for f in files) / 1e6, 1), "MB in", OUT)


if __name__ == "__main__":
    main()
