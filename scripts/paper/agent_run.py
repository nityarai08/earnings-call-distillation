"""Pi searches the knowledge base for each call, one case at a time. Saves the documents it chose as a
knowledge block, which read.py then scores like any other block. A stop loses one case.

  PYTHONPATH=src:scripts .venv/bin/python -W ignore scripts/paper/agent_run.py --stage mini --step pi --arm pi --set dev --n 100 [--forced] [--adapter x]

--forced returns RAG's documents whatever the model asks for, so it isolates the multi-turn format from the
choice of documents. Pi runs with no built-in tools under a macOS sandbox that blocks all but localhost.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import blocks  # noqa: E402
import kb  # noqa: E402
import prompt as P  # noqa: E402
from common import DATA, ROOT, RUNS, Status  # noqa: E402
from kb_service import Case, Service  # noqa: E402
from read import MODEL, masked_text  # noqa: E402

PI = ROOT / "tools" / "pi"
TIMEOUT, CHUNK, PORT = 360, 100, 8099
SYSTEM = ("You are an experienced stock analyst preparing to rate one company. Its name is masked. You have a search tool over past calls, "
          "similar companies and an analyst handbook. Names are masked in it, so search by topic words such as products, costs or demand, never by company name. "
          "Read the documents you find. Make at most four tool calls in all, choosing what would most change your view, then reply DONE.")


def serve(adapter):
    cmd = [str(ROOT / ".venv/bin/python"), "-m", "mlx_lm", "server", "--model", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "--chat-template-args", '{"enable_thinking":false}']
    if adapter:
        cmd += ["--adapter-path", adapter]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        if subprocess.run(["curl", "-s", "-o", "/dev/null", f"127.0.0.1:{PORT}/health"]).returncode == 0:
            return p
        time.sleep(1)
    p.kill()
    raise RuntimeError("local server did not start")


def one(r, text):
    task = f"{P.facts_line(r, 'flat')}\n\nManagement's answers:\n{text}\n\nSearch for what you need, then reply DONE."
    env = {**os.environ, "PI_CODING_AGENT_DIR": str(PI / "agent"), "PI_OFFLINE": "1"}
    cmd = ["sandbox-exec", "-f", str(PI / "no_network.sb"), str(PI / "node_modules/.bin/pi"), "--model", "local/mlx-community/Qwen3.5-9B-4bit", "--no-builtin-tools", "--no-extensions", "-e", str(PI / "kb_tools.ts"),
           "--no-session", "--no-context-files", "--no-skills", "--offline", "--system-prompt", SYSTEM, "--mode", "json", "-p", task]
    start = time.time()
    try:
        p = subprocess.run(cmd, cwd=PI, env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=TIMEOUT)
        return p.returncode, p.stdout.decode(errors="ignore"), time.time() - start
    except subprocess.TimeoutExpired:
        return -1, "", TIMEOUT


def run(a):
    s = pd.read_parquet(DATA / "sets.parquet")
    s = s[s.set == a.set].sort_values("rank").head(a.n)
    out = RUNS / a.stage / "agent" / f"{a.arm}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = {json.loads(x)["row"] for x in out.open() if x.endswith("\n")} if out.exists() else set()
    status = Status(a.stage, a.step)
    db = kb.connect()
    todo = s[~s.row.isin(seen)]
    service = Service()
    while len(todo):
        part, todo = todo.head(CHUNK), todo.iloc[CHUNK:]
        lock = (ROOT / "logs" / "llm.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        server = serve(a.adapter)
        try:
            for r in part.itertuples():
                text = masked_text(db, r.row)
                forced = None
                if a.forced:
                    q = kb.key_terms(db, text)
                    date = str(r.date.date())
                    forced = (kb.search(db, date, r.symbol, r.row, q, "own", 2) + kb.search(db, date, r.symbol, r.row, q, "similar", 3)) if q else []
                c = service.case = Case(r, forced)
                code, log, secs = one(r, text)
                service.case = None
                block = blocks.render(c.own, c.similar, c.facts, c.notes)
                with out.open("a") as f:
                    f.write(json.dumps({"row": int(r.row), "code": code, "seconds": round(secs, 1), "calls": c.calls, "tokens": c.tokens, "queries": c.queries,
                                        "docs": [x["id"] for x in c.own + c.similar + c.notes], "block": block}) + "\n")
                status.save()
        finally:
            server.kill()
            server.wait()
            fcntl.flock(lock, fcntl.LOCK_UN)
    status.save()


def main():
    p = argparse.ArgumentParser()
    for k in ("stage", "step", "arm", "set"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--n", type=int, default=10 ** 9)
    p.add_argument("--forced", action="store_true")
    p.add_argument("--adapter")
    run(p.parse_args())


if __name__ == "__main__":
    main()
