"""Runs one step of configs/paper/commands.yaml, or a whole stage in order. Safe to run twice: finished
commands are skipped, and every command itself resumes from its last saved unit. A command that fails is
retried once. Then the step is marked failed and the runner stops, so it can be fixed and restarted.

  .venv/bin/python scripts/paper/run_step.py mini            # the whole stage
  .venv/bin/python scripts/paper/run_step.py mini sft        # one step
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, RUNS, sha, write_atomic  # noqa: E402



def scripts_hash():
    h = hashlib.sha256()
    for f in sorted((ROOT / "scripts" / "paper").glob("*.py")):
        h.update(sha(f).encode())
    return h.hexdigest()[:16]


def run(stage, step, cmds):
    folder = RUNS / stage / step
    folder.mkdir(parents=True, exist_ok=True)
    done_file = folder / "commands_done.json"
    done = json.loads(done_file.read_text()) if done_file.exists() else []
    log = folder / "run.log"
    start = time.time()
    for cmd in cmds:
        if cmd in done:
            continue
        for attempt in (1, 2):
            with log.open("a") as f:
                f.write(f"\n=== {time.strftime('%F %T')} attempt {attempt}: {cmd}\n")
                f.flush()
                code = subprocess.call(f"caffeinate -i zsh -c {json.dumps(cmd)}", shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            if code == 0:
                break
        if code:
            write_atomic(folder / "FAILED.json", json.dumps({"cmd": cmd, "code": code, "at": time.strftime("%F %T")}))
            s = folder / "status.json"
            x = json.loads(s.read_text()) if s.exists() else {}
            write_atomic(s, json.dumps({**x, "status": "failed", "note": f"See {log.relative_to(ROOT)}"}))
            print("FAILED", stage, step, cmd[:120], flush=True)
            sys.exit(1)
        done.append(cmd)
        write_atomic(done_file, json.dumps(done))
    s = folder / "status.json"
    x = json.loads(s.read_text()) if s.exists() else {}
    write_atomic(s, json.dumps({**x, "status": "done", "seconds": max(x.get("seconds", 0), time.time() - start), "note": x.get("note", "")}))
    write_atomic(folder / "DONE.json", json.dumps({"at": time.strftime("%F %T"), "scripts": scripts_hash(), "commands": len(cmds)}))
    print("done", stage, step, flush=True)


def main():
    plan = yaml.safe_load((ROOT / "configs" / "paper" / "commands.yaml").read_text())
    stage = sys.argv[1]
    steps = sys.argv[2:] or list(plan[stage])  # the order commands.yaml gives
    for s in steps:
        if not (RUNS / stage / s / "DONE.json").exists():
            run(stage, s, plan[stage][s])


if __name__ == "__main__":
    main()
