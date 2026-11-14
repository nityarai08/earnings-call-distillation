"""Rows for the paper run tracker dashboard, from configs/paper/steps.yaml and what the runs wrote.

  .venv/bin/python scripts/paper/tracker.py

Each step leaves runs/paper/<stage>/<step>/status.json:
  {"status": "running|done|failed", "rows_done": 120, "rows_total": 500, "seconds": 1440, "spent": 3.1, "note": ""}
Each stage leaves results.csv (setup, calls, read_match, eps_acc, eps_ic, stock_acc, stock_ic, book_net)
and findings.csv (question, result, verdict). Time left is each step's plan, rescaled by how the finished steps of
the stage compared with their plans, less time already used. A step that reports rows uses its real speed
once 20 rows are in. Calendar time adds the allowance.
"""
import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "paper"
OUT = ROOT / "reports" / "paper" / "tracker"
MIN_ROWS = 20


def span(h):
    return f"{h:.0f} h" if h < 48 else f"{h / 24:.1f} days"


def status(stage, step):
    f = RUNS / stage / step / "status.json"
    return json.loads(f.read_text()) if f.exists() else {}


def table(stage, name):
    f = RUNS / stage / f"{name}.csv"
    return list(csv.DictReader(f.open())) if f.exists() else []


def number(x):
    return float(x) if x not in (None, "") else None


def left(plan, s, pace):
    if s.get("status") == "done":
        return 0
    used = s.get("seconds", 0) / 3600
    if s.get("status") in ("running", "failed"):
        done, total = s.get("rows_done", 0), s.get("rows_total", 0)
        if total and done >= MIN_ROWS:
            return used / done * (total - done)
        return max(plan * pace - used, plan * .1)
    return plan * pace


def stage_rows(c):
    steps, stages = [], []
    for g in c["stages"]:
        info = [(x, status(g["id"], x["id"])) for x in g["steps"]]
        done = [(x, s) for x, s in info if s.get("status") == "done" and x["hours"]]
        planned, used = sum(x["hours"] for x, _ in done), sum(s.get("seconds", 0) / 3600 for _, s in done)
        pace = min(max(used / planned, .5), 2) if planned >= 2 else 1  # a short step says little about the rest
        rows = []
        for n, (x, s) in enumerate(info):
            total, got = s.get("rows_total"), s.get("rows_done")
            state = {"done": "Done", "running": "Running", "failed": "Failed"}.get(s.get("status"), "Waiting")
            hours = left(x["hours"], s, pace)
            used = s.get("seconds", 0) / 3600
            share = 1 if state == "Done" else got / total if total else min(used / (used + hours), .99) if used + hours else 0
            rows.append({"key": f"{g['name']}:{x['id']}", "stage": g["name"], "step": n + 1, "name": x["name"], "work": x["work"], "status": state,
                         "pct": round(share, 4), "planned_h": x["hours"], "actual_h": round(s.get("seconds", 0) / 3600, 2) if s else None,
                         "left_h": round(hours, 2), "rows_done": got, "rows_total": total, "note": s.get("note", "")})
        steps += rows
        plan = sum(r["planned_h"] for r in rows)
        hours = sum(r["left_h"] for r in rows)
        wall = hours if g["id"] == "build" else hours * c["allowance"]
        state = "Done" if all(r["status"] == "Done" for r in rows) else "Failed" if any(r["status"] == "Failed" for r in rows) \
            else "Running" if any(r["status"] != "Waiting" for r in rows) else "Waiting"
        stages.append({"stage": g["name"], "order": len(stages), "status": state, "gated": g["gated"], "test": g["test"], "calls": g["calls"], "steps": len(rows),
                       "steps_done": sum(r["status"] == "Done" for r in rows), "planned_h": round(plan, 1), "left_h": round(hours, 1), "left_wall_h": round(wall, 1), "left_text": span(wall), "spend_text": f"${sum(x['spend'] for x in g['steps'])} planned",
                       "pct": round(sum(r["planned_h"] * r["pct"] for r in rows) / plan, 4), "spend_plan": sum(x["spend"] for x in g["steps"]),
                       "spent": round(sum(s.get("spent", 0) for _, s in info), 2)})
    return steps, stages


def summary(c, steps, stages, now):
    active = next((s for s in stages if s["status"] != "Done"), stages[-1])
    step = next((s for s in steps if s["stage"] == active["stage"] and s["status"] != "Done"), None)
    upto = [s for s in stages if not s["gated"]]
    through = sum(s["left_wall_h"] for s in upto)
    full = next(s for s in stages if s["gated"])
    spent = sum(s["spent"] for s in stages)
    failed = next((s for s in steps if s["status"] == "Failed"), None)
    started = any(s["status"] != "Waiting" for s in steps)
    if failed:
        head = f"{failed['stage']} stopped at {failed['name']}. It resumes from its last saved point once fixed."
    elif not started:
        head = "Nothing has started. Build comes first, then the one-day run, then the week."
    elif active["gated"] and active["status"] == "Waiting":
        head = "The week is finished. The full run starts only if its results earn it."
    elif active["status"] == "Waiting":
        done = [s["stage"] for s in stages if s["status"] == "Done"]
        head = f"{done[-1] if done else 'The last stage'} is finished. {active['stage']} starts when you approve it."
    else:
        head = f"{active['stage']} is running, on {step['name'] if step else 'its last step'}."
    return [{"key": "now", "headline": head, "stage": active["stage"], "step": step["name"] if step else "", "stage_left": span(active["left_wall_h"]),
             "week_left_h": round(through, 1), "week_left": span(through), "week_eta": (now + timedelta(hours=through)).astimezone().strftime("%b %-d, %-I %p") if started else "Starts at go-ahead",
             "full_left": span(full["left_wall_h"]), "spent": f"${spent:,.0f}", "planned_spend": sum(s["spend_plan"] for s in stages), "spend_note": f"of ${c['budget']} cap, ${sum(s['spend_plan'] for s in stages)} planned",
             "started": started, "updated": now.astimezone().strftime("%b %-d, %-I:%M %p")}]


def results(c):
    rows = []
    for g in c["stages"]:
        got = {r["setup"]: r for r in table(g["id"], "results")}
        for kind, names in (("Setup", c["setups"]), ("Baseline", c["baselines"])):
            for name in names:
                r = got.get(name, {})
                rows.append({"key": f"{g['name']}:{name}", "stage": g["name"], "kind": kind, "setup": name, "calls": number(r.get("calls")),
                             **{k: number(r.get(k)) for k in ("read_match", "eps_acc", "eps_ic", "stock_acc", "stock_ic", "book_net")}})
    return rows


def findings(c):
    rows = []
    for g in c["stages"]:
        got = {r["question"]: r for r in table(g["id"], "findings")}
        for q in c["questions"]:
            r = got.get(q["id"], {})
            rows.append({"key": f"{g['name']}:{q['id']}", "stage": g["name"], "question": q["text"], "result": r.get("result", ""), "verdict": r.get("verdict", "")})
    return rows


def write(name, rows):
    tmp = OUT / f".{name}.tmp"
    tmp.write_text(json.dumps(rows))
    os.replace(tmp, OUT / f"{name}.json")


def main():
    c = yaml.safe_load((ROOT / "configs" / "paper" / "steps.yaml").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    steps, stages = stage_rows(c)
    s = summary(c, steps, stages, now)
    for name, rows in (("summary", s), ("stages", stages), ("steps", steps), ("results", results(c)), ("findings", findings(c))):
        write(name, rows)
    print(s[0]["headline"], "| week left", s[0]["week_left"], "| full", s[0]["full_left"], "| planned spend", s[0]["planned_spend"])
    for x in stages:
        print(x["stage"], x["status"], x["planned_h"], x["left_wall_h"], x["spend_plan"])


if __name__ == "__main__":
    main()
