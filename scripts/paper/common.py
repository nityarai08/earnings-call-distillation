"""Shared pieces of the paper pipeline: paths, safe writing, run status and the spend ledger.
Every long job writes small units through these, so a stop loses one unit at most."""
import csv
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "paper"
SEALED = DATA / "sealed"
RUNS = ROOT / "runs" / "paper"
FREEZE = ROOT / "configs" / "paper" / "freeze.json"
LEDGER = RUNS / "spend.csv"
CAP = 200.0


def write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def done_rows(path, key="row"):
    """Rows already written to a results csv. A half-written last line is dropped, so that row is read again."""
    path = Path(path)
    if not path.exists():
        return set()
    lines = path.read_text().split("\n")
    head = lines[0].split(",")
    good = [x for x in lines[1:] if x and x.count(",") == len(head) - 1]
    if len(good) != len(lines[1:]) - 1 or lines[-1]:
        write_atomic(path, "\n".join([lines[0], *good]) + "\n")
    at = head.index(key)
    return {x.split(",")[at] for x in good}


class Status:
    """runs/paper/<stage>/<step>/status.json, read by tracker.py. Time adds up across restarts."""

    def __init__(self, stage, step, total=None):
        self.path = RUNS / stage / step / "status.json"
        self.s = json.loads(self.path.read_text()) if self.path.exists() else {"seconds": 0, "spent": 0, "rows_done": 0}
        if total is not None:
            self.s["rows_total"] = total
        self.mark = time.time()
        self.save("running")

    def save(self, state=None, note=None):
        now = time.time()
        self.s["seconds"] = self.s.get("seconds", 0) + now - self.mark
        self.mark = now
        if state:
            self.s["status"] = state
        if note is not None:
            self.s["note"] = note
        write_atomic(self.path, json.dumps(self.s))

    def rows(self, done):
        self.s["rows_done"] = done
        self.save()

    def spent(self, dollars):
        self.s["spent"] = self.s.get("spent", 0) + dollars
        self.save()

    def done(self, note=""):
        self.save("done", note)

    def failed(self, note):
        self.save("failed", note)


def spent_so_far():
    if not LEDGER.exists():
        return 0.0
    return sum(float(r["dollars"]) for r in csv.DictReader(LEDGER.open()))


def spend(dollars, what):
    """Records a paid call before it is sent, and refuses anything that would pass the cap."""
    total = spent_so_far()
    if total + dollars > CAP:
        raise RuntimeError(f"refused: {what} would take spend to ${total + dollars:.2f}, over the ${CAP:.0f} cap")
    new = not LEDGER.exists()
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        if new:
            f.write("at,dollars,what\n")
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{dollars:.4f},{what.replace(',', ' ')}\n")
        f.flush()
        os.fsync(f.fileno())
