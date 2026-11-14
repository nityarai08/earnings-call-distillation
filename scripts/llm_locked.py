"""Run a command while holding logs/llm.lock, so the daily forward job (and other local
LLM jobs) wait instead of loading a second 9B model (two OOM on 18 GB).

  .venv/bin/python scripts/llm_locked.py <command> [args...]
"""
import fcntl
import subprocess
import sys
from pathlib import Path

lock = (Path(__file__).resolve().parents[1] / "logs" / "llm.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX)
sys.exit(subprocess.call(sys.argv[1:]))
