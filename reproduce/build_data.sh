#!/usr/bin/env bash
# Level 2: rebuilds every dataset and the knowledge base from the source tables, then shows whether the
# rebuilt call sets match the published ones. Several hours; about 6 GB of disk. Run after setup.sh and
# download_data.sh.
#
#   bash reproduce/build_data.sh
#
# The rebuild overwrites the published copies in data/ and configs/paper/freeze.json, so in a git checkout
# `git diff --stat` shows exactly what came out different. It also writes the sealed final test part
# (data/paper/sealed/test_f.parquet), which .gitignore keeps out of the repo.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:scripts
PY=".venv/bin/python -W ignore"
mkdir -p logs

# 1. Text features and masked answer text for 2016-2024 calls (data/features/small_call_*.parquet)
$PY scripts/small_call_check.py build
# 2. The same for 2025-2026 calls, then prices, universe, outcomes, and the fixed sets (data/paper/*)
$PY scripts/paper/build_data.py parse
$PY scripts/paper/build_data.py market
$PY scripts/paper/build_data.py sets
# 3. The shared knowledge base (data/paper/kb.sqlite): masked passages, company facts, past-call cards,
#    and the analyst handbook (data/paper/handbook.json, written once by Claude Opus and published here)
$PY scripts/paper/build_kb.py build
$PY scripts/paper/build_kb.py handbook
$PY scripts/paper/build_kb.py audit

# 4. Did the rebuild give the same calls in the same sets?
$PY - <<'EOF'
import json
from pathlib import Path
new = json.loads(Path("configs/paper/freeze.json").read_text())
print("calls per set:", new["counts"])
EOF
if git rev-parse --git-dir >/dev/null 2>&1; then
  git --no-pager diff --stat -- configs/paper/freeze.json data/paper
  git --no-pager diff --quiet -- configs/paper/freeze.json && echo "Same sets and outcomes as the published run." \
    || echo "freeze.json changed: the source tables differ from the 2026-11-06 snapshot (see reproduce/check_sources.py)."
fi
