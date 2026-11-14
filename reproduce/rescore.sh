#!/usr/bin/env bash
# Level 1: re-score the published model outputs and check they give the published tables.
# macOS or Linux (Windows: WSL), Python 3.10+. Takes about a minute. Needs: pip install -r requirements-rescore.txt
#
#   bash reproduce/rescore.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
export PYTHONPATH=src:scripts
mkdir -p logs

saved="$(mktemp -d)"
for f in results findings faithfulness; do cp "runs/paper/week/$f.csv" "$saved/$f.csv"; done

# final: scores every setup on the 800 opened test calls (Test-W) and runs the paired bootstrap
"$PY" -W ignore scripts/paper/score.py final --stage week --open test_w
# faith: the forced-Demand test on 200 tuning calls
"$PY" -W ignore scripts/paper/score.py faith --stage week --arms sft,rl

status=0
for f in results findings faithfulness; do
  if cmp -s "$saved/$f.csv" "runs/paper/week/$f.csv"; then
    echo "OK    runs/paper/week/$f.csv matches the published table"
  else
    echo "DIFF  runs/paper/week/$f.csv differs from the published table:"; diff "$saved/$f.csv" "runs/paper/week/$f.csv" || true
    status=1
  fi
done
exit $status
