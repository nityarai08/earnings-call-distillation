#!/usr/bin/env bash
# Level 3: reruns a whole stage: every read, training run, agent run and score. Apple Silicon Mac only.
# The mini stage takes about a day; the week stage about six days of machine time. Run after build_data.sh.
#
#   bash reproduce/run_stage.sh mini
#   bash reproduce/run_stage.sh week
#
# The published run folder is moved to runs/paper/<stage>_published first, because the runner skips every
# command it finds already done. Results then land in runs/paper/<stage>/results.csv and findings.csv.
#
# Teacher labels: data/paper/teacher/labels_v2.jsonl already holds every Opus answer the run used, so the
# teacher steps find nothing left to send and cost nothing. To ask Opus afresh (about $37 for the week stage,
# and needing ANTHROPIC_API_KEY in .env), move that file aside first. Opus answers are not deterministic,
# and neither is training on the GPU, so a rerun reproduces the procedure, not every digit.
set -euo pipefail
cd "$(dirname "$0")/.."
stage="${1:?usage: run_stage.sh mini|week}"
[[ "$(uname -s)-$(uname -m)" == "Darwin-arm64" ]] || { echo "Needs an Apple Silicon Mac"; exit 1; }
[[ -f data/paper/kb.sqlite ]] || { echo "Build the knowledge base first: bash reproduce/build_data.sh"; exit 1; }

if [[ -d "runs/paper/$stage" && ! -d "runs/paper/${stage}_published" ]]; then
  mv "runs/paper/$stage" "runs/paper/${stage}_published"
fi
mkdir -p logs

# Keep the EPS table that is in the repo instead of downloading today's (the test step refreshes it otherwise)
export PAPER_PINNED_EPS=1
.venv/bin/python scripts/paper/run_step.py "$stage" 2>&1 | tee -a "logs/paper_$stage.log"
