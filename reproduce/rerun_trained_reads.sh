#!/usr/bin/env bash
# Reruns the test reads of the two trained models with the published adapters, without retraining, and
# compares the new probabilities with the published ones. Apple Silicon Mac, after build_data.sh (it needs the
# knowledge base). About 7 hours for both: 800 calls each at about 4 calls a minute.
#
#   bash reproduce/rerun_trained_reads.sh            # both
#   bash reproduce/rerun_trained_reads.sh sft        # or just one: sft | rl
#   N=20 bash reproduce/rerun_trained_reads.sh sft   # a quick check on the first 20 calls
#
# adapters/paper/week/sft_grown  the SFT model scored on Test-W (fine-tuned on 800 Opus-labelled calls)
# adapters/paper/week/trade      the RL model scored on Test-W (trade reward, started from sft_grown)
#
# New reads go to runs/paper/check/reads/; the published files are not touched.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:scripts
PY=".venv/bin/python"
[[ -f data/paper/kb.sqlite ]] || { echo "Build the knowledge base first: bash reproduce/build_data.sh"; exit 1; }
mkdir -p logs

for m in ${*:-sft rl}; do
  case "$m" in
    sft) adapter=sft_grown ;;
    rl)  adapter=trade ;;
    *)   echo "usage: rerun_trained_reads.sh [sft] [rl]"; exit 1 ;;
  esac
  arm="${m}_best_test"
  # the same command as the week run's test step (configs/paper/commands.yaml), into a separate stage folder
  $PY -W ignore scripts/paper/read.py --stage check --step test --arm "$arm" --set test_w --n "${N:-1000000}" --block rag \
      --adapter "adapters/paper/week/$adapter"
  $PY -W ignore - "$arm" <<'EOF'
import sys
import pandas as pd
arm = sys.argv[1]
old = pd.read_csv(f"runs/paper/week/reads/{arm}.csv")
new = pd.read_csv(f"runs/paper/check/reads/{arm}.csv")
d = old.merge(new, on="row", suffixes=("_published", "_rerun"))
for col in ("p_pos", "p_eps_up"):
    gap = (d[f"{col}_published"] - d[f"{col}_rerun"]).abs()
    print(f"{arm} {col}: {len(d)} calls, largest difference {gap.max():.5f}, mean {gap.mean():.5f}")
same = sum((d[f"{n}_published"] == d[f"{n}_rerun"]).mean() for n in ("demand", "pricing", "margins", "guidance", "candor")) / 5
print(f"{arm} read lines identical on {same:.1%} of values")
EOF
done
