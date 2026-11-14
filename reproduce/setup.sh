#!/usr/bin/env bash
# Sets up the full pipeline on an Apple Silicon Mac (macOS 14+, 18 GB memory or more).
# Needs Python 3.12 and Node 22.19+ on PATH. Downloads the 9B model (about 5.6 GB).
#
#   bash reproduce/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[[ "$(uname -s)-$(uname -m)" == "Darwin-arm64" ]] || { echo "The full pipeline needs an Apple Silicon Mac (MLX). Re-scoring works anywhere: reproduce/rescore.sh"; exit 1; }

python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# The student model, pinned to the revision the run used
.venv/bin/hf download mlx-community/Qwen3.5-9B-4bit --revision 8b2b98c00a6b4d291155e4890773ca8f769aee53

# The pi agent (pinned in tools/pi/package-lock.json)
node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22||(a==22&&b>=19)?0:1)' \
  || { echo "pi needs Node 22.19 or later"; exit 1; }
(cd tools/pi && npm ci)

mkdir -p logs
[[ -f .env ]] || cp .env.example .env
echo "Setup done. Next: bash reproduce/download_data.sh"
