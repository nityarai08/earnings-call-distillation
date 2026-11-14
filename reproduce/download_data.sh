#!/usr/bin/env bash
# Downloads the three large source tables that are not in this repo, from the Hugging Face dataset
# defeatbeta/yahoo-finance-data (ODC-BY), then checks them against the hashes of the copies the run used.
# About 2.9 GB.
#
#   bash reproduce/download_data.sh
#
# The dataset is updated daily and its history was compacted on 2026-11-13, so the 2026-11-06 snapshot
# this run used is no longer downloadable. Newer files usually still rebuild the same calls, because the
# pipeline cuts every table at 2026-11-06; reproduce/build_data.sh checks that against configs/paper/freeze.json.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/US"

get() {  # get <file in the dataset> <path here>
  mkdir -p "$(dirname "$2")"
  if [[ -f "$2" ]]; then echo "have  $2"; return; fi
  echo "fetch $1"
  curl -fL --retry 3 -o "$2.part" "$BASE/$1"
  mv "$2.part" "$2"
}

get stock_earning_call_transcripts.parquet data/raw/transcripts/defeatbeta/transcripts.parquet
get stock_prices.parquet                   data/raw/defeatbeta/stock_prices.parquet
get stock_statement.parquet                data/raw/defeatbeta/stock_statement.parquet

python3 reproduce/check_sources.py
