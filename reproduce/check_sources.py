"""Compares each source table on disk with the SHA-256 the dataset published for the snapshot this run used
(data/raw/defeatbeta/spec.json, 2026-11-06). A mismatch is not an error: it means the upstream table has
changed since, and the rebuild should be checked against configs/paper/freeze.json.

  python3 reproduce/check_sources.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HERE = {"US/stock_earning_call_transcripts.parquet": RAW / "transcripts" / "defeatbeta" / "transcripts.parquet",
        "US/stock_prices.parquet": RAW / "defeatbeta" / "stock_prices.parquet",
        "US/stock_statement.parquet": RAW / "defeatbeta" / "stock_statement.parquet",
        "US/stock_profile.parquet": RAW / "defeatbeta" / "stock_profile.parquet",
        "US/stock_dividend_events.parquet": RAW / "defeatbeta" / "stock_dividend_events.parquet",
        "US/stock_split_events.parquet": RAW / "defeatbeta" / "stock_split_events.parquet",
        "US/company_tickers.json": RAW / "defeatbeta" / "company_tickers.json"}
# stock_tailing_eps.parquet is left out on purpose: the run refreshed it on 2026-11-12 to label late
# quarters, and the published copy is that refreshed table.


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    spec = json.loads((RAW / "defeatbeta" / "spec.json").read_text())["hashes"]
    same = 0
    for name, path in HERE.items():
        if not path.exists():
            print(f"missing  {path.relative_to(ROOT)}")
            continue
        ok = sha256(path) == spec[name]
        same += ok
        print(f"{'same   ' if ok else 'CHANGED'}  {path.relative_to(ROOT)}")
    print(f"{same} of {len(HERE)} tables match the 2026-11-06 snapshot")


if __name__ == "__main__":
    main()
