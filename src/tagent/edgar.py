"""SEC EDGAR download: filing index (submissions) and XBRL company facts.

Raw JSON is cached under data/raw/edgar/ and parsed only with `json`. Requests
carry SEC_USER_AGENT from .env (SEC fair-access policy) and stay under 10/s.

  python -m tagent.edgar            # fetch for all symbols in the case universe
"""
from __future__ import annotations

import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

from .data import ROOT, broad_symbols

EDGAR_DIR = ROOT / "data" / "raw" / "edgar"
_last = [0.0]


def _get(url: str) -> requests.Response:
    load_dotenv(ROOT / ".env")
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise SystemExit("Set SEC_USER_AGENT=\"Name email\" in .env (SEC requires a contact).")
    wait = 0.12 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=60)
    return r


def ticker_cik_map() -> dict[str, int]:
    """Current ticker -> CIK (delisted tickers are absent; they stay uncovered)."""
    path = EDGAR_DIR / "company_tickers.json"
    if not path.exists():
        EDGAR_DIR.mkdir(parents=True, exist_ok=True)
        r = _get("https://www.sec.gov/files/company_tickers.json")
        r.raise_for_status()
        path.write_bytes(r.content)
    raw = json.loads(path.read_text())
    return {v["ticker"].replace(".", "-"): int(v["cik_str"]) for v in raw.values()}


def fetch_company(cik: int, refresh: bool = False) -> tuple[bool, bool]:
    """Download submissions (incl. older pages) and companyfacts for one CIK.
    refresh=True re-downloads the main submissions file and companyfacts (new filings)."""
    sub_dir, fac_dir = EDGAR_DIR / "submissions", EDGAR_DIR / "companyfacts"
    sub_dir.mkdir(parents=True, exist_ok=True)
    fac_dir.mkdir(parents=True, exist_ok=True)
    c = f"CIK{cik:010d}"
    ok_sub = (sub_dir / f"{c}.json").exists() and not refresh
    if not ok_sub:
        r = _get(f"https://data.sec.gov/submissions/{c}.json")
        if r.ok:
            sub = r.json()
            for f in sub.get("filings", {}).get("files", []):  # older filing pages
                if f.get("filingTo", "9999") < "2014-01-01" or (sub_dir / f["name"]).exists():
                    continue
                rr = _get(f"https://data.sec.gov/submissions/{f['name']}")
                if rr.ok:
                    (sub_dir / f["name"]).write_bytes(rr.content)
            (sub_dir / f"{c}.json").write_bytes(r.content)
            ok_sub = True
    ok_fac = (fac_dir / f"{c}.json").exists() and not refresh
    if not ok_fac:
        r = _get(f"https://data.sec.gov/api/xbrl/companyfacts/{c}.json")
        if r.ok:
            (fac_dir / f"{c}.json").write_bytes(r.content)
            ok_fac = True
    return ok_sub, ok_fac


def main(symbols: list[str]):
    cmap = ticker_cik_map()
    have = [s for s in symbols if s in cmap]
    print(f"{len(have)}/{len(symbols)} symbols map to a current CIK", flush=True)
    n_sub = n_fac = 0
    for i, s in enumerate(have):
        a, b = fetch_company(cmap[s])
        n_sub += a
        n_fac += b
        if i % 50 == 0:
            print(f"[{i+1}/{len(have)}] {s} submissions={a} facts={b}", flush=True)
    print(f"done: submissions {n_sub}, companyfacts {n_fac}")


if __name__ == "__main__":
    syms = sys.argv[1:] or [s for s in broad_symbols() if not s.startswith("^")]
    main(syms)
