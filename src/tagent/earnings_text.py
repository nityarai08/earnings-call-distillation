"""Earnings press releases (8-K Item 2.02, exhibit 99) as a text data source.

  python -m tagent.earnings_text fetch                  # watchlist + S&P 100, 2015+
  python -m tagent.earnings_text score [model]          # local LLM reads each release -> JSON scores
  python -m tagent.earnings_text claude-pilot           # 5 direct Sonnet 5.5 calls: shape + cost
  python -m tagent.earnings_text claude-submit          # same prompt/window via Message Batches
  python -m tagent.earnings_text claude-collect BATCH   # -> earnings_text_scores_v2_claude-sonnet-5-5.parquet
  python -m tagent.earnings_text claude-identify-submit # leak check: name the masked company (200 releases)
  python -m tagent.earnings_text claude-identify-collect BATCH

v1 (data/raw/edgar/earnings_text/, scored by the 2B into earnings_text_scores.parquet) picked
exhibits by filename only: ~4% were supplements or SGML headers, and masking without word
boundaries left fragments ("the Companympany"). v2 checks the content of each candidate,
masks on word boundaries, and keeps the unmasked text so masking can be redone offline.

Fetch: for every earnings 8-K, list the filing directory, pick the exhibit-99 press
release, strip HTML, mask the company's own name and ticker, keep the first 12k
characters (headline, highlights, guidance). Saved under data/raw/edgar/earnings_text/.

Score: a small local model rates each masked release (tone, guidance direction, results
vs expectations language). Scores attach to the reaction session like the other
earnings-event features, so a decision only sees releases whose 2-day window closed.
"""
from __future__ import annotations

import html
import json
import re
import sys

import pandas as pd

from .data import ROOT, load_universe, sp100_test_universe
from .edgar import EDGAR_DIR, _get, ticker_cik_map
from .edgar_features import _filings

TEXT_DIR_V1 = EDGAR_DIR / "earnings_text"
SCORES_V1 = ROOT / "data" / "features" / "earnings_text_scores.parquet"
TEXT_DIR = EDGAR_DIR / "earnings_text_v2"
MAX_CHARS = 12000
DEFAULT_SCORER = "mlx-community/Qwen3.5-9B-4bit"


def scores_path(model: str = DEFAULT_SCORER):
    """v2 scores file for a given reader, e.g. earnings_text_scores_v2_Qwen3.5-9B-4bit.parquet"""
    return ROOT / "data" / "features" / f"earnings_text_scores_v2_{model.split('/')[-1]}.parquet"


def _company_names(cik: int) -> list[str]:
    j = json.loads((EDGAR_DIR / "submissions" / f"CIK{cik:010d}.json").read_text())
    names = [j.get("name", "")] + [f.get("name", "") for f in j.get("formerNames", [])]
    out = set()
    for n in names:
        n = re.sub(r"[,.]", " ", n).strip()
        if n:
            out.add(n)
            # also the name without corporate suffixes ("Microsoft Corp" -> "Microsoft")
            core = re.sub(r"\b(inc|corp|corporation|co|company|ltd|plc|holdings?|group|/de|/new)\b",
                          "", n, flags=re.I).strip()
            if len(core) >= 3:
                out.add(core)
    return sorted(out, key=len, reverse=True)


SUFFIX = r"(?:,?\s+(?:inc|incorporated|corp|corporation|co|company|ltd|plc|holdings?|group)\b\.?)*"


def mask(text: str, names: list[str], tickers: list[str]) -> str:
    for n in names:  # the name plus any corporate suffix that follows it in the text
        text = re.sub(rf"\b{re.escape(n)}\b{SUFFIX}", "the Company", text, flags=re.I)
    for t in tickers:
        text = re.sub(rf"\b{re.escape(t)}\b", "TICKER", text)
    return text


def html_to_text(raw: bytes) -> str:
    s = raw.decode("utf-8", errors="ignore")
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\xa0]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n", s).strip()


def candidates(items: list[dict], primary: str) -> list[str]:
    """Exhibit documents in order of likelihood of being the press release. Excludes the
    full-submission .txt and index pages; the primary 8-K document goes last (some
    companies put the release in the 8-K body)."""
    docs = [i for i in items if re.search(r"\.(htm|html|txt)$", i["name"], re.I)
            and i["name"] != primary and "index" not in i["name"]
            and not re.fullmatch(r"R\d+\.htm", i["name"])  # XBRL viewer pages
            and not re.fullmatch(r"\d{10}-\d{2}-\d{6}\.txt", i["name"])]
    ex = [i for i in docs if re.search(r"ex[-_]?99|exhibit99|dex99", i["name"], re.I)] or docs
    # 99.1 (ex99-1, ex99_1, ex99d1, ex991, ex9901) before 99.2, then the larger file
    ex.sort(key=lambda i: (0 if re.search(r"99(?:[-_.d]|\.0)?0?1(?!\d)", i["name"], re.I) else 1,
                           -int(i.get("size") or 0)))
    return [i["name"] for i in ex] + [primary]


RELEASE_RE = re.compile(r"\b(announc|report|post|deliver)\w*\b.{0,160}?\b(results|quarter|earnings)\b",
                        re.I | re.S)
NOT_RELEASE_RE = re.compile(r"supplement|financial data|highlights|presentation|non-gaap", re.I)
TITLE_RE = re.compile(r"(news|press|earnings) release|for immediate release", re.I)


def looks_like_release(text: str, primary: bool = False) -> bool:
    """Heuristic. Strong: a release title or announcement wording in the first 400
    characters (and no "supplement"). Weak: announcement wording further down, and no
    supplement/slides/non-GAAP title. An 8-K body (primary) usually only points to the
    release, so it must also be long."""
    head = text[:400]
    strong = ((RELEASE_RE.search(head) or TITLE_RE.search(head))
              and not re.search(r"supplement", head[:300], re.I))
    weak = RELEASE_RE.search(text[:2500]) and not NOT_RELEASE_RE.search(text[:250])
    return bool(strong or weak) and (not primary or len(text) > 6000)


def _get_retry(url: str, tries: int = 3):
    import time
    for k in range(tries):
        r = _get(url)
        if r.ok or r.status_code == 404:
            return r
        time.sleep(2 * (k + 1))
    return r


def fetch(symbols: list[str], since: str = "2014-06-01"):
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    cmap = ticker_cik_map()
    n_new = n_fail = 0
    for k, sym in enumerate(symbols):
        cik = cmap.get(sym)
        if cik is None:
            continue
        f = _filings(cik)
        if f.empty:
            continue
        f = f[(f.form == "8-K") & f["items"].fillna("").str.contains("2.02", regex=False)
              & (f.filingDate >= since)]
        names = _company_names(cik)
        for r in f.itertuples():
            acc = r.accessionNumber.replace("-", "")
            out = TEXT_DIR / f"{cik}_{acc}.json"
            if out.exists():
                continue
            base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}"
            idx = _get_retry(f"{base}/index.json")
            if not idx.ok:
                n_fail += 1
                continue
            first = None
            for name in candidates(idx.json().get("directory", {}).get("item", []), r.primaryDocument)[:4]:
                doc = _get_retry(f"{base}/{name}")
                if not doc.ok:
                    continue
                raw = html_to_text(doc.content)
                first = first or (name, raw)
                if looks_like_release(raw, primary=name == r.primaryDocument):
                    first = (name, raw)
                    break
            if first is None:
                n_fail += 1
                continue
            name, raw = first
            text = mask(raw, names, [sym, sym.replace("-", ".")])[:MAX_CHARS]
            out.write_text(json.dumps({"symbol": sym, "cik": cik, "accession": r.accessionNumber,
                                       "acceptance": r.acceptanceDateTime, "file": name,
                                       "is_release": looks_like_release(raw, name == r.primaryDocument),
                                       "text": text,
                                       "raw_text": raw[:MAX_CHARS]}))
            n_new += 1
        print(f"[{k+1}/{len(symbols)}] {sym}: {len(f)} earnings 8-Ks (new total {n_new}, failed {n_fail})", flush=True)


SCORE_SYSTEM = """You read a company's quarterly earnings press release (company name masked) \
and rate it. Use only the text. Reply with JSON only."""

SCORE_USER = """Rate this earnings release.
Return JSON: {{"tone": integer -2..2 (very negative .. very positive overall),
 "guidance": integer -1, 0 or 1 (lowered / unchanged or none / raised outlook),
 "vs_expectations": integer -1, 0 or 1 (language of missing / neutral / beating expectations
   or records), "demand": integer -1, 0 or 1 (weakening / stable / strengthening demand or bookings)}}

RELEASE:
{text}"""


def score(model: str = DEFAULT_SCORER, chars: int = 3000):
    """Score every fetched release with a local model (resumable). Holds the LLM lock so
    the daily forward job waits instead of loading a second model."""
    import fcntl
    from .llm import LocalLLM
    out = scores_path(model)
    done = set(pd.read_parquet(out).accession) if out.exists() else set()
    files = sorted(TEXT_DIR.glob("*.json"))
    lock = (ROOT / "logs" / "llm.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    llm = LocalLLM(model)
    rows = []
    for i, p in enumerate(files):
        d = json.loads(p.read_text())
        if d["accession"] in done:
            continue
        msgs = [{"role": "system", "content": SCORE_SYSTEM},
                {"role": "user", "content": SCORE_USER.format(text=d["text"][:chars])}]
        g = llm.chat(msgs, max_tokens=80)
        m = re.search(r"\{.*\}", g.text, re.S)
        try:
            s = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            s = {}
        rows.append({"symbol": d["symbol"], "cik": d["cik"], "accession": d["accession"],
                     "acceptance": d["acceptance"], **{k: s.get(k) for k in
                                                       ("tone", "guidance", "vs_expectations", "demand")},
                     "valid": bool(s), "model": model, "is_release": d.get("is_release")})
        if len(rows) % 100 == 0:
            _save(rows, out)
            rows = []
            print(f"[{i+1}/{len(files)}] scored ({g.latency_s:.1f}s/item)", flush=True)
    _save(rows, out)


CLAUDE_MODEL = "claude-sonnet-5-5"
CLAUDE_PRICE = (2.0, 10.0)  # $/1M tokens (input, output); Batch API is half price
SCORE_SCHEMA = {
    "type": "object",
    "properties": {"tone": {"type": "integer", "enum": [-2, -1, 0, 1, 2]},
                   **{k: {"type": "integer", "enum": [-1, 0, 1]}
                      for k in ("guidance", "vs_expectations", "demand")}},
    "required": ["tone", "guidance", "vs_expectations", "demand"],
    "additionalProperties": False,
}


def claude_params(d: dict, model: str = CLAUDE_MODEL, chars: int = 3000) -> dict:
    """Same prompt and text window as the local reader; thinking off (between_tools) to
    match the non-thinking local models; the schema only constrains the JSON shape."""
    return {"model": model, "max_tokens": 200, "system": SCORE_SYSTEM,
            "thinking": {"type": "between_tools"},
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCORE_SCHEMA}},
            "messages": [{"role": "user", "content": SCORE_USER.format(text=d["text"][:chars])}]}


def _claude_row(d: dict, msg, model: str) -> dict:
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        s = json.loads(text) if msg.stop_reason != "refusal" else {}
    except json.JSONDecodeError:
        s = {}
    return {"symbol": d["symbol"], "cik": d["cik"], "accession": d["accession"],
            "acceptance": d["acceptance"], **{k: s.get(k) for k in ("tone", "guidance", "vs_expectations", "demand")},
            "valid": bool(s), "model": model, "is_release": d.get("is_release")}


def _cost(usage, batch: bool) -> float:
    c = (usage.input_tokens * CLAUDE_PRICE[0] + usage.output_tokens * CLAUDE_PRICE[1]) / 1e6
    return c / 2 if batch else c


def claude_pilot(n: int = 5, model: str = CLAUDE_MODEL):
    """A few direct calls: checks the request shape and measures tokens per release."""
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()
    files = sorted(TEXT_DIR.glob("*.json"))[:: max(1, len(list(TEXT_DIR.glob("*.json"))) // n)][:n]
    spent = 0.0
    for p in files:
        d = json.loads(p.read_text())
        msg = client.messages.create(**claude_params(d, model))
        spent += _cost(msg.usage, batch=False)
        print(d["symbol"], d["accession"], _claude_row(d, msg, model), msg.usage.input_tokens, msg.usage.output_tokens)
    print(f"pilot cost ${spent:.4f}; full batch estimate ${spent / len(files) * len(list(TEXT_DIR.glob('*.json'))) / 2:.2f}")


def claude_submit(model: str = CLAUDE_MODEL):
    """All releases not yet scored by `model`, as one Message Batch (half price)."""
    import time
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    out = scores_path(model)
    done = set(pd.read_parquet(out).accession) if out.exists() else set()
    reqs = []
    for p in sorted(TEXT_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        if d["accession"] not in done:
            reqs.append({"custom_id": p.stem, "params": claude_params(d, model)})
    batch = anthropic.Anthropic().messages.batches.create(requests=reqs)
    log = ROOT / "data" / "features" / "claude_batches.jsonl"
    with log.open("a") as fh:
        fh.write(json.dumps({"batch_id": batch.id, "n": len(reqs), "model": model, "task": "earnings_text_v2",
                             "created": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    print(f"submitted {len(reqs)} requests: batch {batch.id} ({batch.processing_status})")


IDENTIFY_USER = """This is the start of a quarterly earnings press release. The company's name \
and ticker are masked as "the Company" and "TICKER". Which company issued it? Give your best \
guess even if unsure.
Return JSON: {{"ticker": US ticker symbol, "confidence": 0..1}}

RELEASE:
{text}"""
IDENTIFY_SCHEMA = {"type": "object", "properties": {"ticker": {"type": "string"}, "confidence": {"type": "number"}},
                   "required": ["ticker", "confidence"], "additionalProperties": False}


def claude_identify_submit(n: int = 200, model: str = CLAUDE_MODEL, chars: int = 3000, seed: int = 0):
    """Leak check: can the reader name the masked company? If it can, it may also
    remember how the stock moved after the release, and its scores can carry hindsight."""
    import random
    import time
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    files = sorted(TEXT_DIR.glob("*.json"))
    files = random.Random(seed).sample(files, n)
    reqs = [{"custom_id": p.stem, "params": {
        "model": model, "max_tokens": 100, "thinking": {"type": "between_tools"},
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": IDENTIFY_SCHEMA}},
        "messages": [{"role": "user", "content": IDENTIFY_USER.format(text=json.loads(p.read_text())["text"][:chars])}]}}
        for p in files]
    batch = anthropic.Anthropic().messages.batches.create(requests=reqs)
    with (ROOT / "data" / "features" / "claude_batches.jsonl").open("a") as fh:
        fh.write(json.dumps({"batch_id": batch.id, "n": len(reqs), "model": model, "task": "identify_company",
                             "created": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    print(f"submitted {len(reqs)} identify requests: batch {batch.id}")


def _identify_report(df: pd.DataFrame, out, spent: float = 0.0):
    df.to_csv(out, index=False)
    print(f"named the company in {df.hit.mean():.1%} of {len(df)} releases (cost ${spent:.2f}) -> {out}")
    print("hit rate by year:", df.groupby("year").hit.mean().round(2).to_dict())
    print("hit rate by confidence >= 0.7:", round(df[df.confidence >= 0.7].hit.mean(), 2),
          f"({(df.confidence >= 0.7).sum()} releases)")


def identify_local(model: str = DEFAULT_SCORER, n: int = 200, chars: int = 3000, seed: int = 0):
    """The same leak check with a local reader (same 200 releases as the Claude check)."""
    import fcntl
    import random
    from .llm import LocalLLM
    files = random.Random(seed).sample(sorted(TEXT_DIR.glob("*.json")), n)
    lock = (ROOT / "logs" / "llm.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    llm = LocalLLM(model)
    rows = []
    for p in files:
        d = json.loads(p.read_text())
        g = llm.chat([{"role": "user", "content": IDENTIFY_USER.format(text=d["text"][:chars])}], max_tokens=60)
        m = re.search(r"\{.*\}", g.text, re.S)
        try:
            s = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            s = {}
        guess = str(s.get("ticker", "")).upper().replace(".", "-")
        rows.append({"symbol": d["symbol"], "accession": d["accession"], "year": d["acceptance"][:4],
                     "guess": guess, "confidence": s.get("confidence"), "hit": guess == d["symbol"]})
    df = pd.DataFrame(rows)
    df["confidence"] = pd.to_numeric(df.confidence, errors="coerce")
    _identify_report(df, ROOT / "reports" / f"identify_{model.split('/')[-1]}.csv")


def claude_identify_collect(batch_id: str, model: str = CLAUDE_MODEL):
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()
    b = client.messages.batches.retrieve(batch_id)
    if b.processing_status != "ended":
        print(f"not done: {b.processing_status} {b.request_counts}")
        return
    rows, spent = [], 0.0
    for res in client.messages.batches.results(batch_id):
        if res.result.type != "succeeded":
            continue
        d = json.loads((TEXT_DIR / f"{res.custom_id}.json").read_text())
        msg = res.result.message
        spent += _cost(msg.usage, batch=True)
        g = json.loads(next((b.text for b in msg.content if b.type == "text"), "{}") or "{}")
        guess = str(g.get("ticker", "")).upper().replace(".", "-")
        rows.append({"symbol": d["symbol"], "accession": d["accession"], "year": d["acceptance"][:4],
                     "guess": guess, "confidence": g.get("confidence"), "hit": guess == d["symbol"]})
    _identify_report(pd.DataFrame(rows), ROOT / "reports" / f"identify_{model}.csv", spent)


def claude_collect(batch_id: str, model: str = CLAUDE_MODEL):
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()
    b = client.messages.batches.retrieve(batch_id)
    if b.processing_status != "ended":
        print(f"not done: {b.processing_status} {b.request_counts}")
        return
    rows, fails, spent = [], {}, 0.0
    for res in client.messages.batches.results(batch_id):
        if res.result.type != "succeeded":
            fails[res.result.type] = fails.get(res.result.type, 0) + 1
            continue
        d = json.loads((TEXT_DIR / f"{res.custom_id}.json").read_text())
        spent += _cost(res.result.message.usage, batch=True)
        rows.append(_claude_row(d, res.result.message, model))
    _save(rows, scores_path(model))
    print(f"saved {len(rows)} rows ({sum(r['valid'] for r in rows)} valid), failures {fails}, "
          f"cost ${spent:.2f} -> {scores_path(model)}")


def _save(rows, out):
    if not rows:
        return
    new = pd.DataFrame(rows)
    if out.exists():
        new = pd.concat([pd.read_parquet(out), new], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(out, index=False)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "fetch":
        syms = sorted(set(load_universe()["watchlist_v1"]) | set(sp100_test_universe()))
        fetch(syms)
    elif cmd == "score":
        score(*(sys.argv[2:3]))
    elif cmd == "claude-pilot":
        claude_pilot()
    elif cmd == "claude-submit":
        claude_submit()
    elif cmd == "claude-collect":
        claude_collect(sys.argv[2])
    elif cmd == "claude-identify-submit":
        claude_identify_submit()
    elif cmd == "claude-identify-collect":
        claude_identify_collect(sys.argv[2])
    elif cmd == "identify-local":
        identify_local(*(sys.argv[2:3]))
