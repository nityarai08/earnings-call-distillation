"""Builds data/paper/kb.sqlite. Resumes by month: a month's passages commit together with its mark.

  PYTHONPATH=src:scripts nice -n 10 .venv/bin/python -W ignore scripts/paper/build_kb.py build
  PYTHONPATH=src:scripts nice -n 10 .venv/bin/python -W ignore scripts/paper/build_kb.py handbook
  PYTHONPATH=src:scripts nice -n 10 .venv/bin/python -W ignore scripts/paper/build_kb.py audit

Masking as in small_call_read.py: company name, ticker and years. Passages of about 220 words, up to 3,000
words a call. Cards say how a past call turned out and become available when the outcome closed. Test-S
calls get no card, so sealed outcomes stay out of every prompt.
"""
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
import kb  # noqa: E402
from common import DATA, ROOT, Status  # noqa: E402
from small_call_read import names  # noqa: E402
from tagent.earnings_text import mask  # noqa: E402

YEARS = re.compile(r"\b(19|20)\d\d\b|[‘’']\d\d\b")
OLD_TEXT = ROOT / "data" / "features" / "small_call_text.parquet"
NEW_TEXT = DATA / "text_new.parquet"
SRC = ROOT / "data" / "raw" / "defeatbeta"
PIECE, MOST = 220, 3000


def titles():
    return {x["ticker"]: x["title"] for x in json.loads((SRC / "company_tickers.json").read_text()).values()}


def clean(text, symbol, title):
    return YEARS.sub("[year]", mask(text, names(title), [symbol]))


def schema(db):
    db.executescript("""
      create table if not exists passages (id integer primary key, kind text, symbol text, call_row integer, available text, text text);
      create index if not exists p_call on passages(call_row);
      create virtual table if not exists fts using fts5(text, content='passages', content_rowid='id');
      create table if not exists cards (call_row integer primary key, available text, result text);
      create table if not exists facts (symbol text, quarter text, available text, eps real, revenue real, gross real);
      create index if not exists f_sym on facts(symbol, available);
      create table if not exists marks (name text primary key);
    """)


def answers(rows):
    """Management's answers for these rows, from the two text files."""
    keep, got = set(rows), {}
    for f in (OLD_TEXT, NEW_TEXT):
        for b in pq.ParquetFile(f).iter_batches(batch_size=500, columns=["row", "answers"]):
            for n, x in zip(b["row"].to_pylist(), b["answers"].to_pylist()):
                if n in keep:
                    got[n] = x
    return got


def passages(db, calls):
    name = titles()
    seen = {x[0] for x in db.execute("select name from marks")}
    status = Status("build", "kb", calls.month.nunique())
    text = None
    for i, (m, g) in enumerate(calls.groupby("month")):
        if f"calls:{m}" in seen:
            continue
        text = text or answers(calls.row.tolist())
        for r in g.itertuples():
            words = clean(" ".join(text[r.row].split()[:MOST]), r.symbol, name.get(r.symbol, "")).split()
            for s in range(0, len(words), PIECE):
                piece = " ".join(words[s:s + PIECE])
                if len(piece) > 200:
                    cur = db.execute("insert into passages (kind, symbol, call_row, available, text) values ('call', ?, ?, ?, ?)", [r.symbol, r.row, str(r.date.date()), piece])
                    db.execute("insert into fts (rowid, text) values (?, ?)", [cur.lastrowid, piece])
        db.execute("insert into marks values (?)", [f"calls:{m}"])
        db.commit()
        status.rows(i + 1)
        print(m, len(g), flush=True)


def cards(db):
    o = pd.read_parquet(DATA / "kb_outcomes.parquet").dropna(subset=["excess"])
    o["result"] = o.excess.gt(0).map({True: "Then beat its sector", False: "Then lagged its sector"}) + o.next_yoy.map(lambda x: "" if pd.isna(x) else ", and next-quarter EPS rose on a year ago" if x > 0 else ", and next-quarter EPS fell on a year ago") + "."
    db.execute("delete from cards")
    db.executemany("insert into cards values (?, ?, ?)", [(int(r.row), str(r.exit.date()), r.result) for r in o.itertuples()])
    db.commit()
    print(len(o), "cards", flush=True)


def fact_table(db, calls):
    """Quarterly EPS, revenue and gross profit. Available from the first call after the quarter ended, else 60 days on."""
    want = list(calls.symbol.unique())
    e = pd.read_parquet(SRC / "stock_tailing_eps.parquet", columns=["symbol", "report_date", "eps"])
    e = e[e.symbol.isin(want)].assign(eps=lambda x: pd.to_numeric(x.eps, errors="coerce")).drop_duplicates(["symbol", "report_date"])
    s = pq.read_table(SRC / "stock_statement.parquet", columns=["symbol", "report_date", "item_name", "item_value"],
                      filters=[("period_type", "=", "quarterly"), ("item_name", "in", ["total_revenue", "gross_profit"]), ("symbol", "in", want)]).to_pandas()
    s["item_value"] = pd.to_numeric(s.item_value, errors="coerce")
    s = s.drop_duplicates(["symbol", "report_date", "item_name"]).pivot(index=["symbol", "report_date"], columns="item_name", values="item_value").reset_index()
    f = e.merge(s, on=["symbol", "report_date"], how="outer").assign(end=lambda x: pd.to_datetime(x.report_date, errors="coerce"))  # some rows are TTM
    f = f.dropna(subset=["end"]).sort_values("end")
    c = calls[["symbol", "date"]].assign(date=calls.date.astype("datetime64[ns]"), seen=calls.date.astype("datetime64[ns]")).sort_values("date")
    f = pd.merge_asof(f.assign(after=(f.end + pd.Timedelta(days=5)).astype("datetime64[ns]")), c, left_on="after", right_on="date", by="symbol", direction="forward", tolerance=pd.Timedelta(days=100))
    f["available"] = f.seen.fillna(f.end + pd.Timedelta(days=60)).dt.strftime("%Y-%m-%d")
    db.execute("delete from facts")
    db.executemany("insert into facts values (?, ?, ?, ?, ?, ?)", [(r.symbol, r.report_date, r.available, None if pd.isna(r.eps) else r.eps, None if pd.isna(r.total_revenue) else r.total_revenue, None if pd.isna(r.gross_profit) else r.gross_profit) for r in f.itertuples()])
    db.commit()
    print(len(f), "fact rows,", f.total_revenue.notna().mean().round(3), "with revenue", flush=True)


def add_handbook():
    db = sqlite3.connect(kb.DB)
    if db.execute("select count(*) from passages where kind = 'handbook'").fetchone()[0]:
        return
    for n in json.loads((DATA / "handbook.json").read_text()):
        cur = db.execute("insert into passages (kind, symbol, call_row, available, text) values ('handbook', '', -1, '1900-01-01', ?)", [n])
        db.execute("insert into fts (rowid, text) values (?, ?)", [cur.lastrowid, n])
    db.commit()
    vocab(db)
    print(db.execute("select kind, count(*) from passages group by kind").fetchall(), flush=True)


def vocab(db):
    db.executescript("""drop table if exists vt; create virtual table vt using fts5vocab('fts', 'row');
      drop table if exists vocab; create table vocab as select term, doc from vt; create unique index v_term on vocab(term); drop table vt;""")
    db.commit()


def build():
    calls = pd.read_parquet(DATA / "calls.parquet").sort_values("date")
    db = sqlite3.connect(kb.DB)
    db.execute("pragma journal_mode=wal")
    schema(db)
    passages(db, calls)
    cards(db)
    fact_table(db, calls)
    vocab(db)
    print(db.execute("select kind, count(*) from passages group by kind").fetchall(), flush=True)


def audit():
    """1,000 random retrievals: every result must predate the decision, and no call sees itself."""
    s = pd.read_parquet(DATA / "sets.parquet")
    db = kb.connect()
    rng = random.Random(0)
    bad = n = named = 0
    name = titles()
    for r in s.sample(1000, random_state=0).itertuples():
        date = str(r.date.date())
        for scope in ("own", "similar"):
            for x in kb.search(db, date, r.symbol, r.row, '"demand" OR "margin" OR "pricing" OR "orders"', scope, 5):
                n += 1
                when, sym = db.execute("select available, symbol from passages where id = ?", [x["id"]]).fetchone()
                bad += when >= date or x["call"] == r.row or (scope == "own") != (sym == r.symbol)
                if x["after"]:
                    bad += db.execute("select available from cards where call_row = ?", [x["call"]]).fetchone()[0] >= date
                named += bool(re.search(rf"\b{re.escape(sym)}\b", x["text"]))
    test = set(s[s.set.str.startswith("test")].row)
    leaked = db.execute(f"select count(*) from cards where call_row in ({','.join(map(str, test))})").fetchone()[0]
    print(n, "results checked,", bad, "broke a date rule,", leaked, "cards from test calls,", named, "still show a ticker", flush=True)
    assert not bad and not leaked


if __name__ == "__main__":
    {"build": build, "audit": audit, "handbook": add_handbook}[sys.argv[1]]()
