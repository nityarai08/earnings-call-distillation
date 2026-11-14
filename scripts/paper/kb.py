"""The shared knowledge base: one SQLite file with full-text search. RAG, pi and the Opus teacher all read
it through these functions, so a date rule enforced here holds for all of them.

Every passage has an `available` date. A search answers only for a decision date and returns only what was
available strictly before it. Text is masked, and the call being decided is never in its own results."""
import re
import sqlite3

from common import DATA

DB = DATA / "kb.sqlite"
WORD = re.compile(r"[a-z]{4,}")
SKIP = set("that this with have from they were been will would about their there which what when your more also just very than some think going know like into over because really quarter year company".split())


def connect(path=DB):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def fts_query(text, most=12):
    words = list(dict.fromkeys(w for w in WORD.findall(text.lower()) if w not in SKIP))[:most]
    return " OR ".join(f'"{w}"' for w in words)


def key_terms(db, text, most=12):
    """The call's own rarest frequent words, as the query explicit RAG sends."""
    count = {}
    for w in WORD.findall(text.lower()):
        if w not in SKIP:
            count[w] = count.get(w, 0) + 1
    words = [w for w, n in sorted(count.items(), key=lambda x: -x[1])[:80]]
    if not words:
        return ""
    marks = ",".join("?" * len(words))
    df = dict(db.execute(f"select term, doc from vocab where term in ({marks})", words).fetchall())
    best = sorted(words, key=lambda w: count[w] / (1 + df.get(w, 0)), reverse=True)[:most]
    return " OR ".join(f'"{w}"' for w in best)


def search(db, date, symbol, row, query, scope="similar", k=5):
    """scope: own (this company's past calls), similar (other companies), handbook. Returns dicts, best first."""
    if scope == "handbook":
        sql, args = "select p.id, p.kind, p.text, null, null from fts join passages p on p.id = fts.rowid where fts match ? and p.kind = 'handbook' order by rank limit ?", [query, k]
    else:
        same = "p.symbol = ?" if scope == "own" else "p.symbol != ?"
        sql = (f"select p.id, p.kind, p.text, p.call_row, o.result from fts join passages p on p.id = fts.rowid "
               f"left join cards o on o.call_row = p.call_row and o.available < ? "
               f"where fts match ? and p.kind = 'call' and p.available < ? and p.call_row != ? and {same} order by rank limit ?")
        args = [date, query, date, row, symbol, k]
    return [{"id": a, "kind": b, "text": c, "call": d, "after": e} for a, b, c, d, e in db.execute(sql, args).fetchall()]


def read(db, passage, date, symbol, row):
    """One passage by id, under the same rules as search."""
    r = db.execute("select p.id, p.kind, p.text, p.call_row, o.result from passages p left join cards o on o.call_row = p.call_row and o.available < ? "
                   "where p.id = ? and (p.kind = 'handbook' or (p.available < ? and p.call_row != ?))", [date, passage, date, row]).fetchone()
    return None if r is None else {"id": r[0], "kind": r[1], "text": r[2], "call": r[3], "after": r[4]}


def facts(db, date, symbol, last=4):
    """The company's last reported quarters, newest first, as of the decision date. Words, no dates."""
    rows = db.execute("select eps, revenue, gross from facts where symbol = ? and available < ? order by quarter desc limit ?", [symbol, date, last]).fetchall()
    if not rows:
        return "No quarterly figures on file."
    label = ["Last quarter", "One quarter before", "Two quarters before", "Three quarters before"]
    out = []
    for name, (eps, rev, gross) in zip(label, rows):
        bits = [f"EPS {eps:.2f}" if eps is not None else "", f"revenue ${rev / 1e6:,.1f}M" if rev else "", f"gross margin {gross / rev:.0%}" if rev and gross is not None else ""]
        out.append(f"{name}: " + ", ".join(b for b in bits if b))
    return "; ".join(out) + "."
