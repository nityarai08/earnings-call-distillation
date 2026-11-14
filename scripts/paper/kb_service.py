"""Local search service for pi: the same kb.py functions RAG uses, for one case at a time. The runner sets
the case; the tools can only pass a query or a document id. It holds no state worth keeping."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import blocks
import kb

PORT, MOST_CALLS = 8098, 4


class Case:
    def __init__(self, r, forced=None):
        self.r, self.date, self.forced = r, str(r.date.date()), forced
        self.calls, self.tokens, self.queries = 0, 0, []
        self.own, self.similar, self.notes, self.facts = [], [], [], ""


class Service:
    def __init__(self):
        self.case = None
        svc = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                text = svc.answer(self.path.strip("/"), body)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(text.encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def answer(self, tool, b):
        c = self.case
        if c is None:
            return "No case is open."
        if tool == "search" and str(b.get("query", "")) in c.queries:
            return "You already ran that search. Read a document from it or try different words."
        c.calls += 1
        if c.calls > MOST_CALLS:
            return "Tool limit reached. Stop and reply DONE."
        db = kb.connect()
        r = c.r
        if tool == "search":
            q = kb.fts_query(str(b.get("query", "")))
            c.queries.append(str(b.get("query", "")))
            scope = b.get("scope") if b.get("scope") in ("own", "similar", "handbook") else "similar"
            if c.forced is not None:
                hits = c.forced
            else:
                hits = kb.search(db, c.date, r.symbol, r.row, q, scope, 5) if q else []
            return "\n".join(f"id {x['id']}: {' '.join(x['text'].split()[:15])}" for x in hits) or "Nothing found."
        if tool == "read":
            if str(b.get("id")).strip() == "facts":
                c.facts = kb.facts(db, c.date, r.symbol)
                c.tokens += 60
                return c.facts
            if c.tokens >= blocks.LIMIT:
                return "Reading limit reached. Stop and reply DONE."
            try:
                x = kb.read(db, int(b.get("id")), c.date, r.symbol, r.row)
            except (TypeError, ValueError):
                x = None
            if x is None:
                return "No such document."
            c.tokens += int(len(blocks.short(x["text"]).split()) * 1.4)
            if x["kind"] == "handbook":
                c.notes.append(x)
            elif db.execute("select symbol from passages where id = ?", [x["id"]]).fetchone()[0] == r.symbol:
                c.own.append(x)
            else:
                c.similar.append(x)
            return blocks.short(x["text"]) + " " + (x.get("after") or "")
        return "Unknown tool."
