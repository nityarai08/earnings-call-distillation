"""Knowledge blocks: what a setup adds from the knowledge base, under one 1,500-token limit."""
import random

import kb

LIMIT = 1500
EXCERPT = 120


def short(text):
    return " ".join(text.split()[:EXCERPT])


def render(own, similar, facts, notes, tok=None):
    parts = [f"This company, earlier call: {short(x['text'])}" for x in own]
    parts += [f"Similar past call: {short(x['text'])} {x.get('after') or 'Outcome not yet known.'}" for x in similar]
    parts += [f"Company figures: {facts}"] if facts else []
    parts += [f"Analyst note: {x['text']}" for x in notes]
    text = "\n".join(parts)
    if tok is not None:
        ids = tok.encode(text)
        if len(ids) > LIMIT:
            text = tok.decode(ids[:LIMIT])
    return text


def rag(db, r, text, tok=None):
    q = kb.key_terms(db, text)
    date, args = str(r.date.date()), (r.symbol, r.row)
    if not q:
        return render([], [], kb.facts(db, date, r.symbol), [], tok)
    own = kb.search(db, date, *args, q, "own", 2)
    similar = kb.search(db, date, *args, q, "similar", 3)
    notes = kb.search(db, date, *args, q, "handbook", 1)
    return render(own, similar, kb.facts(db, date, r.symbol), notes, tok)


def random_docs(db, r, text, tok=None):
    """Same shape and length as rag, chosen at random from what was available: tests a longer prompt alone."""
    date = str(r.date.date())
    rng = random.Random(int(r.row))
    top = db.execute("select max(id) from passages where kind = 'call' and available < ?", [date]).fetchone()[0] or 1
    got = []
    while len(got) < 5:
        x = kb.read(db, rng.randint(1, top), date, r.symbol, r.row)
        if x and x["kind"] == "call":
            got.append(x)
    return render(got[:2], got[2:], kb.facts(db, date, r.symbol), [], tok)


def shuffled(db, r, text, tok=None):
    """rag's documents with what happened next swapped at random: tests whether outcomes carry the value."""
    q = kb.key_terms(db, text)
    date, args = str(r.date.date()), (r.symbol, r.row)
    own = kb.search(db, date, *args, q, "own", 2) if q else []
    similar = kb.search(db, date, *args, q, "similar", 3) if q else []
    notes = kb.search(db, date, *args, q, "handbook", 1) if q else []
    rng = random.Random(int(r.row))
    for x in similar:
        if x.get("after"):
            x["after"] = rng.choice(["Then beat its sector.", "Then lagged its sector."])
    return render(own, similar, kb.facts(db, date, r.symbol), notes, tok)


def saved(path):
    """A block chosen earlier, such as by pi, read from its log by row."""
    import json
    got = {x["row"]: x["block"] for x in map(json.loads, open(path))}
    return lambda db, r, text, tok=None: got.get(int(r.row), "")


KINDS = {"none": lambda *a, **k: "", "rag": rag, "random": random_docs, "shuffled": shuffled}
