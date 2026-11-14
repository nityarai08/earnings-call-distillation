"""The one prompt and output format every setup, the Opus teacher and training all share.

The model writes seven lines. Labels are fixed and each value is picked from its allowed list, so a read is
always well formed, and the probability of every value is known exactly. Scores are p(next EPS up) and
p(positive action)."""
import math

LINES = [("Demand", ["up", "flat", "down"]), ("Pricing", ["up", "flat", "down"]), ("Margins", ["up", "flat", "down"]),
         ("Guidance", ["raised", "held", "cut", "none"]), ("Candor", ["direct", "evasive"]), ("Next EPS", ["up", "down"])]
ACTIONS = {"flat": ["BUY", "AVOID"], "long": ["HOLD", "SELL"]}
STYLES = {
    "plain": "You are an experienced stock analyst. The company's name is masked. Use only what is below.",
    "habit": "You are an experienced stock analyst. The company's name is masked. Use only what is below. Managers sound positive by habit, so weigh specifics: orders, pricing, margins, customers, and questions they avoided.",
    "change": "You are an experienced stock analyst. The company's name is masked. Use only what is below. Ask what changed against last quarter, and whether the market's move around the call matches what was said.",
}
ASK = ("Reply with exactly these seven lines, choosing one value from each list and writing nothing else.\n"
       "Demand: up, flat or down (over the next two quarters)\n"
       "Pricing: up, flat or down (over the next two quarters)\n"
       "Margins: up, flat or down (over the next two quarters)\n"
       "Guidance: raised, held, cut or none (what management did)\n"
       "Candor: direct or evasive (how they answered)\n"
       "Next EPS: up or down (next quarter's EPS against a year earlier)\n"
       "Action: {a} or {b} (for the next 60 sessions)")

def band(dollars):
    return "under $5M" if dollars < 5e6 else "$5M to $20M" if dollars < 20e6 else "over $20M"


def facts_line(r, position):
    move = r.reaction * r.risk * math.sqrt(2)
    eps = "unknown" if r.eps_yoy != r.eps_yoy else f"{'up' if r.eps_yoy > 0 else 'down'} {abs(r.eps_yoy):.2f} on a year ago"
    held = "none" if position == "flat" else "long"
    return f"Facts: sector {r.sector}; daily dollar volume {band(r.dollars)}; past 60-session return {r.past:+.0%}; daily risk {r.risk:.1%}; move around the call {move:+.1%}; EPS {eps}; position {held}."


def messages(r, text, block="", position="flat", style="plain", react=True):
    facts = facts_line(r, position)
    if not react:  # for reading checks: the score must not repeat the reaction it is compared with
        facts = ";".join(x for x in facts.split(";") if "move around" not in x)
    a, b = ACTIONS[position]
    words = f"Management's answers:\n{text}" if text else "Management's answers: not provided."
    return [{"role": "system", "content": STYLES[style]},
            {"role": "user", "content": f"{facts}\n\nKnowledge: {block or 'none.'}\n\n{words}\n\n{ASK.format(a=a, b=b)}"}]


KEEP = {"all": [n for n, _ in LINES], "read": [n for n, _ in LINES[:5]], "action": []}  # lines an arm writes before Action


def rendered(tok, msgs, lines="all"):
    """Token ids up to and including the first label."""
    first = KEEP[lines][0] if KEEP[lines] else "Action"
    return tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False) + f"{first}:")


def options(tok, position, lines="all"):
    """Per line: its name, each value's tokens, and the tokens that lead to the next label."""
    use = [x for x in LINES if x[0] in KEEP[lines]] + [("Action", ACTIONS[position])]
    out = []
    for i, (name, vals) in enumerate(use):
        ids = {v: tok.encode(" " + v) for v in vals}
        assert len({x[0] for x in ids.values()}) == len(vals), name
        after = tok.encode("\n" + use[i + 1][0] + ":") if i + 1 < len(use) else []
        out.append((name, ids, after))
    return out


def answer_text(read, position):
    """The seven lines as text, from a dict of line name to value."""
    return "\n".join(f"{n}: {read[n]}" for n, _ in LINES + [("Action", ACTIONS[position])])
