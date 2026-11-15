"""Draws the blog figures as plain SVG from the CSVs in results/. No plotting library needed.

  python3 blog/2026-11-small-model-earnings-calls/make_figures.py
"""
import csv
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).parent
BLUE, LIGHT, GREY, INK, MUTED, AXIS = "#2a78d6", "#a3c5ee", "#c4c4bb", "#1f1f1f", "#666666", "#bbbbbb"
KIND = {"Opus": "teacher", "No call": "control", "Known-EPS rule": "rule", "Numbers only": "rule", "Word list": "rule"}
FILL = {"model": BLUE, "control": LIGHT, "rule": GREY, "teacher": "#14365e"}


def text(x, y, s, size=13, fill=INK, anchor="start", weight="normal", halo=False):
    ring = ' stroke="white" stroke-width="4" paint-order="stroke"' if halo else ""  # keeps a label readable over a line
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{ring}>{s}</text>'


def rnd(v, places, scale=1):
    """Half-up rounding of the CSV value as written, so 0.1035 gives 0.104. Float formatting would give 0.103."""
    return (Decimal(str(v)) * scale).quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP)


def three(v):
    return str(rnd(v, 3))


def svg(h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 {h}" width="760" height="{h}" '
            f'font-family="-apple-system, Helvetica, Arial, sans-serif"><rect width="760" height="{h}" fill="white"/>{"".join(body)}</svg>\n')


def stock_ic():
    """Bar per setup: how well it ranks stocks on the 800 sealed test calls, from signals.csv."""
    rows = [{"setup": r["what"], "stock_ic": r["value"]} for r in csv.DictReader(open(HERE / "results" / "signals.csv"))
            if (r["check"], r["calls_set"]) == ("ic", "test")]
    rows.sort(key=lambda r: -float(r["stock_ic"]))
    left, w, y0, rh, top_ic = 170, 470, 104, 32, 0.19
    body = [text(20, 28, "Opus ranks best. The small model's setups are all close to 0.10", 16, weight="600"),
            text(20, 50, "IC of Opus, each setup and each simple rule on the 800 test calls, March to August 2026. 0 means no relation.", 13, MUTED)]
    for x, label, kind in [(20, "Opus (teacher)", "teacher"), (145, "small model", "model"), (250, "small model without the call", "control"), (460, "simple rule (no AI)", "rule")]:
        body += [f'<rect x="{x}" y="66" width="12" height="12" rx="2" fill="{FILL[kind]}"/>', text(x + 18, 77, label, 13, MUTED)]
    for i, r in enumerate(rows):
        ic, y = float(r["stock_ic"]), y0 + i * rh
        bar = ic / top_ic * w
        body += [text(left - 10, y + 15, r["setup"], anchor="end"),
                 f'<rect x="{left}" y="{y + 3}" width="{bar:.1f}" height="{rh - 12}" rx="3" fill="{FILL[KIND.get(r["setup"], "model")]}"/>',
                 text(left + bar + 6, y + 17, three(ic))]
    body.append(f'<line x1="{left}" x2="{left}" y1="{y0 - 6}" y2="{y0 + len(rows) * rh}" stroke="{AXIS}"/>')
    return svg(y0 + len(rows) * rh + 20, body)


def gaps():
    """Dot and 95% range for the gain of each ladder step, from the paired bootstrap in findings.csv."""
    order = {"kb": 0, "rag_pi": 1, "distill": 2, "reward": 3}
    rows = sorted(csv.DictReader(open(HERE / "results" / "findings.csv")), key=lambda r: order[r["question"]])
    left, w, y0, rh, span = 150, 400, 96, 46, 0.1
    x = lambda v: left + (v + span) / (2 * span) * w  # noqa: E731
    sign = lambda v: ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):.3f}"  # noqa: E731
    body = [text(20, 28, "No step gives a gain larger than chance can explain", 16, weight="600"),
            text(20, 50, "IC gain of each step on the same 800 test calls. Dot: the gain. Line: the 95% range.", 13, MUTED),
            f'<line x1="{x(0)}" x2="{x(0)}" y1="{y0 - 18}" y2="{y0 + 4 * rh}" stroke="{MUTED}" stroke-dasharray="4 3"/>',
            text(x(0), y0 - 24, "no difference", 12, MUTED, "middle")]
    for i, r in enumerate(rows):
        # "Pi minus RAG: IC +0.007 (95% -0.051 to +0.073)" -> step name and three numbers
        name, b = re.match(r"(.+?) minus (.+?):", r["result"]).groups()
        gap, lo, hi = map(float, re.findall(r"[-+]\d\.\d+", r["result"]))
        y = y0 + i * rh
        body += [text(left - 14, y + 5, f"{name} vs {b}", anchor="end"),
                 f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{y}" y2="{y}" stroke="{BLUE}" stroke-width="2"/>',
                 f'<circle cx="{x(gap):.1f}" cy="{y}" r="5" fill="{BLUE}"/>',
                 text(left + w + 24, y + 5, f"{sign(gap)}  ({sign(lo)} to {sign(hi)})")]
    for t in (-0.1, -0.05, 0, 0.05, 0.1):
        body.append(text(x(t), y0 + 4 * rh + 22, "0" if t == 0 else sign(t)[:-1].rstrip("0"), 12, MUTED, "middle"))
    return svg(y0 + 4 * rh + 40, body)


ORDER = ["No call", "Call only", "RAG", "Pi", "SFT", "RL", "Known-EPS rule", "Word list", "Numbers only"]


def signal(check, calls_set, what):
    for r in csv.DictReader(open(HERE / "results" / "signals.csv")):
        if (r["check"], r["calls_set"], r["what"]) == (check, calls_set, what):
            return r
    return None


def sign(v, body):
    return ("+" if v > 0 else "−" if v < 0 else "") + body(abs(v))


PCT = lambda v: sign(v, lambda a: f"{rnd(a, 2, 100)}%")  # noqa: E731
SHARE = lambda v: f"{rnd(v, 1, 100)}%"  # noqa: E731
IC = lambda v: three(v) if v >= 0 else "−" + three(-v)  # noqa: E731
GAIN = lambda v: sign(v, three)  # noqa: E731


def legend(y, items):
    """items: (x, label, fill, opacity or 'dot', 'ring')"""
    out = []
    for x, label, fill, mark in items:
        if mark == "dot":
            out.append(f'<circle cx="{x + 6}" cy="{y - 4}" r="5" fill="{fill}"/>')
        elif mark == "ring":
            out.append(f'<circle cx="{x + 6}" cy="{y - 4}" r="4.5" fill="white" stroke="{fill}" stroke-width="2"/>')
        else:
            out.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="2" fill="{fill}"/>')
        out.append(text(x + 18, y + 1, label, 13, MUTED))
    return out


def measures(calls_set):
    """Every measure of every setup, rule and Opus on one call set, from signals.csv (and results.csv for agreement)."""
    rows = list(csv.DictReader(open(HERE / "results" / "signals.csv")))
    agree = {r["setup"]: r["read_match"] for r in csv.DictReader(open(HERE / "results" / "results.csv"))}
    out = {}
    for name in ["Opus", *ORDER, "Buy every call"]:
        m = {}
        for r in rows:
            if r["calls_set"] == calls_set and r["what"] == name:
                if r["check"] in ("ic", "eps_ic", "eps_acc"):
                    m[r["check"]] = float(r["value"])
                elif r["check"] == "book":
                    m["book"], m["buys"] = float(r["value"]), int(r["calls"])
        if calls_set == "test" and agree.get(name):
            m["agree"] = float(agree[name])
        out[name] = m
    return out


def bar_grid(title, subtitle, data, panels, cols=3):
    """Small multiples: one panel a measure, one bar a setup or rule, Opus as a dashed line. Bars start at zero."""
    left, pw, gap, rh = 120, 176, 22, 24
    block = 44 + len(ORDER) * rh + 26
    body = [text(20, 28, title, 16, weight="600"), text(20, 50, subtitle, 13, MUTED),
            *legend(78, [(20, "small model", BLUE, ""), (125, "small model without the call", LIGHT, ""), (335, "simple rule (no AI)", GREY, ""),
                         (500, "Opus", FILL["teacher"], "")])]
    body[-2] = f'<line x1="500" x2="512" y1="74" y2="74" stroke="{FILL["teacher"]}" stroke-width="2" stroke-dasharray="4 2"/>'
    for g in range((len(panels) + cols - 1) // cols):
        y0 = 136 + g * block
        for i, name in enumerate(ORDER):
            body.append(text(left - 10, y0 + i * rh + 15, name, anchor="end"))
        for p, (ptitle, key, lo, hi, fmt) in enumerate(panels[g * cols:(g + 1) * cols]):
            x0 = left + p * (pw + gap)
            x = lambda v: x0 + (v - lo) / (hi - lo) * pw  # noqa: E731
            end = y0 + len(ORDER) * rh
            body += [text(x0, y0 - 30, ptitle, 13, weight="600"),
                     f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{y0 - 4}" y2="{end}" stroke="{AXIS}"/>']
            o = data["Opus"].get(key)
            if o is not None:  # under the bars and labels, so the labels stay readable
                body += [f'<line x1="{x(o):.1f}" x2="{x(o):.1f}" y1="{y0 - 4}" y2="{end}" stroke="{FILL["teacher"]}" stroke-width="2" stroke-dasharray="5 3"/>',
                         text(x(o), y0 - 8, f"Opus {fmt(o)}", 12, FILL["teacher"], "middle", "600")]
            for i, name in enumerate(ORDER):
                v = data[name].get(key)
                if v is None:
                    continue
                y = y0 + i * rh
                a, b = sorted((x(0), x(v)))
                body += [f'<rect x="{a:.1f}" y="{y + 4}" width="{b - a:.1f}" height="{rh - 9}" rx="2" fill="{FILL[KIND.get(name, "model")]}"><title>{name}: {fmt(v)}</title></rect>',
                         text(x(max(v, 0)) + 4, y + 15, fmt(v), 11, MUTED, halo=True)]
    return svg(136 + ((len(panels) + cols - 1) // cols) * block - 30, body)


def results_all():
    """The results table as a chart: six measures on all 800 test calls."""
    return bar_grid("Opus leads on stock ranking and return. The trained models come close on EPS",
                    "All six measures on the 800 test calls, March to August 2026. Dashed line: Opus on the same calls.",
                    measures("test"),
                    [("IC (stock ranking)", "ic", 0, .19, IC), ("EPS IC", "eps_ic", 0, .32, IC), ("Return after costs", "book", -.01, .026, PCT),
                     ("EPS accuracy", "eps_acc", 0, .8, SHARE), ("Agreement with Opus", "agree", 0, 1, SHARE),
                     ("BUY calls (of 800)", "buys", 0, 800, lambda v: f"{v:.0f}")])


def results_jul_aug():
    """The same measures on the 224 calls after Opus's knowledge date."""
    return bar_grid("After Opus's knowledge date, several setups rank as well as Opus",
                    "The 224 test calls from July and August 2026. Dashed line: Opus on the same calls. Only 5 weeks, so the values are noisy.",
                    measures("test_jul_aug"),
                    [("IC (stock ranking)", "ic", -.08, .22, IC), ("EPS IC", "eps_ic", 0, .34, IC), ("Return after costs", "book", -.02, .04, PCT),
                     ("EPS accuracy", "eps_acc", 0, .8, SHARE), ("BUY calls (of 224)", "buys", 0, 230, lambda v: f"{v:.0f}")])


def dot_rows(y0, rh, left, pw, x0, lo, hi, rows, fmt):
    """Dots with 95% ranges. rows: (y offset, value, low, high, colour, filled, tip). Arrows mark cut ranges."""
    x = lambda v: x0 + (min(max(v, lo), hi) - lo) / (hi - lo) * pw  # noqa: E731
    out = []
    for dy, v, a, b, colour, filled, tip in rows:
        y = y0 + dy
        out.append(f'<line x1="{x(a):.1f}" x2="{x(b):.1f}" y1="{y}" y2="{y}" stroke="{colour}" stroke-width="{2 if filled else 1.5}" stroke-opacity="{1 if filled else .6}"/>')
        for edge, beyond, s in ((x(hi), b > hi, 1), (x(lo), a < lo, -1)):
            if beyond:
                out.append(f'<path d="M{edge:.1f},{y - 4} L{edge + 6 * s:.1f},{y} L{edge:.1f},{y + 4} Z" fill="{colour}" fill-opacity=".6"/>')
        out.append(f'<circle cx="{x(v):.1f}" cy="{y}" r="5" fill="{colour}"><title>{tip}</title></circle>' if filled else
                   f'<circle cx="{x(v):.1f}" cy="{y}" r="4.5" fill="white" stroke="{colour}" stroke-width="2"><title>{tip}</title></circle>')
    return out, x


def gap_to_opus():
    """Opus minus each setup or rule, with the 95% range: filled dots for all 800 calls, open dots for July and August."""
    panels = [("IC: Opus minus the setup", "ic_gain", -.2, .35, GAIN), ("EPS IC: Opus minus the setup", "eps_gain", -.12, .45, GAIN),
              ("Return: Opus minus the setup", "book_gain", -.045, .065, PCT)]
    left, pw, gap, y0, rh = 120, 190, 14, 182, 40
    end = y0 + len(ORDER) * rh
    body = [text(20, 28, "Opus beats the trained small model, but less clearly after its knowledge date", 16, weight="600"),
            text(20, 50, "Gap between Opus and each setup or rule on the same calls. Right of the dashed line: Opus is better.", 13, MUTED),
            text(20, 68, "Line: 95% range. July and August cover only 5 weeks, so their ranges are rough. An arrow marks a cut range.", 13, MUTED),
            *legend(98, [(20, "all 800 calls (March to August)", BLUE, "dot"), (270, "224 calls in July and August, after Opus's knowledge date", BLUE, "ring")])]
    for i, name in enumerate(ORDER):
        body.append(text(left - 10, y0 + i * rh + 18, name, anchor="end"))
    for p, (title, check, lo, hi, fmt) in enumerate(panels):
        x0 = left + p * (pw + gap)
        rows = []
        for i, name in enumerate(ORDER):
            for k, calls_set in enumerate(("test", "test_jul_aug")):
                r = signal(check, calls_set, f"Opus minus {name}")
                if r is not None:
                    v, a, b = float(r["value"]), float(r["lo"]), float(r["hi"])
                    rows.append((i * rh + (14 if k == 0 else 28), v, a, b, BLUE, k == 0,
                                 f"Opus minus {name}, {'all 800 calls' if k == 0 else 'July and August'}: {fmt(v)} ({fmt(a)} to {fmt(b)})"))
        marks, x = dot_rows(y0, rh, left, pw, x0, lo, hi, rows, fmt)
        body += [text(x0, y0 - 34, title, 13, weight="600"),
                 f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{y0 - 14}" y2="{end}" stroke="{MUTED}" stroke-dasharray="4 3"/>',
                 text(x(0), y0 - 18, "no difference", 11, MUTED, "middle"), *marks]
        body += [text(x(v), y0 + dy - 8, fmt(v), 11, MUTED, "middle", halo=True) for dy, v, _, _, _, filled, _ in rows if filled]
    return svg(end + 16, body)


ORANGE = "#eb6834"
# (label, group, check, test what, validation what, status)
SIGNALS = [("EPS IC of RL", "ic", "eps_ic", "RL", "RL", "held"),
           ("EPS IC of No call (facts line only)", "ic", "eps_ic", "No call", "No call", "held"),
           ("Stock IC of Call only", "ic", "ic", "Call only", "Call only", "did not repeat"),
           ("Stock IC of RL", "ic", "ic", "RL", "RL", "did not repeat"),
           ("EPS IC gain: SFT minus RAG", "ic", "eps_gain", "SFT minus RAG", "SFT minus RAG", "did not repeat"),
           ("IC gain: RL + rules minus rules", "ic", "mix", "RL + both rules minus both rules", "RL + both rules minus both rules", "lead"),
           ("Return of RL", "ret", "book", "RL", "RL", "not a signal"),
           ("Return gain: RL minus SFT", "ret", "book_gain", "RL minus SFT", "RL minus SFT", "did not repeat"),
           ("Return of the word list", "ret", "book", "Word list", "Word list", "not a signal")]


def signals_two_periods():
    """Each possible signal on the 2026 test calls and again on the 2025 validation calls."""
    left, pw, rh = 250, 380, 36
    groups = [("IC measures", "ic", -.15, .42, lambda v: sign(v, three)), ("Return after costs", "ret", -.07, .045, PCT)]
    body = [text(20, 28, "Only the EPS forecast held up in both periods", 16, weight="600"),
            text(20, 50, "Each possible signal on the 2026 test calls and again on the 2025 validation calls. Line: 95% range.", 13, MUTED),
            *legend(80, [(20, "test calls, March to August 2026", BLUE, "dot"), (280, "validation calls, 2025", ORANGE, "dot")])]
    y0 = 140
    for gtitle, g, lo, hi, fmt in groups:
        sig = [s for s in SIGNALS if s[1] == g]
        end = y0 + len(sig) * rh
        rows = []
        for i, (label, _, check, wt, wv, status) in enumerate(sig):
            body += [text(left - 10, y0 + i * rh + 18, label, anchor="end"), text(left + pw + 14, y0 + i * rh + 18, status, 12, MUTED)]
            for k, (calls_set, what, colour) in enumerate((("test", wt, BLUE), ("validation", wv, ORANGE))):
                r = signal(check, calls_set, what)
                v, a, b = float(r["value"]), float(r["lo"]), float(r["hi"])
                rows.append((i * rh + (12 if k == 0 else 24), v, a, b, colour, True, f"{label}, {calls_set}: {fmt(v)} ({fmt(a)} to {fmt(b)})"))
        marks, x = dot_rows(y0, rh, left, pw, left, lo, hi, rows, fmt)
        body += [text(left, y0 - 30, gtitle, 13, weight="600"), text(left + pw + 14, y0 - 30, "status", 13, weight="600"),
                 f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{y0 - 14}" y2="{end}" stroke="{MUTED}" stroke-dasharray="4 3"/>',
                 text(x(0), y0 - 16, "zero", 11, MUTED, "middle"), *marks]
        y0 = end + 70
    return svg(y0 - 50, body)


if __name__ == "__main__":
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "figures" / "stock_ic_by_setup.svg").write_text(stock_ic())
    (HERE / "figures" / "ladder_step_gains.svg").write_text(gaps())
    (HERE / "figures" / "results_all_measures.svg").write_text(results_all())
    (HERE / "figures" / "results_jul_aug.svg").write_text(results_jul_aug())
    (HERE / "figures" / "gap_to_opus.svg").write_text(gap_to_opus())
    (HERE / "figures" / "signals_two_periods.svg").write_text(signals_two_periods())
    print("wrote", sorted(p.name for p in (HERE / "figures").glob("*.svg")))
