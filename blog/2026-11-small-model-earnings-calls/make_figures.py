"""Draws the two blog figures as plain SVG from the CSVs in results/. No plotting library needed.

  python3 blog/2026-11-small-model-earnings-calls/make_figures.py
"""
import csv
import re
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

HERE = Path(__file__).parent
BLUE, LIGHT, GREY, INK, MUTED, AXIS = "#2a78d6", "#a3c5ee", "#c4c4bb", "#1f1f1f", "#666666", "#bbbbbb"
KIND = {"Opus": "teacher", "No call": "control", "Known-EPS rule": "rule", "Numbers only": "rule", "Word list": "rule"}
FILL = {"model": BLUE, "control": LIGHT, "rule": GREY, "teacher": "#14365e"}


def text(x, y, s, size=13, fill=INK, anchor="start", weight="normal", halo=False):
    ring = ' stroke="white" stroke-width="4" paint-order="stroke"' if halo else ""  # keeps a label readable over a line
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{ring}>{s}</text>'


def three(v):
    """Three decimals from the CSV's four. A tie goes to the even digit, which matches the unrounded values
    behind the blog tables (0.1035 -> 0.104, 0.1045 -> 0.104). Float formatting would give 0.103 for the first."""
    return str(Decimal(str(v)).quantize(Decimal("0.001"), ROUND_HALF_EVEN))


def svg(h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 {h}" width="760" height="{h}" '
            f'font-family="-apple-system, Helvetica, Arial, sans-serif"><rect width="760" height="{h}" fill="white"/>{"".join(body)}</svg>\n')


def stock_ic():
    """Bar per setup: how well it ranks stocks on the 800 sealed test calls. Opus's IC comes from signals.csv."""
    rows = [r for r in csv.DictReader(open(HERE / "results" / "results.csv")) if r["stock_ic"]]
    opus = next(r for r in csv.DictReader(open(HERE / "results" / "signals.csv")) if (r["check"], r["calls_set"], r["what"]) == ("ic", "test", "Opus"))
    rows.append({"setup": "Opus", "stock_ic": opus["value"]})
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


def vs_opus():
    """Three panels (IC, EPS IC, return after costs): one bar per setup or rule, Opus as a dashed line."""
    res = {r["setup"]: r for r in csv.DictReader(open(HERE / "results" / "results.csv"))}
    opus = {"stock_ic": signal("ic", "test", "Opus")["value"], "eps_ic": signal("eps_ic", "test", "Opus")["value"],
            "book_net": signal("book", "test", "Opus")["value"]}
    panels = [("IC (stock ranking)", "stock_ic", 0, .19, three),
              ("EPS IC", "eps_ic", 0, .32, three),
              ("Return after costs", "book_net", -.01, .026, lambda v: ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v) * 100:.2f}%")]
    left, pw, gap, y0, rh = 120, 176, 22, 132, 26
    end = y0 + len(ORDER) * rh
    body = [text(20, 28, "No setup of the small model reaches Opus on stock ranking or return", 16, weight="600"),
            text(20, 50, "800 test calls, March to August 2026. Dashed line: Opus on the same calls.", 13, MUTED)]
    for x, label, fill in [(20, "small model", BLUE), (125, "small model without the call", LIGHT), (335, "simple rule (no AI)", GREY)]:
        body += [f'<rect x="{x}" y="64" width="12" height="12" rx="2" fill="{fill}"/>', text(x + 18, 75, label, 13, MUTED)]
    for i, name in enumerate(ORDER):
        body.append(text(left - 10, y0 + i * rh + 16, name, anchor="end"))
    for p, (title, col, lo, hi, fmt) in enumerate(panels):
        x0 = left + p * (pw + gap)
        x = lambda v: x0 + (v - lo) / (hi - lo) * pw  # noqa: E731
        o = float(opus[col])
        body += [text(x0, y0 - 30, title, 13, weight="600"),
                 f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{y0 - 4}" y2="{end}" stroke="{AXIS}"/>']
        # Opus's line goes under the bars and labels, so the labels stay readable
        body += [f'<line x1="{x(o):.1f}" x2="{x(o):.1f}" y1="{y0 - 4}" y2="{end}" stroke="{FILL["teacher"]}" stroke-width="2" stroke-dasharray="5 3"/>',
                 text(x(o), y0 - 8, f"Opus {fmt(o)}", 12, FILL["teacher"], "middle", "600")]
        for i, name in enumerate(ORDER):
            v = res.get(name, {}).get(col, "")
            if not v:
                continue
            v, y = float(v), y0 + i * rh
            kind = KIND.get(name, "model")
            a, b = sorted((x(0), x(v)))
            body += [f'<rect x="{a:.1f}" y="{y + 4}" width="{b - a:.1f}" height="{rh - 10}" rx="2" fill="{FILL[kind]}"><title>{name}: {fmt(v)}</title></rect>',
                     text(x(max(v, 0)) + 4, y + 16, fmt(v), 11, MUTED, halo=True)]
    return svg(end + 16, body)


def gap_to_opus():
    """Opus minus each setup or rule, with the 95% range: filled dots for all 800 calls, open dots for July and August."""
    panels = [("IC: Opus minus the setup", "ic_gain", -.2, .35, lambda v: ("+" if v > 0 else "−" if v < 0 else "") + three(abs(v))),
              ("Return after costs: Opus minus the setup", "book_gain", -.045, .065, lambda v: ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v) * 100:.2f}%")]
    left, pw, gap, y0, rh = 120, 290, 30, 172, 34
    end = y0 + len(ORDER) * rh
    body = [text(20, 28, "Opus beats the trained small model, but less clearly after its knowledge date", 16, weight="600"),
            text(20, 50, "Gap between Opus and each setup or rule on the same calls. Right of the dashed line: Opus is better.", 13, MUTED),
            text(20, 68, "Line: 95% range. July and August cover only 5 weeks, so their ranges are rough. An arrow marks a cut range.", 13, MUTED),
            f'<circle cx="26" cy="92" r="5" fill="{BLUE}"/>', text(38, 97, "all 800 calls (March to August)", 13, MUTED),
            f'<circle cx="276" cy="92" r="4.5" fill="white" stroke="{BLUE}" stroke-width="2"/>', text(288, 97, "224 calls in July and August, after Opus's knowledge date", 13, MUTED)]
    for i, name in enumerate(ORDER):
        body.append(text(left - 10, y0 + i * rh + 14, name, anchor="end"))
    for p, (title, check, lo, hi, fmt) in enumerate(panels):
        x0 = left + p * (pw + gap)
        x = lambda v: x0 + (min(max(v, lo), hi) - lo) / (hi - lo) * pw  # noqa: E731
        body += [text(x0, y0 - 34, title, 13, weight="600"),
                 f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{y0 - 14}" y2="{end}" stroke="{MUTED}" stroke-dasharray="4 3"/>',
                 text(x(0), y0 - 18, "no difference", 11, MUTED, "middle")]
        for i, name in enumerate(ORDER):
            for k, (calls_set, dy) in enumerate((("test", 8), ("test_jul_aug", 20))):
                r = signal(check, calls_set, f"Opus minus {name}")
                if r is None:
                    continue
                v, a, b, y = float(r["value"]), float(r["lo"]), float(r["hi"]), y0 + i * rh + dy
                tip = f"Opus minus {name}, {'all 800 calls' if k == 0 else 'July and August'}: {fmt(v)} ({fmt(a)} to {fmt(b)})"
                dot = (f'<circle cx="{x(v):.1f}" cy="{y}" r="5" fill="{BLUE}"><title>{tip}</title></circle>' if k == 0 else
                       f'<circle cx="{x(v):.1f}" cy="{y}" r="4.5" fill="white" stroke="{BLUE}" stroke-width="2"><title>{tip}</title></circle>')
                body += [f'<line x1="{x(a):.1f}" x2="{x(b):.1f}" y1="{y}" y2="{y}" stroke="{BLUE}" stroke-width="{2 if k == 0 else 1.5}" stroke-opacity="{1 if k == 0 else .6}"/>', dot]
                for edge, beyond, s in ((x(hi), b > hi, 1), (x(lo), a < lo, -1)):  # a range wider than the panel: arrow at the cut end
                    if beyond:
                        body.append(f'<path d="M{edge:.1f},{y - 4} L{edge + 6 * s:.1f},{y} L{edge:.1f},{y + 4} Z" fill="{BLUE}" fill-opacity=".6"/>')
                if k == 0:
                    body.append(text(x(b) + 6, y + 4, fmt(v), 11, MUTED, halo=True))
    return svg(end + 16, body)


if __name__ == "__main__":
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "figures" / "stock_ic_by_setup.svg").write_text(stock_ic())
    (HERE / "figures" / "ladder_step_gains.svg").write_text(gaps())
    (HERE / "figures" / "setups_vs_opus.svg").write_text(vs_opus())
    (HERE / "figures" / "gap_to_opus.svg").write_text(gap_to_opus())
    print("wrote", sorted(p.name for p in (HERE / "figures").glob("*.svg")))
