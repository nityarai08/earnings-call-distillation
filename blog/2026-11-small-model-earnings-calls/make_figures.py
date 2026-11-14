"""Draws the two blog figures as plain SVG from the CSVs in results/. No plotting library needed.

  python3 blog/2026-11-small-model-earnings-calls/make_figures.py
"""
import csv
import re
from pathlib import Path

HERE = Path(__file__).parent
BLUE, LIGHT, GREY, INK, MUTED, AXIS = "#2a78d6", "#a3c5ee", "#c4c4bb", "#1f1f1f", "#666666", "#bbbbbb"
KIND = {"No call": "control", "Known-EPS rule": "rule", "Numbers only": "rule", "Word list": "rule"}
FILL = {"model": BLUE, "control": LIGHT, "rule": GREY}


def text(x, y, s, size=13, fill=INK, anchor="start", weight="normal"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'


def svg(h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 {h}" width="760" height="{h}" '
            f'font-family="-apple-system, Helvetica, Arial, sans-serif"><rect width="760" height="{h}" fill="white"/>{"".join(body)}</svg>\n')


def stock_ic():
    """Bar per setup: how well it ranks stocks on the 800 sealed test calls."""
    rows = [r for r in csv.DictReader(open(HERE / "results" / "results.csv")) if r["stock_ic"]]
    rows.sort(key=lambda r: -float(r["stock_ic"]))
    left, w, y0, rh, top_ic = 170, 470, 104, 32, 0.13
    body = [text(20, 28, "Every setup that reads the call has an IC of approximately 0.10", 16, weight="600"),
            text(20, 50, "IC of each setup and simple rule on the 800 test calls, March to August 2026. 0 means no relation.", 13, MUTED)]
    for i, (label, kind) in enumerate([("setup", "model"), ("setup without the call", "control"), ("simple rule (no AI)", "rule")]):
        x = 20 + i * 180
        body += [f'<rect x="{x}" y="66" width="12" height="12" rx="2" fill="{FILL[kind]}"/>', text(x + 18, 77, label, 13, MUTED)]
    for i, r in enumerate(rows):
        ic, y = float(r["stock_ic"]), y0 + i * rh
        bar = ic / top_ic * w
        body += [text(left - 10, y + 15, r["setup"], anchor="end"),
                 f'<rect x="{left}" y="{y + 3}" width="{bar:.1f}" height="{rh - 12}" rx="3" fill="{FILL[KIND.get(r["setup"], "model")]}"/>',
                 text(left + bar + 6, y + 17, f"{ic:.3f}")]
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


if __name__ == "__main__":
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "figures" / "stock_ic_by_setup.svg").write_text(stock_ic())
    (HERE / "figures" / "ladder_step_gains.svg").write_text(gaps())
    print("wrote", sorted(p.name for p in (HERE / "figures").glob("*.svg")))
