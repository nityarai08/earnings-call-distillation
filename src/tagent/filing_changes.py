"""What changed in a company's filing against the same filing a year earlier.

A 10-K is compared with the previous 10-K and a 10-Q with the same quarter's 10-Q a year before,
so seasonal wording cancels. Digits are ignored when matching, so a sentence that only updates its
numbers does not count as a change. Years become "YEAR", so the text gives no date.
"""
from __future__ import annotations

import gzip
import re

import pandas as pd

from .edgar import EDGAR_DIR

FILINGS = EDGAR_DIR / "filings"
SENT = re.compile(r"(?<=[.;:])\s+(?=[A-Z])")
WORDS = re.compile(r"[a-z]{3,}")


def read(cik: int, accession: str) -> str:
    with gzip.open(FILINGS / f"{cik}_{accession.replace('-', '')}.txt.gz", "rt") as g:
        return g.read()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT.split(text) if 60 <= len(s) <= 500 and len(WORDS.findall(s.lower())) >= 8]


def key(s: str) -> str:
    return " ".join(WORDS.findall(s.lower()))


def changes(old: str, new: str) -> dict:
    a, b = sentences(old), sentences(new)
    ka, kb = {key(s) for s in a}, {key(s) for s in b}
    added = list(dict.fromkeys(s for s in b if key(s) not in ka))
    removed = list(dict.fromkeys(s for s in a if key(s) not in kb))
    same = len(ka & kb) / max(len(ka | kb), 1)
    return {"similarity": same, "added": added, "removed": removed, "old_sentences": len(ka), "new_sentences": len(kb)}


def prior(index: pd.DataFrame) -> pd.DataFrame:
    """index: one company's filings. Adds the accession of the filing a year earlier, same form."""
    d = index.assign(filed=pd.to_datetime(index.filed)).sort_values("filed")
    out = []
    for r in d.itertuples():
        w = d[(d.form == r.form) & (d.filed >= r.filed - pd.Timedelta(days=400)) & (d.filed <= r.filed - pd.Timedelta(days=330))]
        out.append(w.accession.iloc[-1] if len(w) else None)
    return d.assign(prior=out)


def _prose(s: str) -> bool:
    """Drops table rows and page furniture: too many digits, or no ordinary sentence shape."""
    digits = sum(c.isdigit() for c in s) / len(s)
    return digits < .06 and s[0].isupper() and s[-1] in ".;:" and "  " not in s


def _novelty(s: str, other: list[set]) -> float:
    w = set(WORDS.findall(s.lower()))
    return 1 - max((len(w & o) / len(w | o) for o in other), default=0.0)


def brief(old: str, new: str, chars: int = 6000, keep: float = .5) -> dict:
    """The changes worth reading: real prose with at least `keep` of its wording new, most novel first,
    two parts added to one part removed, within `chars`. Years become YEAR."""
    c = changes(old, new)
    a = [set(WORDS.findall(s.lower())) for s in sentences(old)]
    b = [set(WORDS.findall(s.lower())) for s in sentences(new)]
    add = sorted(((_novelty(s, a), s) for s in c["added"] if _prose(s)), reverse=True)
    rem = sorted(((_novelty(s, b), s) for s in c["removed"] if _prose(s)), reverse=True)
    add, rem = [x for x in add if x[0] >= keep], [x for x in rem if x[0] >= keep]

    def fit(rows, room):
        out = []
        for _, s in rows:
            if len(s) + 1 > room:
                break
            room -= len(s) + 1
            out.append(re.sub(r"\b(19|20)\d\d\b", "YEAR", s))
        return out

    return {"similarity": c["similarity"], "new_prose": len(add), "gone_prose": len(rem),
            "old_sentences": c["old_sentences"], "new_sentences": c["new_sentences"],
            "added": fit(add, chars * 2 // 3), "removed": fit(rem, chars // 3)}
