"""Deterministic checks. Everything a rule can decide lives here, not in the judge.

Each assertion returns True / False, or None when it does not apply to the case.
They were moved OUT of the judge prompt (see judge_v0.txt -> judge_v1.txt): paying
a model to check whether "Section 3" exists is slower, costs money, and has off days.
"""

import re

from policy_index import effective_dates, load_index

# The app's refusal sentence (app/generation/prompts.py). Matching the phrase, not
# the full sentence, so "I cannot answer that from the documents" still counts.
REFUSAL_RE = re.compile(r"\bcannot answer\b|\bcan't answer\b|\bnot (?:covered|contained|addressed) in the (?:provided )?(?:documents|context)", re.I)

# A figure written with digits and a unit: "2 weeks", "600 USD", "$75", "5%", "10:00".
# \s also covers the narrow no-break space the model emits ("5 %").
NUMERIC_FIGURE_RE = re.compile(
    r"(?:\$\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?:\b\d[\d,]*(?:\.\d+)?\s*%)"  # no \b after "%": it is not a word character
    r"|(?:\b\d[\d,]*(?:\.\d+)?\s*(?:percent|usd|dollars|weeks?|days?|business days?|months?|hours?|minutes?|x)\b)"
    r"|(?:\b\d{1,2}:\d{2}\b)",
    re.I,
)

SECTION_REF_RE = re.compile(r"(?:\bSection|\bSec\.|§)\s*(\d+)", re.I)
DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b"
)
_WINDOW = 120  # chars either side of "Section N" searched for the policy's name


def is_refusal(answer):
    return bool(REFUSAL_RE.search(answer or ""))


def _nearest_policy(text, pos):
    """The policy named closest to position `pos` within the window, or None."""
    lo, hi = max(0, pos - _WINDOW), pos + _WINDOW
    window = text[lo:hi].lower()
    best, best_dist = None, None
    for fname, doc in load_index().items():
        for alias in doc["aliases"]:
            for m in re.finditer(re.escape(alias), window):
                # distance from the section ref to the nearer end of the policy name
                start, end = lo + m.start(), lo + m.end()
                dist = min(abs(start - pos), abs(end - pos))
                if best_dist is None or dist < best_dist:
                    best, best_dist = fname, dist
    return best


def section_refs(answer):
    """[(section_no, resolved_source_filename_or_None, resolves: bool)] for every 'Section N'."""
    text = answer or ""
    refs = []
    for m in SECTION_REF_RE.finditer(text):
        n = int(m.group(1))
        fname = _nearest_policy(text, m.start())
        ok = fname is not None and n in load_index()[fname]["sections"]
        refs.append((n, fname, ok))
    return refs


def cited_versions(answer):
    """Effective dates in the answer that match a real handbook version."""
    valid = effective_dates()
    found = set()
    for m in DATE_RE.finditer(answer or ""):
        date = f"{m.group(1)} {int(m.group(2))}, {m.group(3)}"
        if date in valid:
            found.add(date)
    return found


# ---- the assertions -----------------------------------------------------------

def a_refusal_path(case, answer):
    """Out-of-jurisdiction / not-in-handbook question -> the refusal path fires."""
    if case["expect"] != "refuse":
        return None
    return is_refusal(answer)


def a_no_false_refusal(case, answer):
    """Answerable handbook question -> the app does NOT refuse."""
    if case["expect"] != "answer":
        return None
    return not is_refusal(answer)


def a_figure_numeric(case, answer):
    """Questions asking for an amount / duration / notice period get a numeric figure."""
    if case["expect"] != "answer" or not case.get("numeric"):
        return None
    return bool(NUMERIC_FIGURE_RE.search(answer or ""))


def a_section_ref_resolves(case, answer):
    """At least one 'Section N' reference, and every one resolves to a real section."""
    if case["expect"] != "answer":
        return None
    refs = section_refs(answer)
    return bool(refs) and all(ok for _, _, ok in refs)


def a_version_cited(case, answer):
    """The handbook version (a real effective date) of a cited policy is stated."""
    if case["expect"] != "answer":
        return None
    versions = cited_versions(answer)
    if not versions:
        return False
    cited_docs = {f for _, f, ok in section_refs(answer) if ok}
    if not cited_docs:
        return True  # a real version is cited even if no section resolved; that fails its own check
    return any(load_index()[f]["effective"] in versions for f in cited_docs)


ASSERTIONS = {
    "refusal_path": a_refusal_path,
    "no_false_refusal": a_no_false_refusal,
    "figure_numeric": a_figure_numeric,
    "section_ref_resolves": a_section_ref_resolves,
    "version_cited": a_version_cited,
}


def run_assertions(case, answer):
    return {name: fn(case, answer) for name, fn in ASSERTIONS.items()}
