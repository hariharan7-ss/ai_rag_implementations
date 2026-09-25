"""Lookup table of the handbook: policy name -> effective date + section numbers.

Built by parsing data/hr_policies/*.txt, so the assertions that check "does this
section reference resolve?" and "is the handbook version cited?" are an `if` and a
dict lookup, not a model call.
"""

import os
import re
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLICY_DIR = os.path.join(ROOT, "data", "hr_policies")

_SECTION_RE = re.compile(r"^(\d+)\.\s+(.+?)\s*$")
_EFFECTIVE_RE = re.compile(r"^Effective:\s*(.+?)\s*$")


def _aliases(name):
    """Ways an answer may name a policy: 'Parental Leave Policy' -> also 'Parental Leave'."""
    low = name.lower()
    out = {low}
    for suffix in (" policy", " process", " overview"):
        if low.endswith(suffix):
            out.add(low[: -len(suffix)])
    if "addendum" in low:
        out |= {"policy addendum", "addendum", "2026 addendum", "hr addendum"}
    if low.startswith("code of conduct"):
        out.add("code of conduct")
    return out


@lru_cache(maxsize=1)
def load_index():
    """{source filename: {"name", "effective", "sections": {n: heading}, "aliases"}}"""
    index = {}
    for fname in sorted(os.listdir(POLICY_DIR)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(POLICY_DIR, fname), encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f]
        # "Solstice Analytics — Leave and Time Off Policy"
        name = re.split(r"\s+[—–-]\s+", lines[0], maxsplit=1)[-1].strip()
        effective, sections = None, {}
        for line in lines[1:]:
            m = _EFFECTIVE_RE.match(line)
            if m and effective is None:
                effective = m.group(1)
                continue
            m = _SECTION_RE.match(line)
            if m:
                sections[int(m.group(1))] = m.group(2)
        index[fname] = {"name": name, "effective": effective, "sections": sections, "aliases": _aliases(name)}
    return index


def effective_dates():
    return {doc["effective"] for doc in load_index().values() if doc["effective"]}


if __name__ == "__main__":
    for fname, doc in load_index().items():
        print(f"{fname:42} {doc['name']!r:48} effective={doc['effective']!r:20} sections={sorted(doc['sections'])}")
