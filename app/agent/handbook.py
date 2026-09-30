"""Section-level BM25 over the company-wide handbook (data/hr_policies/*.txt).

The agent and the workflow share this search, so it is local and deterministic:
the race should measure the loop, not Pinecone round-trips or the Week 3/5
namespace contamination. Each numbered policy section is one searchable unit
with a stable id, HB-<file prefix>-<section number>, that answers can cite.
"""

import math
import os
import re
from collections import Counter
from functools import lru_cache
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLICY_DIR = os.path.join(ROOT, "data", "hr_policies")

_SECTION_RE = re.compile(r"^(\d+)\.\s+(.+?)\s*$")
_EFFECTIVE_RE = re.compile(r"^Effective:\s*(.+?)\s*$")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an and are as at be by can do does for from has have how i if in is it its many much my of on or "
    "our per the their them they this to was what when where which who will with".split()
)


def _tokens(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP]


@lru_cache(maxsize=1)
def load_sections() -> List[Dict]:
    sections = []
    for fname in sorted(os.listdir(POLICY_DIR)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(POLICY_DIR, fname), encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f]
        policy = re.split(r"\s+[—–-]\s+", lines[0], maxsplit=1)[-1].strip()
        prefix = fname.split("_", 1)[0]
        effective, current = None, None
        for line in lines[1:]:
            m = _EFFECTIVE_RE.match(line)
            if m and effective is None:
                effective = m.group(1)
                continue
            m = _SECTION_RE.match(line)
            if m:
                current = {"section_id": f"HB-{prefix}-{m.group(1)}", "policy": policy,
                           "section": f"{m.group(1)}. {m.group(2)}", "effective": effective, "text": ""}
                sections.append(current)
            elif current is not None and line.strip():
                # Unnumbered trailing text (e.g. the Parental Leave "Note:") stays with the section above it.
                current["text"] = (current["text"] + " " + line.strip()).strip()
    return sections


@lru_cache(maxsize=1)
def _index():
    docs = [_tokens(f"{s['policy']} {s['section']} {s['text']}") for s in load_sections()]
    df = Counter(t for d in docs for t in set(d))
    avgdl = sum(len(d) for d in docs) / len(docs)
    return docs, df, avgdl


def search(query: str, top_k: int = 3, k1: float = 1.5, b: float = 0.75) -> List[Dict]:
    docs, df, avgdl = _index()
    n = len(docs)
    q = _tokens(query)
    scored = []
    for sec, d in zip(load_sections(), docs):
        tf = Counter(d)
        score = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        if score > 0:
            scored.append((score, sec))
    scored.sort(key=lambda x: -x[0])
    return [dict(sec) for _, sec in scored[:top_k]]
