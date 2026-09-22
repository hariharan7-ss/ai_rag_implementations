"""Shared setup for the eval harness.

Import this BEFORE anything from `app` -- it puts rag_app on sys.path, loads the
.env, and pins PINECONE_NAMESPACE so the eval corpus never lands in the
namespace the running app uses.
"""

import json
import os
import re
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(EVAL_DIR)  # rag_app/

if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from dotenv import load_dotenv

# Explicit path, not find_dotenv(): find_dotenv walks up from the calling file
# and comes up empty when a script is piped in over stdin.
load_dotenv(os.path.join(APP_ROOT, ".env"), override=True)

# Everything the harness reads or writes lives in its own namespace. The app's
# default namespace ("") is never touched by the eval scripts.
EVAL_NAMESPACE = os.environ.get("EVAL_NAMESPACE", "hr")
os.environ["PINECONE_NAMESPACE"] = EVAL_NAMESPACE

CORPUS_DIR = os.path.join(EVAL_DIR, "corpus")
CHUNKS_PATH = os.path.join(EVAL_DIR, "chunks.jsonl")
GOLDEN_PATH = os.path.join(EVAL_DIR, "golden_set.jsonl")
RUNS_DIR = os.path.join(EVAL_DIR, "runs")

_MONTHS = (
    "January|February|March|April|May|June|July|"
    "August|September|October|November|December"
)

# Tokens a question can pin an answer to that dense embeddings tend to blur:
# policy numbers, section/clause pointers, dates, region codes. Flags are
# scoped -- (?i:...) per branch -- because region codes must stay case
# sensitive (IN / UK / US-CA are tokens, "in" is a stopword).
EXACT_TOKEN_RE = re.compile(
    r"""
      (?i:\b[A-Z]{2,4}-\d{1,4}\b)                        # policy number: HR-207
    | (?i:\bsections?\s+\d+(?:\.\d+)*\b)                 # section 4.2
    | (?i:\bclauses?\s+\d+(?:\.\d+)*\b)                  # clause 7.1
    | \u00a7\s*\d+(?:\.\d+)*                             # section sign: 4.2
    | \b\d{4}-\d{2}-\d{2}\b                              # ISO date: 2025-04-01
    | (?i:\b\d{1,2}(?:st|nd|rd|th)?\s+(?:"""
    + _MONTHS
    + r""")\s+\d{4}\b)                                   # 1 April 2025
    | (?i:\b(?:"""
    + _MONTHS
    + r""")\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b)       # April 1, 2025
    | \b[A-Z]{2,3}(?:-[A-Z]{2,3})?\b                     # region code: IN, UK, US-CA
    """,
    re.VERBOSE,
)


def exact_tokens(text):
    """Exact tokens in `text`, in order of appearance, de-duplicated."""
    seen = []
    for match in EXACT_TOKEN_RE.findall(text or ""):
        token = match.strip()
        if token and token not in seen:
            seen.append(token)
    return seen


def load_jsonl(path):
    """Read a .jsonl file into a list of dicts. Blank lines are skipped."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    """Write dicts to a .jsonl file, creating the parent directory."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def gold_ids(q):
    """Gold chunk ids for a golden-set row.

    Accepts either "gold_chunk_id": "x" or "gold_chunk_ids": ["x", "y"] -- a
    question whose answer legitimately spans two chunks counts as a hit if
    either one is retrieved.
    """
    ids = []
    one = q.get("gold_chunk_id")
    if one:
        ids.append(one)
    many = q.get("gold_chunk_ids") or []
    for chunk_id in many:
        if chunk_id and chunk_id not in ids:
            ids.append(chunk_id)
    return ids


def rank_of(chunk_ids, gold):
    """1-based rank of the best gold chunk in `chunk_ids`, or None if absent."""
    for i, chunk_id in enumerate(chunk_ids):
        if chunk_id in gold:
            return i + 1
    return None
