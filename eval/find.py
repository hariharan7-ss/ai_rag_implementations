"""Grep eval/chunks.jsonl for the chunk that actually answers a question.

The tagging tool: write a question, run this with a couple of distinctive terms
from the answer, and it hands back the chunk_id to paste into golden_set.jsonl.

    python eval/find.py "carry over" vacation
"""

import argparse
import os
import re

import common  # noqa: F401  (must come first: sets sys.path + PINECONE_NAMESPACE)

DEFAULT_CONTEXT = 160


def ascii_safe(text):
    # The Windows console is cp1252 and the corpus carries curly quotes/dashes.
    return str(text).encode("ascii", "replace").decode("ascii")


def flatten(text):
    """Collapse every run of whitespace to a single space.

    Matching runs on the flattened text so a term like "carry over" still hits
    when the corpus wrapped those two words across a line break.
    """
    return re.sub(r"\s+", " ", text or "").strip()


def term_spans(text, terms):
    """Spans of every occurrence of every term, merged and in order."""
    low = text.lower()
    spans = []
    for term in terms:
        start = low.find(term)
        while start != -1:
            spans.append([start, start + len(term)])
            start = low.find(term, start + 1)

    spans.sort()
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def mark_terms(text, terms):
    """Wrap each matched term in [[ ]]. Back to front, so spans stay valid."""
    marked = text
    for start, end in reversed(term_spans(text, terms)):
        marked = marked[:start] + "[[" + marked[start:end] + "]]" + marked[end:]
    return marked


def snippet(text, terms, context):
    """Window around the first occurrence of the FIRST term, terms marked."""
    hit = text.lower().find(terms[0])
    if hit < 0:
        hit = 0
    start = max(0, hit - context)
    end = min(len(text), hit + len(terms[0]) + context)
    body = mark_terms(text[start:end], terms)
    return ("..." if start > 0 else "") + body + ("..." if end < len(text) else "")


def main():
    parser = argparse.ArgumentParser(
        description="Print chunks from chunks.jsonl containing ALL the given terms.",
        epilog='example: python eval/find.py "carry over" vacation',
    )
    parser.add_argument(
        "terms",
        nargs="*",
        help="substrings that must ALL appear in the chunk (case-insensitive); "
        "quote a multi-word term",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=DEFAULT_CONTEXT,
        help="characters of context on each side of the match (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="print at most N matches (default: all of them)",
    )
    args = parser.parse_args()

    # Terms get the same flattening as the chunk text, or a term typed with a
    # stray double space could never match.
    terms = [flatten(t).lower() for t in args.terms if flatten(t)]
    if not terms:
        parser.print_usage()
        print('give at least one term, e.g.: python eval/find.py "carry over" vacation')
        return

    if not os.path.exists(common.CHUNKS_PATH):
        raise SystemExit(
            ascii_safe(
                f"no chunks at {common.CHUNKS_PATH} -- run: python eval/ingest_hr.py first"
            )
        )

    rows = common.load_jsonl(common.CHUNKS_PATH)
    context = max(0, args.context)

    matched = []
    for row in rows:
        text = flatten(row.get("text", ""))
        low = text.lower()
        if all(term in low for term in terms):
            matched.append((row, text))

    # Every chunk is scanned even under --limit, so the tally below stays honest
    # about how many chunks really matched.
    shown = matched[: args.limit] if args.limit > 0 else matched
    for row, text in shown:
        print(ascii_safe(row.get("chunk_id") or "?"))
        print("  " + ascii_safe(row.get("title") or ""))
        print("  " + ascii_safe(snippet(text, terms, context)))
        print()

    tail = f"{len(matched)} of {len(rows)} chunks matched"
    if len(shown) < len(matched):
        tail += f" (showing first {len(shown)})"
    print(tail)


if __name__ == "__main__":
    main()
