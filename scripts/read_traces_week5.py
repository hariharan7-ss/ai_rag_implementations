"""Render sampled traces in a readable form for hand open-coding.

Read-only. This script does not judge, score, or categorise anything -- it only
lays out what each trace contains so a human can read it and write one sentence
about what they saw.

Usage:
    python scripts/read_traces_week5.py --sample data/sample_week5.json
    python scripts/read_traces_week5.py --sample data/sample_week5.json --full
"""

import argparse
import json
import os
import sys

# Trace text contains characters the Windows console codepage cannot encode.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACES = os.path.join(ROOT, "traces", "traces.jsonl")


def load_traces():
    by_id = {}
    for line in open(TRACES, encoding="utf-8"):
        line = line.strip()
        if line:
            rec = json.loads(line)
            by_id[rec["trace_id"]] = rec
    return by_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/sample_week5.json")
    ap.add_argument("--full", action="store_true", help="print full answer and chunk text")
    ap.add_argument("--chunk-chars", type=int, default=0)
    a = ap.parse_args()

    sample = json.load(open(os.path.join(ROOT, a.sample), encoding="utf-8"))
    traces = load_traces()

    print(f"# sample: {a.sample}  seed={sample['seed']}  n={sample['n']}  "
          f"frame_size={sample['frame_size']}")

    for i, row in enumerate(sample["selected"], 1):
        t = traces[row["trace_id"]]
        print("\n" + "=" * 100)
        print(f"[{i:02}] {t['trace_id']}   {t.get('timestamp','')}")
        print(f"Q: {t.get('query')}")
        print(f"-- retrieved (final_k, score desc) --")
        for r in t.get("retrieved", []):
            print(f"   {r.get('score'):.4f}  {r.get('source')}  {r.get('chunk_id')}")
            if a.chunk_chars:
                txt = (r.get("text") or "").replace("\n", " ")
                print(f"           {txt[:a.chunk_chars]}")
        ans = t.get("raw_output") or t.get("answer") or ""
        print(f"-- answer ({len(ans)} chars, finish={t.get('finish_reason')}) --")
        print(ans if a.full else ans[:1200])


if __name__ == "__main__":
    main()
