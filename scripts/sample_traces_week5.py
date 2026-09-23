"""Draw a provable, seeded random sample of traces for Week 5 error analysis.

The sample must be reproducible by anyone holding this repo, so the selection is
a pure function of (seed, sampling frame):

    random.Random(SEED).sample(sorted(frame_trace_ids), N)

Sorting the frame first makes the draw independent of the order traces happen to
sit in the file. Nothing here looks at the answers, so the sample cannot be
steered toward traces that are already known to be broken.

Usage:
    python scripts/sample_traces_week5.py --seed 20260923 --n 20
    python scripts/sample_traces_week5.py --frame demo --n 10 --out data/sample_demo_week5.json
"""

import argparse
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACES = os.path.join(ROOT, "traces", "traces.jsonl")


def load_traces():
    by_id = {}
    for line in open(TRACES, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        by_id[rec["trace_id"]] = rec
    return by_id


def frame_ids(frame_file):
    """The sampling frame: every trace produced by the recorded run, no filtering."""
    rows = json.load(open(os.path.join(ROOT, frame_file), encoding="utf-8"))
    return [r["trace_id"] for r in rows if r.get("trace_id")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--frame", default="data/run_results_week5.json",
                    help="run-results JSON defining the sampling frame")
    ap.add_argument("--out", default="data/sample_week5.json")
    a = ap.parse_args()

    ids = frame_ids(a.frame)
    universe = sorted(set(ids))
    if len(universe) != len(ids):
        raise SystemExit(f"frame has duplicate trace_ids ({len(ids)} rows, {len(universe)} unique)")

    rng = random.Random(a.seed)
    picked = rng.sample(universe, a.n)

    traces = load_traces()
    missing = [t for t in picked if t not in traces]
    if missing:
        raise SystemExit(f"sampled trace_ids absent from traces.jsonl: {missing}")

    out = {
        "method": "random.Random(SEED).sample(sorted(set(frame_trace_ids)), N)",
        "seed": a.seed,
        "n": a.n,
        "frame_file": a.frame,
        "frame_size": len(universe),
        "sampling_rate": round(a.n / len(universe), 4),
        "selected": [
            {"trace_id": t, "query": traces[t].get("query")} for t in picked
        ],
    }
    path = os.path.join(ROOT, a.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"seed={a.seed}  n={a.n}  frame={a.frame}  frame_size={len(universe)}")
    print(f"method: {out['method']}")
    print()
    for i, t in enumerate(picked, 1):
        q = (traces[t].get("query") or "")[:88]
        print(f"{i:2}. {t}  {q}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
