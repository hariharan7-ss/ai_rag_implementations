"""Replay one trace from the trace record ALONE.

Week 5 requirement 1: prove traces are replayable. Picks a trace_id with a
seeded RNG (seed is printed so the selection is reproducible), audits the record
for the fields replay needs, then re-sends the recorded prompt to the recorded
model with the recorded parameters and diffs the replayed output against the
original.

Nothing is read from the live app: retrieval is NOT re-run. If the trace cannot
supply a field, the audit says so rather than silently falling back to a default.

Usage:
    python scripts/replay_trace_week5.py [--seed N] [--trace-id ID] [--frame post|all]
"""

import argparse
import difflib
import json
import os
import random
import sys
import time

# Trace text contains characters the Windows console codepage cannot encode.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRACES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "traces", "traces.jsonl")

# What a trace must carry to be replayable without touching the live app.
REQUIRED = [
    ("trace_id",        lambda t: t.get("trace_id")),
    ("query",           lambda t: t.get("query")),
    ("prompt_version",  lambda t: t.get("prompt_version")),
    ("prompt",          lambda t: t.get("prompt")),
    ("retrieved chunk_ids", lambda t: [r.get("chunk_id") for r in t.get("retrieved", [])] or None),
    ("retrieved scores",    lambda t: [r.get("score") for r in t.get("retrieved", [])] or None),
    ("model",           lambda t: t.get("model")),
    ("model params",    lambda t: t.get("model_params")),
    ("retrieval params", lambda t: t.get("retrieval_params")),
    ("raw output",      lambda t: t.get("raw_output") if t.get("raw_output") is not None else t.get("answer")),
]


def load(frame):
    rows = [json.loads(l) for l in open(TRACES, encoding="utf-8") if l.strip()]
    if frame == "post":
        rows = [r for r in rows if r.get("schema_version")]
    return rows


def audit(trace):
    print("\n--- FIELD AUDIT (can this trace be replayed from itself alone?) ---")
    missing = []
    for name, get in REQUIRED:
        val = get(trace)
        ok = val is not None and val != [] and val != ""
        if not ok:
            missing.append(name)
        shown = ""
        if ok:
            s = json.dumps(val) if not isinstance(val, str) else val
            shown = s[:78].replace("\n", " ")
        print(f"  [{'OK ' if ok else 'MISSING'}] {name:<22} {shown}")
    return missing


def replay(trace):
    """Re-send the RECORDED prompt to the RECORDED model with the RECORDED params."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dotenv import load_dotenv
    load_dotenv(override=True)
    from groq import Groq

    params = trace.get("model_params") or {}
    kwargs = {k: v for k, v in params.items() if k in ("temperature", "top_p", "max_tokens", "seed")}
    # The free tier rate-limits on tokens per minute; retry rather than fail the
    # evidence run.
    last = None
    for attempt in range(8):
        try:
            resp = Groq().chat.completions.create(
                messages=[{"role": "user", "content": trace["prompt"]}],
                model=trace["model"],
                **kwargs,
            )
            return resp.choices[0].message.content, kwargs
        except Exception as exc:  # noqa: BLE001 - surfaced below if all retries fail
            last = exc
            if "rate_limit" not in str(exc) and "429" not in str(exc):
                raise
            time.sleep(8 * (attempt + 1))
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--trace-id")
    ap.add_argument("--frame", default="all", choices=["all", "post"])
    ap.add_argument("--audit-only", action="store_true")
    a = ap.parse_args()

    rows = load(a.frame)
    if a.trace_id:
        trace = next(r for r in rows if r["trace_id"] == a.trace_id)
        how = f"explicitly requested trace_id={a.trace_id}"
    else:
        ids = sorted(r["trace_id"] for r in rows)          # sort => order-independent
        rng = random.Random(a.seed)
        picked = rng.choice(ids)
        trace = next(r for r in rows if r["trace_id"] == picked)
        how = f"random.Random({a.seed}).choice(sorted(trace_ids)) over {len(ids)} traces in frame '{a.frame}'"

    print("=" * 78)
    print("REPLAY EVIDENCE")
    print("=" * 78)
    print(f"selection : {how}")
    print(f"trace_id  : {trace['trace_id']}")
    print(f"logged at : {trace.get('timestamp')}")
    print(f"query     : {trace.get('query')}")

    missing = audit(trace)
    if missing:
        print(f"\n  => NOT fully replayable. Missing: {', '.join(missing)}")
    else:
        print("\n  => Fully replayable from the trace alone.")

    if a.audit_only:
        return

    print("\n--- REPLAY (recorded prompt -> recorded model, retrieval NOT re-run) ---")
    replayed, kwargs = replay(trace)
    print(f"params sent: {kwargs or '{} (none recorded -> provider defaults)'}")

    original = trace.get("raw_output") or trace.get("answer") or ""
    print("\n--- ORIGINAL OUTPUT ---")
    print(original)
    print("\n--- REPLAYED OUTPUT ---")
    print(replayed)
    print("\n--- COMPARISON ---")
    if original.strip() == replayed.strip():
        print("IDENTICAL (byte-for-byte after strip)")
    else:
        sm = difflib.SequenceMatcher(None, original, replayed)
        print(f"NOT identical - similarity {sm.ratio():.3f}")
        for line in difflib.unified_diff(original.splitlines(), replayed.splitlines(),
                                         "original", "replayed", lineterm="", n=1):
            print(line)


if __name__ == "__main__":
    main()
