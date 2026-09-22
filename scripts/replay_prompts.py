"""Replay stored prompts from traces/traces.jsonl against the same model, N times each.

llm.py sets no temperature, so one trace is one sample of what the model might say.
Re-sending the exact Week 3 prompt and the exact after-fix prompt N times each shows
what the retrieved context changes, separately from sampling noise. No retrieval
happens here, so Pinecone is never touched.

    python scripts/replay_prompts.py [out_path]
"""

import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"), override=True)

from app.generation.llm import generate_response
from app.services.rag_service import LLM_MODEL

N = 8
RUNS = {
    "before": os.path.join(ROOT, "data", "run_results.json"),
    "after": os.path.join(ROOT, "data", "run_results_after_fix.json"),
}


def refused(text):
    return "cannot answer" in text.lower()


def weeks(n):
    # \W{0,3} tolerates the non-breaking spaces the model emits ("20\u202fweeks");
    # requiring "weeks" stops "2026" from counting as "20".
    return re.compile(rf"\b{n}\W{{0,3}}weeks?\b", re.I)


# question number (the "n" in run_results) -> outcome label for one answer
CLASSIFY = {
    5: lambda t: "refuse" if refused(t) else ("answer" if re.search(r"\b15\b", t) and "approv" in t.lower() else "other"),
    14: lambda t: "refuse" if refused(t) else ("20 weeks" if weeks(20).search(t) else ("16 weeks" if weeks(16).search(t) else "other")),
    18: lambda t: "refuse" if refused(t) else ("answer" if "not eligible" in t.lower() or t.strip().lower().startswith("no") else "other"),
    20: lambda t: "refuse" if refused(t) else ("yes, $50 wellness" if "wellness" in t.lower() or t.strip().lower().startswith("yes") else "no"),
}


def tally(n, outputs):
    counts = {}
    for text in outputs:
        label = CLASSIFY[n](text)
        counts[label] = counts.get(label, 0) + 1
    return counts


def call(prompt):
    for attempt in range(6):
        try:
            return generate_response(prompt, model=LLM_MODEL)
        except Exception:  # rate limits: back off and retry
            time.sleep(5 * (attempt + 1))
    return "ERROR"


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "replay_results.json")
    with open(os.path.join(ROOT, "traces", "traces.jsonl"), "r", encoding="utf-8") as f:
        traces = {t["trace_id"]: t for t in map(json.loads, filter(str.strip, f))}
    runs = {}
    for side, path in RUNS.items():
        with open(path, "r", encoding="utf-8") as f:
            runs[side] = {r["n"]: r for r in json.load(f)}

    outputs, tallies = {}, {}
    for n in CLASSIFY:
        for side, run in runs.items():
            prompt = traces[run[n]["trace_id"]]["prompt"]
            key = f"q{n}_{side}"
            outputs[key] = []
            for _ in range(N):
                outputs[key].append(call(prompt))
                time.sleep(2.1)  # stay under the Groq rate limit
            tallies[key] = tally(n, outputs[key])
            print(f"#{n:<2} {side:6} {tallies[key]}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": LLM_MODEL, "samples_per_prompt": N, "tallies": tallies, "outputs": outputs},
                  f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}")
