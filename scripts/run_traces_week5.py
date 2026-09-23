"""Run the Week 5 fresh question batch through the running API and record results.

Reads data/questions_week5.json (or a path passed as argv[1]) and writes one row
per question — n, question, trace_id, answer — to data/run_results_week5.json
(argv[2]). trace_id links each answer to its full record in traces/traces.jsonl.

Usage:
    python scripts/run_traces_week5.py [questions_path] [out_path]
"""

import json
import sys
import time
import requests

API_URL = "http://127.0.0.1:8000/api/v1/query"
QUESTIONS_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/questions_week5.json"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "data/run_results_week5.json"

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

def post_query(q, retries=6):
    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, json={"query": q}, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} for {q!r}: {type(e).__name__} {str(e)[:120]}")
            time.sleep(4 * (attempt + 1))

results = []
for i, q in enumerate(questions, 1):
    data = post_query(q)
    results.append({"n": i, "question": q, "trace_id": data.get("trace_id"), "answer": data.get("answer")})
    safe_q = q.encode("ascii", "replace").decode("ascii")
    safe_a = (data.get("answer") or "")[:160].encode("ascii", "replace").decode("ascii")
    print(f"[{i}/{len(questions)}] {safe_q}")
    print(f"  -> trace_id={data.get('trace_id')}")
    print(f"  -> {safe_a!r}")
    time.sleep(0.4)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nDone. {len(results)} traces run -> {OUT_PATH}")