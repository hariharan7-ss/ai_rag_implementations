import json
import sys
import time
import requests

API_URL = "http://127.0.0.1:8000/api/v1/query"
# Pass a path to keep an earlier run's results instead of overwriting them.
OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/run_results.json"

with open("data/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

results = []
for i, q in enumerate(questions, 1):
    resp = requests.post(API_URL, json={"query": q}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    results.append({"n": i, "question": q, "trace_id": data.get("trace_id"), "answer": data.get("answer")})
    safe_q = q.encode("ascii", "replace").decode("ascii")
    safe_a = (data.get("answer") or "")[:200].encode("ascii", "replace").decode("ascii")
    print(f"[{i}/{len(questions)}] {safe_q}")
    print(f"  -> trace_id={data.get('trace_id')}")
    print(f"  -> answer: {safe_a!r}")
    time.sleep(0.3)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nDone. {len(results)} traces run.")
