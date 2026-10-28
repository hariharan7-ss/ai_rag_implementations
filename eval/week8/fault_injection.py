"""A/B the Week-8 mitigation under forced failure (the natural run never exercised it).

    python eval/week8/fault_injection.py            # 10 questions x {without, with} feedback -> runs/fault.json

On every run the SECOND model call is rejected like Groq's tool_use_failed. After that, a retry
whose message list is byte-identical to the rejected one is rejected again - the behaviour seen in
Week 7 q02, where the pinned seed (20260923) reproduced the same bad generation. A retry whose
messages differ goes to the REAL model. So the without-feedback arm cannot recover by
construction (it measures what max_iters costs); the with-feedback arm tests whether the real
model recovers from the feedback and what the recovery costs. The injection is the only fake part.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import httpx, groq
import trajectory_eval as te
from app.agent.agent import run_agent
from app.agent.model import chat, wait_for_full_bucket

MSG = ("Tool call validation failed: attempted to call tool 'search_handbook<|channel|>commentary' "
       "which was not in request.tools")

def rejection():
    resp = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com/x"))
    return groq.BadRequestError(f"Error code: 400 - {MSG}", response=resp, body={"error": {"message": MSG}})

def injecting_chat():
    st = {"n": 0, "rejected": None}
    def fn(messages, **kw):
        st["n"] += 1
        snap = json.dumps(messages, sort_keys=True, default=str)
        if st["n"] == 2 or snap == st["rejected"]:
            st["rejected"] = snap
            raise rejection()
        return chat(messages, **kw)
    return fn

def main():
    cases, paths, emp = te.load_cases(), te.load_paths(), te.known_employees()
    rows = []
    for c in cases:
        for arm, fb in (("without_feedback", False), ("with_feedback", True)):
            wait_for_full_bucket()
            run = run_agent(c["question"], log=lambda s: None, chat_fn=injecting_chat(), feedback_on_reject=fb)
            run.update(id=c["id"], **{"class": c["class"]}, trial=1, arm=arm)
            rows.append(run)
            print(f"{c['id']} {arm:17} stopped_by={run['stopped_by']} laps={run['llm_calls']} tok={run['total_tokens']:,} "
                  f"final={run['final'].get('value')} {run['final'].get('unit')}", flush=True)
    os.makedirs(te.RUNS_DIR, exist_ok=True)
    json.dump({"rows": rows}, open(os.path.join(te.RUNS_DIR, "fault.json"), "w", encoding="utf-8"), indent=1, default=str)
    for arm in ("without_feedback", "with_feedback"):
        rs = [te.annotate(dict(r), next(c for c in cases if c["id"] == r["id"]), paths[r["id"]], emp) for r in rows if r["arm"] == arm]
        s = te.summarise(rs)
        print(f"\n{arm}: outcome {s['outcome_rate']:.0%} trajectory {s['traj_rate']:.0%} budget_hit runs {s['mode_runs']['budget_hit']} "
              f"rejected laps {s['mode_counts']['rejected_tool_call']} (1 injected per run) cost p50 ${s['cost_p50']:.6f} max ${s['cost_max']:.6f} "
              f"tokens p50 {s['tokens_p50']:,.0f} max {s['tokens_max']:,} latency p50 {s['latency_p50']:.2f}s max {s['latency_max']:.2f}s "
              f"laps mean {sum(r['llm_calls'] for r in rs)/len(rs):.1f}")
if __name__ == "__main__":
    main()
