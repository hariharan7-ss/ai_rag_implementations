"""Offline proof that all four budgets are enforced by the loop (no API calls, no tokens spent).

    python eval/week7/test_budgets.py

A stub model that NEVER stops asking for tools - the worst case, a spinning agent - is run
four times with one budget tightened each time. Each run must end with that budget named in
`stopped_by`, a well-formed cannot_answer contract, and no overshoot of the budget itself.
"""

import os
import sys
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agent.agent import Budgets, run_agent  # noqa: E402
from app.agent.model import Call, DeadlineExceeded  # noqa: E402


def spinning_model(delay_s=0.0):
    """Always requests get_employee_record again; the prompt grows ~600 tokens a lap like the real loop."""
    n = {"lap": 0}

    def chat_fn(messages, tools=None, max_tokens=None, deadline=None, response_format=None):
        if deadline is not None and time.monotonic() + delay_s >= deadline:
            raise DeadlineExceeded("stub call would run past the deadline")
        time.sleep(delay_s)
        n["lap"] += 1
        tc = SimpleNamespace(id=f"call_{n['lap']}", type="function",
                             function=SimpleNamespace(name="get_employee_record", arguments='{"employee_id": "E1004"}'))
        msg = SimpleNamespace(content="", tool_calls=[tc])
        return Call(message=msg, finish_reason="tool_calls", prompt_tokens=900 + 600 * n["lap"],
                    completion_tokens=min(120, max_tokens), cached_tokens=0, latency_s=delay_s, rate_limit_wait_s=0.0)
    return chat_fn


CASES = [
    ("max_iters",    Budgets(max_iters=3, max_tokens=10**9, max_cost_usd=10.0, max_wall_s=600), 0.0),
    ("max_tokens",   Budgets(max_iters=99, max_tokens=8_000, max_cost_usd=10.0, max_wall_s=600), 0.0),
    ("max_cost_usd", Budgets(max_iters=99, max_tokens=10**9, max_cost_usd=0.0008, max_wall_s=600), 0.0),
    ("max_wall_s",   Budgets(max_iters=99, max_tokens=10**9, max_cost_usd=10.0, max_wall_s=1.0), 0.3),
]


def main():
    failures = 0
    for expected, budgets, delay in CASES:
        lines = []
        run = run_agent("The company is eliminating E1004's role. What minimum notice must it give?",
                        budgets=budgets, log=lines.append, chat_fn=spinning_model(delay))
        ok = (run["stopped_by"] == expected
              and run["final"]["status"] == "cannot_answer"
              and run["llm_calls"] <= budgets.max_iters
              and run["total_tokens"] <= budgets.max_tokens
              and run["cost_usd"] <= budgets.max_cost_usd
              and run["wall_s"] <= budgets.max_wall_s + 0.5)
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {expected:13} stopped_by={run['stopped_by']!s:13} laps={run['llm_calls']} "
              f"tokens={run['total_tokens']:,} cost=${run['cost_usd']:.5f} wall={run['wall_s']:.2f}s")
        print("      " + next(l for l in lines if "STOP" in l).split("] ", 1)[1])
    print(f"\n{len(CASES) - failures}/{len(CASES)} budgets fired as expected")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
