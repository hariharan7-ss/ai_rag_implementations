"""The same task as a fixed workflow: hard-coded steps, same tools, same model, same output contract.

    python -m app.agent.workflow "The company is eliminating E1004's role. What minimum notice must it give?"

Six steps, always in this order, each run exactly once - no loop, no model-chosen tool calls:
  1. code   employee_id = first E#### in the question
  2. LLM    route: topic (enum) + handbook keywords          (JSON mode)
  3. tool   get_employee_record(employee_id)
  4. tool   get_jurisdiction_rules(record.jurisdiction, topic)
  5. tool   search_handbook(handbook keywords)
  6. LLM    answer from the facts gathered in 3-5, in the shared output contract

Step 4's arguments depend on what step 3 found (the record's jurisdiction), so tenure- and
jurisdiction-dependent rules are handled - but the path itself never changes. Anything the
record points to (a prior record, a second jurisdiction) is not fetched.
"""

import argparse
import json
import sys
import time
from typing import Any, Callable, Dict

from app.agent.contract import (EMPLOYEE_ID, PROMPT_VERSION, WORKFLOW_ANSWER, WORKFLOW_ROUTER,
                                parse_contract, stopped_contract)
from app.agent.model import MODEL, chat
from app.agent.tools import TOPICS, run_tool


def run_workflow(question: str, log: Callable[[str], None] = print, chat_fn=chat) -> Dict[str, Any]:
    t0 = time.monotonic()
    calls, steps, tools_used = [], [], []

    def el() -> str:
        return f"{time.monotonic() - t0:6.2f}s"

    def tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        result = run_tool(name, args)
        tools_used.append(name)
        steps.append({"step": name, "args": args, "result": result})
        log(f"[workflow {el()}] tool {name}({json.dumps(args)}) -> {json.dumps(result, ensure_ascii=False)[:140]}")
        return result

    def finish(final: Dict[str, Any]) -> Dict[str, Any]:
        wall = time.monotonic() - t0
        rl = sum(c.rate_limit_wait_s for c in calls)
        return {"system": "workflow", "prompt_version": PROMPT_VERSION, "model": MODEL, "question": question,
                "final": final, "stopped_by": None, "stop_detail": None, "budgets": None,
                "llm_calls": len(calls), "tool_calls": tools_used,
                "prompt_tokens": sum(c.prompt_tokens for c in calls),
                "completion_tokens": sum(c.completion_tokens for c in calls),
                "total_tokens": sum(c.total_tokens for c in calls),
                "cost_usd": sum(c.cost_usd for c in calls),
                "wall_s": wall, "rate_limit_wait_s": rl, "latency_s": wall - rl, "steps": steps}

    log(f"[workflow {el()}] question: {question}")
    # 1. employee id (code)
    m = EMPLOYEE_ID.search(question)
    if not m:
        return finish(stopped_contract(question, "The question names no employee ID."))
    employee_id = m.group(0)

    # 2. route (LLM, JSON mode)
    route_call = chat_fn([{"role": "system", "content": WORKFLOW_ROUTER},
                          {"role": "user", "content": question}], response_format={"type": "json_object"})
    calls.append(route_call)
    try:
        route = json.loads(route_call.message.content)
    except (TypeError, json.JSONDecodeError):
        route = {}
    topic = route.get("topic") if route.get("topic") in TOPICS else None
    query = route.get("handbook_query") or question
    steps.append({"step": "route", "result": route, "tokens": route_call.total_tokens})
    log(f"[workflow {el()}] llm route -> topic={topic} query={query!r} ({route_call.total_tokens:,} tok)")

    # 3-5. tools, fixed order
    record = tool("get_employee_record", {"employee_id": employee_id})
    rules = (tool("get_jurisdiction_rules", {"jurisdiction": record.get("jurisdiction"), "topic": topic})
             if topic and "error" not in record else {"note": "no jurisdiction rule looked up"})
    handbook = tool("search_handbook", {"query": query})

    # 6. answer (LLM)
    facts = json.dumps({"employee_record": record, "jurisdiction_rule": rules, "handbook_sections": handbook["sections"]},
                       ensure_ascii=False, indent=1)
    answer_call = chat_fn([{"role": "system", "content": WORKFLOW_ANSWER},
                           {"role": "user", "content": f"Facts:\n{facts}\n\nQuestion: {question}"}])
    calls.append(answer_call)
    log(f"[workflow {el()}] llm answer ({answer_call.total_tokens:,} tok): {(answer_call.message.content or '')[:400]}")
    return finish(parse_contract(answer_call.message.content))


def main():
    ap = argparse.ArgumentParser(description="Run the fixed HR entitlement workflow on one question.")
    ap.add_argument("question")
    ap.add_argument("--save", help="write the full run to this JSON file")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run = run_workflow(a.question, log=lambda s: print(s, flush=True))
    print(json.dumps(run["final"], indent=2, ensure_ascii=False))
    print(f"llm_calls={run['llm_calls']} tokens={run['total_tokens']:,} cost=${run['cost_usd']:.5f} "
          f"latency={run['latency_s']:.2f}s")
    if a.save:
        with open(a.save, "w", encoding="utf-8") as f:
            json.dump(run, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
