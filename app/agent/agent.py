"""Hand-built agent loop: think -> call tools -> read results -> repeat, every step logged.

    python -m app.agent.agent "The company is eliminating E1004's role. What minimum notice must it give?"
    python -m app.agent.agent --max-tokens 4000 "..."     # tighten any budget from the command line

Four budgets are checked in code before EVERY lap, not just declared:
  max_iters    - model calls (laps) in this run
  max_tokens   - prompt + completion tokens summed over every lap. The loop re-sends the whole
                 message list each lap, so the next lap's prompt is projected and must fit too.
  max_cost_usd - the same sum priced per token, projected the same way
  max_wall_s   - wall clock; also the timeout on each model call and the cap on any 429 wait
A lap's completion is capped to what the token and cost budgets have left, so a single lap
cannot overshoot them either. When a budget fires the loop stops and returns a well-formed
"cannot_answer" contract naming the budget - it never raises and never spins.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

import groq

from app.agent.contract import AGENT_SYSTEM, PROMPT_VERSION, parse_contract, stopped_contract
from app.agent.model import MODEL, PARAMS, PRICE_PER_M, DeadlineExceeded, chat, cost_usd
from app.agent.tools import TOOL_SPECS, run_tool


@dataclass
class Budgets:
    # Measured over 20 race runs (eval/week7/README.md): 3-5 laps, peak 6.4k tokens, $0.0005, 6.2s.
    # Tokens/cost leave ~3x/6x headroom; wall-clock also has to absorb a free-tier 429 wait (up to ~60s).
    # max_iters is the one that caught a real spin: rejected tool calls report no tokens at all.
    max_iters: int = 8
    max_tokens: int = 20_000
    max_cost_usd: float = 0.003
    max_wall_s: float = 90.0


# A lap that cannot afford this many completion tokens is not started.
MIN_COMPLETION = 256


def _chars(obj: Any) -> int:
    return len(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False))


def _short(obj: Any, n: int = 160) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "..."


def run_agent(question: str, budgets: Optional[Budgets] = None, log: Callable[[str], None] = print,
              chat_fn=chat, feedback_on_reject: bool = True) -> Dict[str, Any]:
    # feedback_on_reject=False restores the Week 7 behaviour; it exists only so eval/week8/fault_injection.py can A/B the mitigation.
    b = budgets or Budgets()
    t0 = time.monotonic()
    deadline = t0 + b.max_wall_s
    messages: List[Dict[str, Any]] = [{"role": "system", "content": AGENT_SYSTEM},
                                      {"role": "user", "content": question}]
    steps: List[Dict[str, Any]] = []
    tokens = prompt_tokens = completion_tokens = 0
    cost = api_latency = rl_wait = 0.0
    laps = 0
    tool_calls_made: List[str] = []
    # ~3 chars/token over-estimates, which is the safe direction for a budget.
    next_prompt = (_chars(messages) + _chars(TOOL_SPECS)) // 3
    final, stopped_by, stop_detail = None, None, None

    def el() -> str:
        return f"{time.monotonic() - t0:6.2f}s"

    def gate():
        """All four budgets. Returns (budget, detail) if the next lap must not start."""
        if laps >= b.max_iters:
            return "max_iters", f"{laps} laps done, limit {b.max_iters}"
        elapsed = time.monotonic() - t0
        if elapsed >= b.max_wall_s:
            return "max_wall_s", f"{elapsed:.1f}s elapsed, limit {b.max_wall_s:.0f}s"
        need = next_prompt + MIN_COMPLETION
        if tokens + need > b.max_tokens:
            return "max_tokens", (f"spent {tokens:,} + next lap >= {need:,} (prompt ~{next_prompt:,} re-sent) "
                                  f"= {tokens + need:,} > limit {b.max_tokens:,}")
        need_cost = cost_usd(next_prompt, MIN_COMPLETION)
        if cost + need_cost > b.max_cost_usd:
            return "max_cost_usd", (f"spent ${cost:.5f} + next lap >= ${need_cost:.5f} "
                                    f"= ${cost + need_cost:.5f} > limit ${b.max_cost_usd:.5f}")
        return None

    log(f"[agent {el()}] question: {question}")
    log(f"[agent {el()}] budgets: iters={b.max_iters} tokens={b.max_tokens:,} cost=${b.max_cost_usd} wall={b.max_wall_s:.0f}s")
    while True:
        hit = gate()
        if hit:
            stopped_by, stop_detail = hit
            break
        # Cap this lap's completion by whatever the two spend budgets have left.
        room_tokens = b.max_tokens - tokens - next_prompt
        room_cost = int((b.max_cost_usd - cost - cost_usd(next_prompt, 0)) * 1e6 / PRICE_PER_M["output"])
        cap = min(PARAMS["max_tokens"], room_tokens, room_cost)
        binding = None if cap == PARAMS["max_tokens"] else ("max_tokens" if room_tokens <= room_cost else "max_cost_usd")
        try:
            call = chat_fn(messages, tools=TOOL_SPECS, max_tokens=cap, deadline=deadline)
        except DeadlineExceeded as e:
            stopped_by, stop_detail = "max_wall_s", str(e)
            break
        except groq.BadRequestError as e:
            laps += 1
            # Groq rejects a malformed tool call (tool_use_failed); the lap is spent.
            log(f"[agent {el()}] lap {laps}: model produced an invalid tool call ({_short(str(e), 120)}) - retrying")
            steps.append({"lap": laps, "error": str(e)[:300]})
            # Week 8 mitigation (re-planning), the one change raced before->after: feed the
            # rejection back so the retry differs. Week 7 re-sent the identical request, and the
            # identical rejection came back up to 6 times until max_iters fired (q02).
            feedback = (f"Your last tool call was rejected before any tool ran: {_short(str(e), 200)}. "
                        f"Re-issue it calling exactly one tool from the list, with the tool name "
                        f"exactly as listed and nothing appended to it - or give the final JSON answer.")
            if feedback_on_reject:
                messages.append({"role": "user", "content": feedback})
                next_prompt += len(feedback) // 3
            continue

        laps += 1
        tokens += call.total_tokens
        prompt_tokens += call.prompt_tokens
        completion_tokens += call.completion_tokens
        cost += call.cost_usd
        api_latency += call.latency_s
        rl_wait += call.rate_limit_wait_s
        msg = call.message
        log(f"[agent {el()}] lap {laps}: prompt {call.prompt_tokens:,} + completion {call.completion_tokens:,} tok "
            f"({call.latency_s:.2f}s) | run total {tokens:,}/{b.max_tokens:,} tok, ${cost:.5f}/${b.max_cost_usd}"
            + (f", slept {call.rate_limit_wait_s:.0f}s on 429" if call.rate_limit_wait_s else ""))
        step = {"lap": laps, "prompt_tokens": call.prompt_tokens, "completion_tokens": call.completion_tokens,
                "latency_s": round(call.latency_s, 3), "finish_reason": call.finish_reason, "tool_calls": []}
        steps.append(step)

        if msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                            for tc in msg.tool_calls]})
            added = _chars(messages[-1])
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    result = run_tool(tc.function.name, args)
                except json.JSONDecodeError as e:
                    args, result = tc.function.arguments, {"error": f"arguments are not valid JSON: {e}"}
                log(f"[agent {el()}]   -> {tc.function.name}({_short(args, 100)})")
                log(f"[agent {el()}]   <- {_short(result)}")
                content = json.dumps(result, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                added += len(content)
                tool_calls_made.append(tc.function.name)
                step["tool_calls"].append({"name": tc.function.name, "args": args, "result": result})
            # Next lap re-sends everything: this lap's prompt plus what was just appended.
            next_prompt = call.prompt_tokens + added // 3
            continue

        if call.finish_reason == "length" and binding:
            stopped_by, stop_detail = binding, f"lap {laps} completion truncated at the {cap}-token cap the budget left"
            break
        final = parse_contract(msg.content)
        log(f"[agent {el()}] final answer after {laps} laps: {_short(msg.content, 400)}")
        break

    if stopped_by:
        log(f"[agent {el()}] STOP: budget '{stopped_by}' fired - {stop_detail}")
        log(f"[agent {el()}] terminated cleanly after {laps} laps; returning cannot_answer")
        final = stopped_contract(question, f"Stopped by the {stopped_by} budget before an answer was reached.")

    wall = time.monotonic() - t0
    return {"system": "agent", "prompt_version": PROMPT_VERSION, "model": MODEL, "question": question,
            "final": final, "stopped_by": stopped_by, "stop_detail": stop_detail, "budgets": asdict(b),
            "llm_calls": laps, "tool_calls": tool_calls_made,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": tokens,
            "cost_usd": cost, "wall_s": wall, "rate_limit_wait_s": rl_wait, "latency_s": wall - rl_wait,
            "steps": steps}


def main():
    ap = argparse.ArgumentParser(description="Run the HR entitlement agent on one question.")
    ap.add_argument("question")
    d = Budgets()
    ap.add_argument("--max-iters", type=int, default=d.max_iters)
    ap.add_argument("--max-tokens", type=int, default=d.max_tokens)
    ap.add_argument("--max-cost", type=float, default=d.max_cost_usd)
    ap.add_argument("--max-wall", type=float, default=d.max_wall_s)
    ap.add_argument("--save", help="write the full run (every step) to this JSON file")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run = run_agent(a.question, Budgets(a.max_iters, a.max_tokens, a.max_cost, a.max_wall),
                    log=lambda s: print(s, flush=True))
    print(json.dumps(run["final"], indent=2, ensure_ascii=False))
    print(f"laps={run['llm_calls']} tokens={run['total_tokens']:,} cost=${run['cost_usd']:.5f} "
          f"latency={run['latency_s']:.2f}s stopped_by={run['stopped_by']}")
    if a.save:
        with open(a.save, "w", encoding="utf-8") as f:
            json.dump(run, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
