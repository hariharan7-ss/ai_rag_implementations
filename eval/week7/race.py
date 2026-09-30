"""Race the agent against the fixed workflow over the same 10 employee questions.

    python eval/week7/race.py                    # 1 trial per question per system -> race.csv
    python eval/week7/race.py --trials 3         # repeat to see how stable the numbers are
    python eval/week7/race.py --extend 1         # add one more trial to runs/race.json, keep the old ones
    python eval/week7/race.py --only q08,q09     # debug a subset (does not overwrite race.csv)

Both systems get the identical question text, model, params, tools and output contract.
Order alternates per question (agent first on odd, workflow first on even). Between runs the
harness waits for Groq's per-minute token bucket to refill, OUTSIDE the timed region; any 429
that still lands inside a run is slept off and subtracted from latency (reported separately).

Tokens are summed over EVERY model call in a run - for the agent, every lap, each of which
re-sends the whole message list - not just the final call.

Pass = contract parses AND status matches AND (value, unit) match AND every must_cite id is in sources
(a must_cite entry that is a list means "any one of these"). Every row is re-graded against the
current questions.jsonl when the tables are printed, so a grading fix applies to all trials alike.
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import OrderedDict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agent.agent import Budgets, run_agent  # noqa: E402
from app.agent.model import MODEL, PARAMS, PRICE_PER_M, wait_for_full_bucket  # noqa: E402
from app.agent.workflow import run_workflow  # noqa: E402

QUESTIONS = os.path.join(HERE, "questions.jsonl")
RUNS_DIR = os.path.join(HERE, "runs")
SYSTEMS = OrderedDict([("agent", lambda q: run_agent(q, log=lambda s: None)),
                       ("workflow", lambda q: run_workflow(q, log=lambda s: None))])
CLASSES = ["single_lookup", "param_branch", "path_branch"]


def grade(case, final):
    exp = case["expect"]
    if final.get("contract_error"):
        return False, "contract_error"
    if final["status"] != exp["status"]:
        return False, "abstained" if final["status"] == "cannot_answer" else "wrong_status"
    if exp["status"] == "answered":
        if final["value"] is None or abs(final["value"] - exp["value"]) > 0.01 or final["unit"] != exp["unit"]:
            return False, "wrong_figure"
        missing = [s if isinstance(s, str) else "|".join(s) for s in exp["must_cite"]
                   if not set([s] if isinstance(s, str) else s) & set(final["sources"])]
        if missing:
            return False, "missing_cite:" + ",".join(missing)
    return True, ""


def summarise(rows, system):
    rs = [r for r in rows if r["system"] == system]
    n_q = len({r["id"] for r in rs})
    return {"system": system, "runs": len(rs), "questions": n_q,
            "pass_rate": sum(r["pass"] for r in rs) / len(rs),
            "p50_latency_s": statistics.median(r["latency_s"] for r in rs),
            "total_tokens": sum(r["total_tokens"] for r in rs) / (len(rs) / n_q),  # per pass over the 10
            "cost_per_question_usd": sum(r["cost_usd"] for r in rs) / len(rs),
            "mean_llm_calls": sum(r["llm_calls"] for r in rs) / len(rs),
            "mean_tool_calls": sum(len(r["tool_calls"]) for r in rs) / len(rs),
            "p95_latency_s": sorted(r["latency_s"] for r in rs)[max(0, round(0.95 * len(rs)) - 1)],
            "rate_limit_wait_s": sum(r["rate_limit_wait_s"] for r in rs)}


def print_tables(rows, trials):
    print(f"\nmodel={MODEL} reasoning_effort={PARAMS['reasoning_effort']} trials={trials} "
          f"price=${PRICE_PER_M['input']}/M in, ${PRICE_PER_M['output']}/M out")
    head = f"{'system':9} {'pass rate':>10} {'p50 latency':>12} {'total tokens':>13} {'cost/question':>14} {'LLM calls/q':>12}"
    print(head + "\n" + "-" * len(head))
    for s in SYSTEMS:
        m = summarise(rows, s)
        passed = f"{round(m['pass_rate'] * m['runs'])}/{m['runs']}"
        print(f"{s:9} {passed:>6} {m['pass_rate']:>3.0%} "
              f"{m['p50_latency_s']:>11.2f}s {m['total_tokens']:>13,.0f} {'$' + format(m['cost_per_question_usd'], '.6f'):>14} "
              f"{m['mean_llm_calls']:>12.1f}")
    print("\nby question class (pass | mean tokens | mean latency):")
    for c in CLASSES:
        cells = []
        for s in SYSTEMS:
            rs = [r for r in rows if r["system"] == s and r["class"] == c]
            if rs:
                cells.append(f"{s} {sum(r['pass'] for r in rs)}/{len(rs)} | {statistics.mean(r['total_tokens'] for r in rs):,.0f} tok"
                             f" | {statistics.mean(r['latency_s'] for r in rs):.2f}s")
        print(f"  {c:14} " + "    ".join(cells))
    print("\nfailures:")
    for r in rows:
        if not r["pass"]:
            f = r["final"]
            print(f"  {r['id']} t{r['trial']} {r['system']:8} {r['fail_reason']:28} got {f.get('value')} {f.get('unit')} "
                  f"sources={f.get('sources')}  tools={r['tool_calls']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--only", help="comma-separated question ids (debug; race.csv is not written)")
    ap.add_argument("--extend", type=int, default=0, help="add this many trials to an existing runs/<label>.json")
    ap.add_argument("--label", default="race")
    a = ap.parse_args()

    with open(QUESTIONS, encoding="utf-8") as f:
        cases = [json.loads(l) for l in f if l.strip()]
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
        a.label = a.label + "_debug"
    print(f"{len(cases)} questions x {len(SYSTEMS)} systems x {a.trials} trials; agent budgets {Budgets()}")

    os.makedirs(RUNS_DIR, exist_ok=True)
    partial = os.path.join(RUNS_DIR, f"{a.label}.partial.json")
    rows = json.load(open(partial, encoding="utf-8")) if os.path.exists(partial) else []
    first_trial = 1
    if a.extend:
        prior = json.load(open(os.path.join(RUNS_DIR, f"{a.label}.json"), encoding="utf-8"))["rows"]
        first_trial = max(r["trial"] for r in prior) + 1
        a.trials = first_trial - 1 + a.extend
        rows = prior + [r for r in rows if r["trial"] >= first_trial]
    done = {(r["id"], r["system"], r["trial"]) for r in rows}
    if rows:
        print(f"resuming: {len(rows)} runs already checkpointed")

    for trial in range(first_trial, a.trials + 1):
        for i, case in enumerate(cases):
            order = list(SYSTEMS) if i % 2 == 0 else list(reversed(SYSTEMS))
            for system in order:
                if (case["id"], system, trial) in done:
                    continue
                wait_for_full_bucket()
                run = SYSTEMS[system](case["question"])
                ok, why = grade(case, run["final"])
                row = {"id": case["id"], "class": case["class"], "trial": trial, "pass": ok, "fail_reason": why, **run}
                rows.append(row)
                with open(partial, "w", encoding="utf-8") as f:
                    json.dump(rows, f, indent=1, ensure_ascii=False, default=str)
                fin = run["final"]
                print(f"t{trial} {case['id']} {system:8} {'PASS' if ok else 'FAIL ' + why:30} "
                      f"{fin.get('value')} {fin.get('unit')}  calls={run['llm_calls']} tok={run['total_tokens']:,} "
                      f"{run['latency_s']:.2f}s" + (f" stopped_by={run['stopped_by']}" if run["stopped_by"] else ""))

    by_id = {c["id"]: c for c in cases}
    for r in rows:
        r["pass"], r["fail_reason"] = grade(by_id[r["id"]], r["final"])
    print_tables(rows, a.trials)

    out = {"label": a.label, "created_at": datetime.now(timezone.utc).isoformat(), "model": MODEL, "params": PARAMS,
           "price_per_m": PRICE_PER_M, "trials": a.trials, "agent_budgets": vars(Budgets()),
           "summary": [summarise(rows, s) for s in SYSTEMS], "rows": rows}
    with open(os.path.join(RUNS_DIR, f"{a.label}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    os.remove(partial)
    if a.only:
        return

    with open(os.path.join(HERE, "race.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "questions", "runs", "pass_rate", "p50_latency_s", "total_tokens",
                    "cost_per_question_usd", "mean_llm_calls", "p95_latency_s"])
        for s in SYSTEMS:
            m = summarise(rows, s)
            w.writerow([s, m["questions"], m["runs"], f"{m['pass_rate']:.2f}", f"{m['p50_latency_s']:.2f}",
                        f"{m['total_tokens']:.0f}", f"{m['cost_per_question_usd']:.6f}",
                        f"{m['mean_llm_calls']:.1f}", f"{m['p95_latency_s']:.2f}"])
    with open(os.path.join(HERE, "race_by_question.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "class", "trial", "system", "pass", "fail_reason", "value", "unit", "sources",
                    "llm_calls", "tool_calls", "total_tokens", "cost_usd", "latency_s", "rate_limit_wait_s"])
        for r in sorted(rows, key=lambda r: (r["id"], r["trial"], r["system"])):
            fin = r["final"]
            w.writerow([r["id"], r["class"], r["trial"], r["system"], r["pass"], r["fail_reason"], fin.get("value"),
                        fin.get("unit"), " ".join(fin.get("sources") or []), r["llm_calls"], " ".join(r["tool_calls"]),
                        r["total_tokens"], f"{r['cost_usd']:.6f}", f"{r['latency_s']:.2f}", f"{r['rate_limit_wait_s']:.1f}"])
    print(f"\nwrote eval/week7/race.csv, race_by_question.csv, runs/{a.label}.json")


if __name__ == "__main__":
    main()
