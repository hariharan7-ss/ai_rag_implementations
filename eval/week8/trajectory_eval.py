"""Week 8: trajectory eval for the HR agent - score the path, not just the answer.

    python eval/week8/trajectory_eval.py --run before --trials 3    # run the agent 3x over the 10 questions
    python eval/week8/trajectory_eval.py --score before             # score trajectories (offline, no API)
    python eval/week8/trajectory_eval.py --score after --compare before   # per-mode regression + price table

Expected tool sequences live in expected_paths.json: every acceptable core path per case,
alternate paths asserted as a set, handbook-search position never asserted (see its _doc).

Four trajectory numbers per run set:
  tool-choice accuracy   runs whose call sequence matches an accepted path
  argument validity      executed calls whose arguments are real AND grounded - an employee id
                         must exist and come from the question or an earlier result, a jurisdiction
                         must come from a fetched record, not the model's prior
  step efficiency        steps taken / steps needed (shortest accepted path), mean + max
  cost per question      p50 AND max (the mean hides the run that spun)

Failure-mode taxonomy (per-run counts, the Week-8 zoo):
  rejected_tool_call  lap rejected by the API (malformed tool name/args) - no tool ran
  wrong_path          call sequence matches no accepted path (missing / unexpected / misordered R,J)
  ungrounded_args     an executed call whose arguments were fluent fiction (see above)
  duplicate_call      the exact same call (tool + arguments) issued twice
  budget_hit          a Week-7 budget terminated the run

Trajectory PASS = no rejected laps, accepted path, all args grounded, no budget hit.
(duplicate_call alone does not fail: a repeated H is waste, charged to step efficiency;
a repeated R/J already fails wrong_path.) Outcome PASS = the Week-7 race grader. The
outcome-vs-trajectory gap is outcome rate minus trajectory rate over the same runs.
"""

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval", "week7"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from race import grade  # noqa: E402  (Week-7 outcome grader, unchanged)
from app.agent.tools import JURISDICTIONS, TOPICS  # noqa: E402

QUESTIONS = os.path.join(ROOT, "eval", "week7", "questions.jsonl")
RUNS_DIR = os.path.join(HERE, "runs")
MODES = ["rejected_tool_call", "wrong_path", "ungrounded_args", "duplicate_call", "budget_hit"]
EID_RE = re.compile(r"\bE\d{4}\b")


def load_cases():
    with open(QUESTIONS, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_paths():
    with open(os.path.join(HERE, "expected_paths.json"), encoding="utf-8") as f:
        return json.load(f)["cases"]


def known_employees():
    with open(os.path.join(ROOT, "data", "hr_agent", "employees.json"), encoding="utf-8") as f:
        return {r["employee_id"] for r in json.load(f)["records"]}


def canon(name, args):
    if name == "get_employee_record":
        return f"R:{args.get('employee_id')}"
    if name == "get_jurisdiction_rules":
        return f"J:{args.get('jurisdiction')}:{args.get('topic')}"
    return "H"


def flatten_calls(run):
    """Executed tool calls in order, plus the count of API-rejected laps."""
    calls, rejected = [], 0
    for step in run["steps"]:
        if "error" in step:
            rejected += 1
            continue
        for tc in step.get("tool_calls", []):
            calls.append(tc)
    return calls, rejected


def annotate(run, case, spec, employees):
    """Attach trajectory verdicts and mode counts to one run (pure function of the saved steps)."""
    calls, rejected = flatten_calls(run)
    tokens = [canon(tc["name"], tc["args"] if isinstance(tc["args"], dict) else {}) for tc in calls]

    # --- argument validity: real AND grounded in the question or an earlier result
    grounded_ids = set(EID_RE.findall(case["question"]))
    grounded_jurs = {j for j in JURISDICTIONS if j in case["question"]}
    arg_flags = []
    for tc in calls:
        args = tc["args"] if isinstance(tc["args"], dict) else {}
        if tc["name"] == "get_employee_record":
            eid = str(args.get("employee_id", ""))
            ok = bool(re.fullmatch(r"E\d{4}", eid)) and eid in employees and eid in grounded_ids
        elif tc["name"] == "get_jurisdiction_rules":
            ok = (args.get("jurisdiction") in JURISDICTIONS and args.get("topic") in TOPICS
                  and args.get("jurisdiction") in grounded_jurs)
        else:
            ok = bool(str(args.get("query", "")).strip())
        arg_flags.append(ok)
        dump = json.dumps(tc.get("result", {}), ensure_ascii=False)
        grounded_ids |= set(EID_RE.findall(dump))
        grounded_jurs |= {j for j in JURISDICTIONS if f'"{j}"' in dump or f"-{j}-" in dump}

    # --- sequence match against the accepted path set (H position/surplus never asserted)
    non_h = [t for t in tokens if t != "H"]
    h_count = len(tokens) - len(non_h)
    matched = any(non_h == p["core"] and h_count >= p["h_min"] for p in spec["paths"])
    steps_needed = min(len(p["core"]) + p["h_min"] for p in spec["paths"])
    steps_taken = len(tokens)

    sig = [(tc["name"], json.dumps(tc["args"], sort_keys=True) if isinstance(tc["args"], dict) else str(tc["args"]))
           for tc in calls]
    modes = {
        "rejected_tool_call": rejected,
        "wrong_path": int(not matched),
        "ungrounded_args": sum(not f for f in arg_flags),
        "duplicate_call": len(sig) - len(set(sig)),
        "budget_hit": int(bool(run.get("stopped_by"))),
    }
    outcome_pass, fail_reason = grade(case, run["final"])
    traj_pass = (modes["rejected_tool_call"] == 0 and modes["wrong_path"] == 0
                 and modes["ungrounded_args"] == 0 and modes["budget_hit"] == 0)
    run.update(tokens_seq=tokens, matched=matched, steps_taken=steps_taken, steps_needed=steps_needed,
               step_efficiency=(steps_taken / steps_needed) if steps_taken else 0.0,
               arg_calls=len(arg_flags), arg_valid=sum(arg_flags), modes=modes,
               outcome_pass=outcome_pass, outcome_fail_reason=fail_reason, traj_pass=traj_pass)
    return run


def summarise(rows):
    costs = sorted(r["cost_usd"] for r in rows)
    eff = [r["step_efficiency"] for r in rows if r["steps_taken"]]
    total_calls = sum(r["arg_calls"] for r in rows)
    out = {
        "runs": len(rows),
        "tool_choice_accuracy": sum(r["matched"] for r in rows) / len(rows),
        "arg_validity": (sum(r["arg_valid"] for r in rows) / total_calls) if total_calls else 1.0,
        "step_efficiency_mean": statistics.mean(eff),
        "step_efficiency_max": max(eff),
        "cost_p50": statistics.median(costs),
        "cost_max": costs[-1],
        "tokens_p50": statistics.median(r["total_tokens"] for r in rows),
        "tokens_max": max(r["total_tokens"] for r in rows),
        "latency_p50": statistics.median(r["latency_s"] for r in rows),
        "latency_max": max(r["latency_s"] for r in rows),
        "outcome_rate": sum(r["outcome_pass"] for r in rows) / len(rows),
        "traj_rate": sum(r["traj_pass"] for r in rows) / len(rows),
        "mode_counts": {m: sum(r["modes"][m] for r in rows) for m in MODES},
        "mode_runs": {m: sum(1 for r in rows if r["modes"][m]) for m in MODES},
    }
    out["gap"] = out["outcome_rate"] - out["traj_rate"]
    return out


def print_report(label, rows):
    s = summarise(rows)
    print(f"\n=== {label}: {s['runs']} runs ({len({r['id'] for r in rows})} questions x "
          f"{len({r['trial'] for r in rows})} trials) ===")
    print(f"tool-choice accuracy   {s['tool_choice_accuracy']:>7.0%}   (sequence matches an accepted path)")
    print(f"argument validity      {s['arg_validity']:>7.1%}   ({sum(r['arg_valid'] for r in rows)}/"
          f"{sum(r['arg_calls'] for r in rows)} executed calls grounded)")
    print(f"step efficiency        {s['step_efficiency_mean']:>7.2f}   mean (max {s['step_efficiency_max']:.2f}) - steps taken / steps needed")
    print(f"cost per question      p50 ${s['cost_p50']:.6f}   max ${s['cost_max']:.6f}")
    print(f"tokens per question    p50 {s['tokens_p50']:,.0f}   max {s['tokens_max']:,}")
    print(f"latency per question   p50 {s['latency_p50']:.2f}s   max {s['latency_max']:.2f}s")
    print(f"\noutcome pass rate      {s['outcome_rate']:>7.0%}")
    print(f"trajectory pass rate   {s['traj_rate']:>7.0%}")
    print(f"OUTCOME-VS-TRAJECTORY GAP  {s['gap']:+.0%}  ({s['gap'] * s['runs']:+.0f} runs)")
    print("\nper-mode counts (total occurrences | runs affected):")
    for m in MODES:
        print(f"  {m:20} {s['mode_counts'][m]:>4} | {s['mode_runs'][m]:>2}")
    print("\nper-run detail:")
    for r in sorted(rows, key=lambda r: (r["trial"], r["id"])):
        flags = " ".join(f"{m}={c}" for m, c in r["modes"].items() if c)
        print(f"  t{r['trial']} {r['id']} out={'P' if r['outcome_pass'] else 'F'} traj={'P' if r['traj_pass'] else 'F'} "
              f"eff={r['step_efficiency']:.2f} ${r['cost_usd']:.6f}  [{' '.join(r['tokens_seq'])}]"
              + (f"  {flags}" if flags else ""))
    gap_rows = [r for r in rows if r["outcome_pass"] and not r["traj_pass"]]
    if gap_rows:
        print("\nright answer, wrong path (outcome PASS, trajectory FAIL):")
        for r in gap_rows:
            why = ", ".join(f"{m}x{c}" for m, c in r["modes"].items() if c and m != "duplicate_call")
            print(f"  t{r['trial']} {r['id']}: answered {r['final'].get('value')} {r['final'].get('unit')} "
                  f"correctly via [{' '.join(r['tokens_seq'])}] with {why}")
    return s


def write_csv(label, rows):
    import csv
    path = os.path.join(HERE, f"trajectory_results_{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "class", "trial", "outcome_pass", "traj_pass", "sequence", "matched",
                    "steps_taken", "steps_needed", "step_efficiency", "arg_valid", "arg_calls",
                    *MODES, "total_tokens", "cost_usd", "latency_s", "stopped_by"])
        for r in sorted(rows, key=lambda r: (r["id"], r["trial"])):
            w.writerow([r["id"], r["class"], r["trial"], r["outcome_pass"], r["traj_pass"],
                        " ".join(r["tokens_seq"]), r["matched"], r["steps_taken"], r["steps_needed"],
                        f"{r['step_efficiency']:.2f}", r["arg_valid"], r["arg_calls"],
                        *[r["modes"][m] for m in MODES],
                        r["total_tokens"], f"{r['cost_usd']:.6f}", f"{r['latency_s']:.2f}", r.get("stopped_by") or ""])
    print(f"\nwrote {os.path.relpath(path, ROOT)}")


def print_compare(before_s, after_s):
    print("\n=== regression check: per-mode counts, before -> after ===")
    print(f"{'mode':22} {'before':>10} {'after':>10}   verdict")
    for m in MODES:
        b, a = before_s["mode_counts"][m], after_s["mode_counts"][m]
        verdict = "improved" if a < b else ("WORSE" if a > b else ("unchanged" if b else "still zero"))
        print(f"{m:22} {b:>10} {a:>10}   {verdict}")
    print("\n=== price paid (after - before) ===")
    for key, fmt, unit in [("cost_p50", "+.6f", "$/question p50"), ("cost_max", "+.6f", "$/question max"),
                           ("tokens_p50", "+,.0f", "tokens p50"), ("tokens_max", "+,.0f", "tokens max"),
                           ("latency_p50", "+.2f", "s latency p50"), ("latency_max", "+.2f", "s latency max"),
                           ("step_efficiency_mean", "+.2f", "step efficiency mean")]:
        print(f"  {format(after_s[key] - before_s[key], fmt):>12} {unit}")
    for key in ("tool_choice_accuracy", "arg_validity", "outcome_rate", "traj_rate", "gap"):
        print(f"  {after_s[key] - before_s[key]:>+11.0%} {key}")


def do_runs(label, trials):
    from app.agent.agent import run_agent
    from app.agent.model import MODEL, PARAMS, wait_for_full_bucket

    cases = load_cases()
    os.makedirs(RUNS_DIR, exist_ok=True)
    partial = os.path.join(RUNS_DIR, f"{label}.partial.json")
    rows = json.load(open(partial, encoding="utf-8")) if os.path.exists(partial) else []
    done = {(r["id"], r["trial"]) for r in rows}
    if rows:
        print(f"resuming: {len(rows)} runs already checkpointed")
    for trial in range(1, trials + 1):
        for case in cases:
            if (case["id"], trial) in done:
                continue
            wait_for_full_bucket()
            run = run_agent(case["question"], log=lambda s: None)
            rows.append({"id": case["id"], "class": case["class"], "trial": trial, **run})
            with open(partial, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=1, ensure_ascii=False, default=str)
            fin = run["final"]
            print(f"t{trial} {case['id']}: {fin.get('value')} {fin.get('unit')}  laps={run['llm_calls']} "
                  f"tok={run['total_tokens']:,} {run['latency_s']:.2f}s"
                  + (f" stopped_by={run['stopped_by']}" if run["stopped_by"] else ""), flush=True)
    out = {"label": label, "created_at": datetime.now(timezone.utc).isoformat(),
           "model": MODEL, "params": PARAMS, "trials": trials, "rows": rows}
    with open(os.path.join(RUNS_DIR, f"{label}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    os.remove(partial)
    print(f"wrote eval/week8/runs/{label}.json")


def score(label):
    with open(os.path.join(RUNS_DIR, f"{label}.json"), encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    cases = {c["id"]: c for c in load_cases()}
    paths = load_paths()
    employees = known_employees()
    return [annotate(r, cases[r["id"]], paths[r["id"]], employees) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", metavar="LABEL", help="run the agent over the 10 questions -> runs/<LABEL>.json")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--score", metavar="LABEL", help="score a saved run set (offline)")
    ap.add_argument("--compare", metavar="LABEL", help="with --score: baseline to diff modes/price against")
    a = ap.parse_args()
    if a.run:
        do_runs(a.run, a.trials)
    if a.score:
        rows = score(a.score)
        s = print_report(a.score, rows)
        write_csv(a.score, rows)
        if a.compare:
            print_compare(summarise(score(a.compare)), s)
    if not a.run and not a.score:
        ap.error("nothing to do: pass --run and/or --score")


if __name__ == "__main__":
    main()
