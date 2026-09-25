"""Week 6 eval: one command, every case, pass rate by failure mode.

    python eval/week6/run_eval.py --label baseline --judge none   # before labels exist
    python eval/week6/run_eval.py --label baseline_v1 --answers-from baseline --judge v1
    python eval/week6/run_eval.py --label after                   # full run, judge v2

For each case in cases.jsonl: ask the real app (RAGService, in-process, same
Pinecone namespace as the running app), run the deterministic assertions, then
the LLM judge on answerable cases. A case passes only if every applicable
assertion passes AND the judge says PASS. The result table is broken down by the
Week 5 taxonomy mode, never only as one average.

--answers-from LABEL re-scores the frozen answers of an earlier run instead of
asking the app again, which is how the judge is compared v1 vs v2 on exactly the
answers that were hand-labelled.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"), override=True)

from assertions import ASSERTIONS, run_assertions  # noqa: E402
import judge as judge_mod  # noqa: E402

CASES_PATH = os.path.join(HERE, "cases.jsonl")
RUNS_DIR = os.path.join(HERE, "runs")
TRACES_PATH = os.path.join(ROOT, "traces", "traces.jsonl")
MODES = ["contamination", "stale_addendum", "false_refusal", "chunk_severed", "no_attribution"]


def sha(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def load_cases():
    with open(CASES_PATH, encoding="utf-8") as f:
        cases = [json.loads(l) for l in f if l.strip()]
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for c in cases:
        assert c["mode"] in MODES, f"{c['id']}: unknown mode {c['mode']!r}"
    return cases


def check_regressions(cases):
    """Regression cases must be the failed trace's question, verbatim."""
    traces = {}
    with open(TRACES_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                traces[t["trace_id"]] = t["query"]
    for c in cases:
        tid = c.get("regression_of")
        if tid:
            assert tid in traces, f"{c['id']}: trace {tid} not in traces.jsonl"
            assert traces[tid] == c["question"], f"{c['id']}: question differs from trace {tid}"
    return sum(1 for c in cases if c.get("regression_of"))


def _retry_after(err):
    """Seconds Groq asks us to wait ('try again in 7m48.7s'), else a short default."""
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(err))
    return (int(m.group(1) or 0) * 60 + float(m.group(2)) + 5) if m else 10


def ask_app(cases, label):
    """Ask the app every case, checkpointing each answer so a rate-limit stop can --resume."""
    from app.services.rag_service import RAGService
    from app.generation.prompts import RAG_PROMPT_VERSION

    partial = os.path.join(RUNS_DIR, f"{label}.partial.json")
    out = {}
    if os.path.exists(partial):
        with open(partial, encoding="utf-8") as f:
            saved = json.load(f)
        if saved["app_prompt_version"] == RAG_PROMPT_VERSION:
            out = saved["answers"]
            print(f"  resuming: {len(out)} answers already checkpointed")

    rag = RAGService()
    os.makedirs(RUNS_DIR, exist_ok=True)
    for i, c in enumerate(cases, 1):
        if c["id"] in out:
            continue
        for attempt in range(20):
            try:
                res = asyncio.run(rag.answer_query(c["question"]))
                break
            except Exception as e:  # Groq rate limits: wait as long as Groq says
                wait = _retry_after(e)
                print(f"  rate-limited on {c['id']}, waiting {wait:.0f}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"{c['id']}: app failed 20 times - re-run the same command to resume")
        out[c["id"]] = {"answer": res["answer"], "trace_id": res["trace_id"],
                        "sources": [s.get("source") for s in res["sources"]]}
        with open(partial, "w", encoding="utf-8") as f:
            json.dump({"app_prompt_version": RAG_PROMPT_VERSION, "answers": out}, f, indent=2, ensure_ascii=False)
        print(f"  [{i:2}/{len(cases)}] {c['id']} {res['trace_id']}")
        time.sleep(1.0)
    os.remove(partial)
    return out, RAG_PROMPT_VERSION


def frozen_answers(label):
    with open(os.path.join(RUNS_DIR, f"{label}.json"), encoding="utf-8") as f:
        run = json.load(f)
    return ({r["id"]: {"answer": r["answer"], "trace_id": r["trace_id"], "sources": r.get("sources")}
             for r in run["rows"]}, run["app_prompt_version"])


def fmt(num, den):
    return f"{num}/{den}" if den else "-"


def print_table(rows, judged):
    checks = list(ASSERTIONS) + (["judge"] if judged else [])
    by_mode = OrderedDict((m, [r for r in rows if r["mode"] == m]) for m in MODES)
    head = f"{'mode':16} {'n':>3} {'PASS':>11}  " + "  ".join(f"{c:>20}" for c in checks)
    print(head)
    print("-" * len(head))

    def line(name, rs):
        passed = sum(r["pass"] for r in rs)
        cells = []
        for c in checks:
            vals = [r["judge"]["verdict"] == "PASS" for r in rs if r.get("judge")] if c == "judge" \
                else [r["assertions"][c] for r in rs if r["assertions"][c] is not None]
            cells.append(f"{fmt(sum(vals), len(vals)):>20}")
        print(f"{name:16} {len(rs):>3} {fmt(passed, len(rs)):>6} {passed / len(rs):>4.0%}  " + "  ".join(cells))

    for m, rs in by_mode.items():
        if rs:
            line(m, rs)
    print("-" * len(head))
    line("ALL", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="name of this run -> runs/<label>.json")
    ap.add_argument("--judge", default="v2", help="judge prompt version (judge_<v>.txt), or 'none'")
    ap.add_argument("--answers-from", help="re-score the answers of runs/<label>.json instead of asking the app")
    args = ap.parse_args()

    cases = load_cases()
    n_reg = check_regressions(cases)
    print(f"{len(cases)} cases, {n_reg} regression cases verified verbatim against traces.jsonl")

    labels_commit = None
    if args.judge != "none":
        try:
            labels_commit = judge_mod.labels_commit()
        except judge_mod.LabelsNotCommitted as e:
            sys.exit(f"REFUSING TO RUN THE JUDGE: {e}\n(use --judge none to score assertions only)")
        print(f"labels_25.json committed in {labels_commit[0][:10]} at {labels_commit[1]} - judge may run")

    if args.answers_from:
        answers, prompt_version = frozen_answers(args.answers_from)
        print(f"re-scoring frozen answers from runs/{args.answers_from}.json")
    else:
        print("asking the app ...")
        answers, prompt_version = ask_app(cases, args.label)

    rows = []
    for c in cases:
        a = answers[c["id"]]
        row = {"id": c["id"], "mode": c["mode"], "question": c["question"], "expect": c["expect"],
               "regression_of": c.get("regression_of"), "answer": a["answer"], "answer_sha": sha(a["answer"]),
               "trace_id": a["trace_id"], "sources": a.get("sources"),
               "assertions": run_assertions(c, a["answer"]), "judge": None}
        if args.judge != "none" and c["expect"] == "answer":
            row["judge"] = judge_mod.judge(c, a["answer"], args.judge)
            print(f"  judged {c['id']}: {row['judge']['verdict']}")
            time.sleep(1.0)
        ok = all(v for v in row["assertions"].values() if v is not None)
        if row["judge"] is not None:
            ok = ok and row["judge"]["verdict"] == "PASS"
        row["pass"] = ok
        rows.append(row)

    judged = args.judge != "none"
    n_assert = len(ASSERTIONS)
    print()
    print(f"run={args.label}  app_prompt={prompt_version}  judge={args.judge}"
          f"{' (' + judge_mod.JUDGE_MODEL + ')' if judged else ''}")
    print(f"criteria: {n_assert} deterministic assertions, {1 if judged else 0} judged criterion")
    print()
    print_table(rows, judged)

    os.makedirs(RUNS_DIR, exist_ok=True)
    out = {"label": args.label, "created_at": datetime.now(timezone.utc).isoformat(),
           "app_prompt_version": prompt_version, "answers_from": args.answers_from,
           "judge_version": None if not judged else args.judge,
           "judge_model": judge_mod.JUDGE_MODEL if judged else None,
           "labels_commit": labels_commit, "n_assertions": n_assert, "n_judged_criteria": 1 if judged else 0,
           "rows": rows}
    path = os.path.join(RUNS_DIR, f"{args.label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
