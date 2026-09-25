"""Blind hand-labelling of 25 app answers on the judge's single criterion.

    python eval/week6/label.py --run baseline --labeler "Hari"

Shows the question, the reference and the app's answer - nothing else. No judge
verdicts (none may exist yet) and no assertion results, so the label is your own
reading of one question: is the policy this answer gives CORRECT?

Answers are shown in a seeded shuffle so modes are not labelled in blocks. Progress
is saved after every label; re-run with --resume to continue. When all 25 are done,
COMMIT labels_25.json before anything runs the judge (judge.py enforces this).
"""

import argparse
import hashlib
import json
import os
import random
import sys
import textwrap
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(HERE, "labels_25.json")
CASES_PATH = os.path.join(HERE, "cases.jsonl")
N = 25
SEED = 20260923

CRITERION = ("Is the policy this answer gives CORRECT? PASS = every policy fact it states agrees with the "
             "reference and it gives the fact asked for. FAIL = wrong/outdated figure or rule, contradiction, "
             "missing the asked-for fact, or a refusal. Ignore citations, dates, format, tone.")


def now():
    return datetime.now(timezone.utc).isoformat()


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--labeler", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    run_path = os.path.join(HERE, "runs", f"{args.run}.json")
    with open(run_path, encoding="utf-8") as f:
        run = json.load(f)
    if any(r.get("judge") for r in run["rows"]) or run.get("judge_version"):
        sys.exit("This run already has judge verdicts - labelling it now would not be blind.")

    with open(CASES_PATH, encoding="utf-8") as f:
        refs = {c["id"]: c["reference"] for c in map(json.loads, filter(str.strip, f))}

    rows = [r for r in run["rows"] if r["expect"] == "answer"]
    if len(rows) != N:
        sys.exit(f"expected {N} judged (answerable) cases in the run, found {len(rows)}")
    random.Random(SEED).shuffle(rows)

    if os.path.exists(LABELS_PATH):
        if not args.resume:
            sys.exit("labels_25.json already exists. Use --resume to continue; never relabel after the judge ran.")
        with open(LABELS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"criterion": CRITERION, "labeler": args.labeler, "run": args.run,
                "run_file_sha256": file_sha(run_path), "blind": True, "seed": SEED,
                "started_at": now(), "finished_at": None, "labels": []}
    done = {l["id"] for l in data["labels"]}

    print(f"\nCRITERION: {CRITERION}\n")
    print("Keys: p = PASS, f = FAIL, q = save and quit. After the verdict you can type a short note.\n")
    for i, r in enumerate(rows, 1):
        if r["id"] in done:
            continue
        print("=" * 90)
        print(f"[{i}/{N}]  case {r['id']}")
        print(f"\nQUESTION:\n{textwrap.fill(r['question'], 88)}")
        print(f"\nREFERENCE:\n{textwrap.fill(refs[r['id']], 88)}")
        print(f"\nAPP ANSWER:\n{r['answer']}\n")
        while True:
            v = input("verdict [p/f/q]: ").strip().lower()
            if v in ("p", "f", "q"):
                break
        if v == "q":
            break
        note = input("note (optional, enter to skip): ").strip()
        data["labels"].append({"id": r["id"], "label": "PASS" if v == "p" else "FAIL", "note": note,
                               "answer_sha": r["answer_sha"], "labeled_at": now()})
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    if len(data["labels"]) == N:
        data["finished_at"] = data["finished_at"] or now()
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        p = sum(l["label"] == "PASS" for l in data["labels"])
        print(f"\nAll {N} labelled ({p} PASS / {N - p} FAIL). Now COMMIT labels_25.json before running the judge:")
        print("  git add eval/week6/labels_25.json && git commit -m \"week 6: blind hand labels (before judge)\"")
    else:
        print(f"\nSaved {len(data['labels'])}/{N}. Continue with --resume.")


if __name__ == "__main__":
    main()
