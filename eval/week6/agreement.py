"""Judge vs human agreement on the 25 hand-labelled answers.

    python eval/week6/agreement.py --run baseline_judge_v1
    python eval/week6/agreement.py --run baseline_judge_v2 --exclude f02,s01

Agreement = judge verdict == human label, over the labelled answers. The judge run
must have scored the SAME answers that were labelled (checked by answer hash), which
is why it is made with run_eval.py --answers-from <labelled run>.

--exclude drops cases from the count: use it for the disagreements that were copied
into judge_v2 as few-shot examples, since the judge has now seen their answers.
"""

import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--exclude", default="", help="comma-separated case ids to leave out")
    args = ap.parse_args()
    exclude = {x for x in args.exclude.split(",") if x}

    with open(os.path.join(HERE, "labels_25.json"), encoding="utf-8") as f:
        labels = {l["id"]: l for l in json.load(f)["labels"]}
    with open(os.path.join(HERE, "runs", f"{args.run}.json"), encoding="utf-8") as f:
        run = json.load(f)
    rows = {r["id"]: r for r in run["rows"]}

    pairs = []
    for cid, lab in labels.items():
        if cid in exclude:
            continue
        r = rows[cid]
        if r["answer_sha"] != lab["answer_sha"]:
            sys.exit(f"{cid}: run answer differs from the labelled answer - use --answers-from the labelled run")
        pairs.append((cid, lab["label"], r["judge"]["verdict"], r))

    n = len(pairs)
    agree = sum(h == j for _, h, j, _ in pairs)
    cm = {(h, j): 0 for h in ("PASS", "FAIL") for j in ("PASS", "FAIL", None)}
    for _, h, j, _ in pairs:
        cm[(h, j)] += 1

    print(f"run={args.run}  judge={run['judge_version']} ({run['judge_model']})  "
          f"labels committed {run['labels_commit'][0][:10]} at {run['labels_commit'][1]}")
    if exclude:
        print(f"excluded (few-shot examples): {', '.join(sorted(exclude))}")
    print(f"\nagreement = {agree}/{n} = {agree / n:.1%}\n")
    print(f"{'':14}{'judge PASS':>12}{'judge FAIL':>12}{'no verdict':>12}")
    for h in ("PASS", "FAIL"):
        print(f"human {h:8}{cm[(h, 'PASS')]:>12}{cm[(h, 'FAIL')]:>12}{cm[(h, None)]:>12}")

    dis = [(cid, h, j, r) for cid, h, j, r in pairs if h != j]
    print(f"\n{len(dis)} disagreement(s):")
    for cid, h, j, r in dis:
        print("-" * 90)
        print(f"{cid} [{r['mode']}]  human={h}  judge={j}")
        print(f"Q: {r['question']}")
        print(f"A: {r['answer'][:600]}")
        print(f"judge reason: {r['judge']['reason'][:400]}")
        if labels[cid].get("note"):
            print(f"human note: {labels[cid]['note']}")


if __name__ == "__main__":
    main()
