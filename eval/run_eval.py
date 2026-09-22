"""Measure hit-rate@3 and retrieval latency over the golden set.

Writes eval/runs/<label>.json, the run file inspect_view.py and compare.py read.
Run it once before the retrieval change and once after, with different labels.
"""

import argparse
import json
import os
import statistics
import time

import common  # noqa: F401  (must come first: sets sys.path + PINECONE_NAMESPACE)

from app.retrieval.vector_search import perform_vector_search
from app.services.retrieval_service import retrieval_service

DENSE_TOP_K = 25
QUESTION_WIDTH = 60


def emit(line):
    # The Windows console is cp1252. A question id, a --label or a repo path
    # carrying one non-ASCII character would otherwise raise UnicodeEncodeError
    # mid-run, after the retrieval calls have already been paid for.
    print(line.encode("ascii", "replace").decode("ascii"))


def load_golden():
    if not os.path.exists(common.GOLDEN_PATH):
        raise SystemExit(
            f"No golden set at {common.GOLDEN_PATH}\n"
            "Build it first -- see eval/README.md."
        )
    questions = common.load_jsonl(common.GOLDEN_PATH)
    if not questions:
        raise SystemExit(
            f"Golden set at {common.GOLDEN_PATH} is empty\n"
            "Build it first -- see eval/README.md."
        )
    return questions


def chunk_ids(results):
    # .get the whole way down: a vector upserted without chunk_id metadata
    # should cost one row its score, not abort the run.
    ids = []
    for result in results:
        payload = result.get("payload") or {}
        ids.append(payload.get("chunk_id"))
    return ids


def time_retrieve(query, repeats):
    """Median retrieve() latency in ms, plus the results of the last call."""
    timings = []
    results = []
    for _ in range(repeats):
        start = time.perf_counter()
        # No initial_k/final_k here: the run should measure the retriever the
        # app is actually configured with, before and after the one change.
        results = retrieval_service.retrieve(query)
        timings.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(timings), results


def short_question(text):
    flat = " ".join((text or "").split())
    if len(flat) > QUESTION_WIDTH:
        flat = flat[: QUESTION_WIDTH - 3] + "..."
    return flat


def fmt_rank(rank):
    return str(rank) if rank else "-"


def score_of(result):
    score = result.get("score")
    return round(score, 4) if isinstance(score, (int, float)) else score


def main():
    parser = argparse.ArgumentParser(
        description="Score the golden set and write eval/runs/<label>.json"
    )
    parser.add_argument("--label", required=True, help="names the run file, e.g. baseline")
    parser.add_argument("--repeats", type=int, default=5, help="timed calls per question")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    # The label becomes a filename; a slash in it would write outside runs/.
    if os.path.basename(args.label) != args.label or not args.label:
        parser.error("--label must be a plain name, no path separators")

    questions = load_golden()
    emit(
        f"run_eval label={args.label} namespace={common.EVAL_NAMESPACE} "
        f"n={len(questions)} repeats={args.repeats}"
    )

    # Warm-up, result discarded: the first call pays for the SentenceTransformer
    # load and the first Pinecone round trip, which would otherwise land entirely
    # in q01's latency and make question one look like the slow question.
    retrieval_service.retrieve(questions[0].get("question", ""))

    rows = []
    for i, q in enumerate(questions, 1):
        qid = q.get("id") or f"q{i:02d}"
        question = q.get("question", "")
        gold = common.gold_ids(q)

        latency_ms, results = time_retrieve(question, args.repeats)
        ids = chunk_ids(results)
        rank = common.rank_of(ids, gold)
        hit = rank is not None and rank <= 3
        top3 = [
            {"chunk_id": cid, "score": score_of(result)}
            for cid, result in zip(ids[:3], results[:3])
        ]

        # Untimed, one extra call: this is the evidence for the retrieval change.
        # If the gold chunk is not in the dense top-25, a cross-encoder reranking
        # that top-25 can never surface it, and BM25+RRF is the better bet.
        dense = perform_vector_search(question, top_k=DENSE_TOP_K)
        dense_rank = common.rank_of(chunk_ids(dense), gold)

        if "exact_token" in q:
            exact_token = bool(q["exact_token"])
        else:
            exact_token = bool(common.exact_tokens(question))

        rows.append({
            "id": qid,
            "question": question,
            "gold": gold,
            "exact_token": exact_token,
            "hit": hit,
            "rank": rank,
            "dense_rank_top25": dense_rank,
            "top3": top3,
            "latency_ms": round(latency_ms, 2),
        })

        flag = "HIT " if hit else "MISS"
        emit(
            f"[{flag}] {qid:<4} rank={fmt_rank(rank):<2} "
            f"dense25={fmt_rank(dense_rank):<3} {latency_ms:7.1f}ms  "
            f"{short_question(question)}"
        )

    hits = sum(1 for row in rows if row["hit"])
    summary = {
        "n": len(rows),
        "hits": hits,
        "hit_rate_at_3": round(hits / len(rows), 3),
        "p50_latency_ms": round(statistics.median([row["latency_ms"] for row in rows]), 2),
        "repeats": args.repeats,
    }

    os.makedirs(common.RUNS_DIR, exist_ok=True)
    out_path = os.path.join(common.RUNS_DIR, f"{args.label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label": args.label,
                "namespace": common.EVAL_NAMESPACE,
                "summary": summary,
                "rows": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    emit(
        f"\n{args.label}: hit-rate@3 {hits}/{summary['n']} = {summary['hit_rate_at_3']:.3f}"
        f" | p50 latency {summary['p50_latency_ms']} ms | repeats={args.repeats}"
    )
    emit(f"wrote {out_path}")


if __name__ == "__main__":
    main()
