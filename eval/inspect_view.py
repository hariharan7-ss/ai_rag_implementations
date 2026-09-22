"""Inspection view for labelling every eval question R / G / Not-In-Corpus.

Reads a run file written by run_eval.py for the recorded ranks, then re-runs
retrieval and the real RAG answer so each judgement can be made from one page:
was the gold chunk reachable at all, and did the LLM fumble a chunk it was
already handed?

    python eval/inspect_view.py --label baseline [--misses]
"""

import argparse
import asyncio
import json
import os
import re

import common  # noqa: F401  (must come first: sets sys.path + PINECONE_NAMESPACE)

from app.services.rag_service import RAGService
from app.services.retrieval_service import retrieval_service

SNIPPET_CHARS = 160

LABEL_LINE = "label: ___  (R / G / Not-In-Corpus / correct)   evidence: ___"

HIT_HINT = (
    "gold IS in top-3: retrieval did its job. If the answer below is wrong, "
    "this is G (generation)."
)
MISS_HINT = (
    "gold NOT in top-3: R candidate, or Not-In-Corpus. Check the gold chunk "
    "text below actually answers the question - if it does not, the label is "
    "Not-In-Corpus, not R."
)


def collapse(text):
    return re.sub(r"\s+", " ", text or "").strip()


def cell(text):
    """One markdown table cell: single line, pipes escaped so the table holds."""
    return collapse(text).replace("|", "\\|")


def snippet(text, limit=SNIPPET_CHARS):
    flat = cell(text)
    if len(flat) > limit:
        flat = flat[:limit].rstrip() + "..."
    return flat


def console(text):
    # The Windows console is cp1252 and the corpus is not ASCII.
    return (text or "").encode("ascii", "replace").decode("ascii")


def blockquote(text):
    lines = [line.rstrip() for line in (text or "").splitlines()]
    return ["> " + line if line else ">" for line in lines]


def fmt_rank(rank):
    return str(rank) if rank else "-"


def fmt_dense(rank):
    return str(rank) if rank else "not in top 25"


def fmt_score(score):
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return "-"


def triage_hint(hit, dense_rank):
    """The labelling rule spelled out per question, so the sheet is self-contained."""
    if hit:
        return HIT_HINT
    if dense_rank:
        reach = (
            f" Gold sits at rank {dense_rank} of the dense top-25, so a top-25 "
            "reranker could still reach it."
        )
    else:
        reach = (
            " Gold is not in the dense top-25 at all, so a reranker cannot help "
            "here - only recall (BM25) can."
        )
    return MISS_HINT + reach


def load_run(label):
    path = os.path.join(common.RUNS_DIR, label + ".json")
    if not os.path.exists(path):
        raise SystemExit(
            console(f"No run file at {path} -- run run_eval.py --label {label} first.")
        )
    with open(path, "r", encoding="utf-8") as f:
        try:
            return path, json.load(f)
        except ValueError as exc:
            raise SystemExit(console(f"{path} is not valid JSON ({exc})"))


def load_chunk_index():
    """chunk_id -> chunk row, or None when there is no chunks.jsonl at all.

    None and {} have to stay distinguishable: "no corpus index here" and "the
    golden set points at an id the corpus does not have" need different advice.
    """
    if not os.path.exists(common.CHUNKS_PATH):
        print(console(f"WARNING: no chunks at {common.CHUNKS_PATH} -- gold text unavailable."))
        return None
    return {row.get("chunk_id"): row for row in common.load_jsonl(common.CHUNKS_PATH)}


def load_golden_index():
    """question id -> golden row, keyed exactly the way run_eval.py keys them.

    The positional fallback has to match run_eval.py's: a golden row written
    without an `id` is scored as q01/q02/... there, and would be keyed None here,
    so its expected_answer would never be found again.
    """
    if not os.path.exists(common.GOLDEN_PATH):
        print(console(f"WARNING: no golden set at {common.GOLDEN_PATH} -- no expected answers."))
        return {}
    rows = common.load_jsonl(common.GOLDEN_PATH)
    return {(row.get("id") or f"q{i:02d}"): row for i, row in enumerate(rows, 1)}


def matched_tokens(tokens, text):
    """Which of the question's exact tokens literally appear in a chunk.

    The most useful column in the table: a question whose policy number sits in
    a chunk dense retrieval ranked low is a lexical-recall failure (BM25), not a
    ranking failure (reranker).
    """
    flat = collapse(text).lower()
    return [t for t in tokens if t.lower() in flat]


def retrieved_rows(question, gold, tokens):
    rows = []
    for i, result in enumerate(retrieval_service.retrieve(question), 1):
        payload = result.get("payload") or {}
        chunk_id = payload.get("chunk_id") or "(no chunk_id)"
        text = payload.get("text", "")
        rows.append(
            {
                "rank": i,
                "chunk_id": chunk_id,
                "is_gold": chunk_id in gold,
                "score": result.get("score"),
                "in_top3": i <= 3,
                "exact_tokens": matched_tokens(tokens, text),
                "snippet": snippet(text),
            }
        )
    return rows


def app_answer(question):
    """The app's real answer, which is what separates an R from a G."""
    try:
        result = asyncio.run(RAGService().answer_query(question))
        return result.get("answer", ""), None
    except Exception as e:  # one flaky Groq call must not kill the whole view
        return "", f"{type(e).__name__}: {e}"


def build_row(run_row, chunks, golden):
    question = run_row.get("question", "")
    gold = run_row.get("gold") or []
    tokens = common.exact_tokens(question)
    hit = bool(run_row.get("hit"))
    dense_rank = run_row.get("dense_rank_top25")
    golden_row = golden.get(run_row.get("id")) or {}

    gold_chunks = []
    for chunk_id in gold:
        chunk = (chunks or {}).get(chunk_id)
        gold_chunks.append(
            {
                "chunk_id": chunk_id,
                "found": chunk is not None,
                "no_chunk_index": chunks is None,
                "text": (chunk or {}).get("text", ""),
            }
        )

    answer, answer_error = app_answer(question)
    return {
        "id": run_row.get("id"),
        "question": question,
        "gold": gold,
        "gold_chunks": gold_chunks,
        "rank": run_row.get("rank"),
        "dense_rank_top25": dense_rank,
        "hit": hit,
        "exact_tokens": tokens,
        "exact_token": bool(run_row.get("exact_token")),
        "triage_hint": triage_hint(hit, dense_rank),
        "expected_answer": golden_row.get("expected_answer"),
        "retrieved": retrieved_rows(question, gold, tokens),
        "answer": answer,
        "answer_error": answer_error,
        "label": "",
        "evidence": "",
    }


def render_table(rows):
    out = [
        "| rank | chunk_id | score | in top-3 | exact tokens | snippet |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    if not rows:
        out.append("| - | (retriever returned nothing) | - | - | - | - |")
        return out
    for row in rows:
        marker = " (gold)" if row["is_gold"] else ""
        out.append(
            "| {rank} | `{chunk_id}`{marker} | {score} | {top3} | {tokens} | {snippet} |".format(
                rank=row["rank"],
                chunk_id=cell(row["chunk_id"]),
                marker=marker,
                score=fmt_score(row["score"]),
                top3="yes" if row["in_top3"] else "no",
                tokens=cell(", ".join(row["exact_tokens"])) or "-",
                snippet=row["snippet"] or "-",
            )
        )
    return out


def render_row(row):
    gold_ids = ", ".join(f"`{g}`" for g in row["gold"]) or "-"
    tokens = ", ".join(f"`{t}`" for t in row["exact_tokens"]) or "-"
    # exact_token in the run file can be set by hand in the golden row, so it can
    # disagree with what EXACT_TOKEN_RE finds here. Say so, rather than showing a
    # bare "-" for a question compare.py prints as an exact-token question.
    if row["exact_token"] and not row["exact_tokens"]:
        tokens += " (golden row is tagged exact_token, but the regex matched none)"

    out = [f"## {row['id']} - {collapse(row['question'])}", ""]
    out.append(f"- gold chunk id(s): {gold_ids}")
    out.append(f"- rank in retriever output: {fmt_rank(row['rank'])}")
    out.append(f"- rank in dense top-25: {fmt_dense(row['dense_rank_top25'])}")
    out.append(f"- exact tokens in question: {tokens}")
    out.append(f"- triage hint: {row['triage_hint']}")
    if row["expected_answer"]:
        out.append(f"- expected answer: {collapse(row['expected_answer'])}")
    out.append("")

    if not row["gold_chunks"]:
        # Nothing to retrieve means the question can never hit; counting it as an
        # R miss would inflate the tally that picks the retrieval change.
        out.append("> **NO GOLD CHUNK ID** on this question, so it can never be a hit.")
        out.append("> Fix `gold_chunk_id` / `gold_chunk_ids` in the golden row and re-run.")
        out.append("")

    for gold_chunk in row["gold_chunks"]:
        out.append(f"**Gold chunk** `{gold_chunk['chunk_id']}`")
        out.append("")
        if gold_chunk["found"]:
            out.extend(blockquote(gold_chunk["text"]) or ["> (chunk has no text)"])
        elif gold_chunk["no_chunk_index"]:
            out.append("> **No chunks.jsonl** -- there is no gold text to show. Run")
            out.append("> `python eval/ingest_hr.py` (it writes chunks.jsonl) and view again;")
            out.append("> this says nothing about whether the gold id is still valid.")
        else:
            out.append("> **STALE GOLD ID -- NOT IN chunks.jsonl.** This question points at a")
            out.append("> chunk id the current corpus does not contain, so its rank is")
            out.append("> meaningless. Re-tag the golden row against a real chunk id (or")
            out.append("> re-ingest the corpus) before counting this question at all.")
        out.append("")

    out.append("**Retrieved now** (retrieval_service.retrieve)")
    out.append("")
    out.extend(render_table(row["retrieved"]))
    out.append("")

    out.append("**App answer** (RAGService.answer_query)")
    out.append("")
    if row["answer_error"]:
        out.append(f"> ANSWER FAILED: {collapse(row['answer_error'])}")
    else:
        out.extend(blockquote(row["answer"]) or ["> (empty answer)"])
    out.append("")

    out.append(LABEL_LINE)
    out.append("")
    return out


def render_header(label, namespace, shown, total, misses_only, run_path):
    scope = f"{shown} of {total} ({'misses only' if misses_only else 'all questions'})"
    ns_line = f"- namespace: `{namespace}`"
    if namespace != common.EVAL_NAMESPACE:
        ns_line += f" -- but the live re-run below queried `{common.EVAL_NAMESPACE}`"
    return [
        f"# Inspection view - {label}",
        "",
        f"- run file: `{run_path}`",
        ns_line,
        f"- questions in this view: {scope}",
        "- ranks come from the run file; the retrieved table and the answer are a live re-run",
        "",
        "**What the labels mean**",
        "",
        "- **R** (retrieval): the gold chunk does answer the question but is not in the top-3.",
        "- **G** (generation): the gold chunk IS in the top-3 and the answer is still wrong.",
        "- **Not-In-Corpus**: no chunk actually answers the question -- re-tag or drop it,",
        "  do not count it as a retrieval failure.",
        "- **correct**: the answer is right; nothing to fix.",
        "",
        "For an R, the dense top-25 rank decides the remedy: in the top-25 means a reranker",
        "over the dense top-25 could reach the gold chunk; missing from the top-25 means only",
        "better recall (BM25 + RRF) can.",
        "",
        "Fill in the `label:` line under each question.",
        "",
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Inspection view for labelling eval misses R / G / Not-In-Corpus."
    )
    parser.add_argument("--label", required=True, help="run label; reads eval/runs/<LABEL>.json")
    parser.add_argument(
        "--misses", action="store_true", help="only the questions where hit is false"
    )
    args = parser.parse_args()

    # The label becomes a filename; a slash in it would write outside runs/.
    if os.path.basename(args.label) != args.label or not args.label:
        parser.error("--label must be a plain name, no path separators")

    run_path, run = load_run(args.label)
    all_rows = run.get("rows") or []
    rows = [r for r in all_rows if not r.get("hit")] if args.misses else list(all_rows)
    namespace = run.get("namespace") or common.EVAL_NAMESPACE

    chunks = load_chunk_index()
    golden = load_golden_index()

    print(
        console(
            f"inspect_view: label={args.label} namespace={namespace} "
            f"questions={len(rows)} of {len(all_rows)}"
        )
    )
    if namespace != common.EVAL_NAMESPACE:
        # The recorded ranks and the live re-run would be reading different corpora.
        print(
            console(
                f"WARNING: the run was scored in namespace {namespace} but this "
                f"process queries {common.EVAL_NAMESPACE}."
            )
        )
    if args.misses and not rows:
        print(console("no misses in this run -- nothing left to label."))

    built = []
    for i, run_row in enumerate(rows, 1):
        row = build_row(run_row, chunks, golden)
        built.append(row)
        print(
            console(
                f"[{i}/{len(rows)}] {row['id']} hit={'yes' if row['hit'] else 'no'} "
                f"rank={fmt_rank(row['rank'])} dense={fmt_rank(row['dense_rank_top25'])} "
                f"-> {collapse(row['question'])[:70]}"
            )
        )
        if row["answer_error"]:
            print(console(f"    answer failed: {row['answer_error']}"))

    suffix = "_inspect_misses" if args.misses else "_inspect"
    os.makedirs(common.RUNS_DIR, exist_ok=True)
    md_path = os.path.join(common.RUNS_DIR, args.label + suffix + ".md")
    json_path = os.path.join(common.RUNS_DIR, args.label + suffix + ".json")

    lines = render_header(
        args.label, namespace, len(built), len(all_rows), args.misses, run_path
    )
    for row in built:
        lines.extend(render_row(row))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label": args.label,
                "namespace": namespace,
                "misses_only": args.misses,
                "source_run": run_path,
                "n": len(built),
                "rows": built,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(console(f"wrote {md_path}"))
    print(console(f"wrote {json_path}"))


if __name__ == "__main__":
    main()
