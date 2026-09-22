"""Compare two eval runs written by run_eval.py.

Prints a metric table (hit-rate@3, p50 latency) and a per-question before/after
table to stdout. Nothing is written to disk -- the printed markdown is what goes
into the write-up.

    python eval/compare.py baseline after_bm25
"""

import argparse
import json
import os
import statistics
import sys

import common  # noqa: F401  (must come first: sets sys.path + PINECONE_NAMESPACE)

QUESTION_WIDTH = 50
GOLD_WIDTH = 44

STATUS = {
    (False, True): "FIXED",
    (True, False): "REGRESSED",
    (True, True): "still hit",
    (False, False): "still broken",
}


def emit(line):
    # The Windows console is cp1252, so fold anything the corpus dragged in
    # rather than raising UnicodeEncodeError mid-table.
    print(line.encode("ascii", "replace").decode("ascii"))


def load_run(label):
    """Load eval/runs/<label>.json, or exit with a message naming the path."""
    path = os.path.join(common.RUNS_DIR, label + ".json")
    if not os.path.isfile(path):
        emit(f"ERROR: no run file for label '{label}'")
        emit(f"  looked for: {path}")
        emit(f"  write one with: python eval/run_eval.py --label {label}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except ValueError as exc:
            emit(f"ERROR: {path} is not valid JSON ({exc})")
            sys.exit(1)


def rows_by_id(run):
    # Keyed, but still in file order (dicts keep insertion order), so this doubles
    # as the question order for the table below.
    return {row.get("id"): row for row in run.get("rows") or [] if row.get("id")}


def check_same_ids(before_rows, after_rows, before_label, after_label):
    """Refuse to compare runs over different questions.

    Intersecting them silently would hide a golden-set edit and quietly change
    what the hit-rate is a rate *of*.
    """
    only_before = sorted(set(before_rows) - set(after_rows))
    only_after = sorted(set(after_rows) - set(before_rows))
    if not only_before and not only_after:
        return
    emit("ERROR: the two runs do not cover the same question ids.")
    if only_before:
        emit(f"  only in {before_label}: {', '.join(only_before)}")
    if only_after:
        emit(f"  only in {after_label}: {', '.join(only_after)}")
    emit("Re-run both labels against the same golden set, then compare.")
    sys.exit(2)


def warn_mismatches(before, after, before_label, after_label):
    """Differences that do not invalidate the ids but do colour the numbers."""
    warnings = []
    if before.get("namespace") != after.get("namespace"):
        warnings.append(
            f"namespace differs -- {before_label}={before.get('namespace')}, "
            f"{after_label}={after.get('namespace')}"
        )
    before_repeats = (before.get("summary") or {}).get("repeats")
    after_repeats = (after.get("summary") or {}).get("repeats")
    if before_repeats != after_repeats:
        warnings.append(
            f"repeats differ -- {before_label}={before_repeats}, "
            f"{after_label}={after_repeats}; latency is not comparable"
        )
    for warning in warnings:
        emit(f"WARNING: {warning}")
    if warnings:
        emit("")


def hit_counts(rows):
    return sum(1 for row in rows if row.get("hit")), len(rows)


def p50_latency(run, rows):
    """Median of the per-query medians.

    Recomputed from the rows so the metric table can never disagree with the
    per-question table under it; the run's own summary is only a fallback.
    """
    values = [row["latency_ms"] for row in rows if row.get("latency_ms") is not None]
    if values:
        return statistics.median(values)
    return (run.get("summary") or {}).get("p50_latency_ms")


def fmt_rate(hits, n):
    if not n:
        return "0/0 = n/a"
    return f"{hits}/{n} = {hits / n:.3f}"


def fmt_ms(value):
    return "n/a" if value is None else f"{value:.2f}"


def fmt_latency(value):
    return "n/a" if value is None else f"{value:.1f}"


def fmt_rank(rank):
    return "-" if rank is None else str(rank)


def fmt_gold(gold):
    if not gold:
        return "-"
    if isinstance(gold, str):
        return gold
    return ",".join(str(chunk_id) for chunk_id in gold)


def ratio_verdict(before_ms, after_ms):
    # Bare truthiness on both: a missing or 0.0 latency makes the ratio either
    # meaningless or a division by zero.
    if not before_ms or not after_ms:
        return "n/a"
    ratio = after_ms / before_ms
    if ratio > 1.0:
        return f"{ratio:.2f}x slower"
    if ratio < 1.0:
        return f"{1.0 / ratio:.2f}x faster"
    return "same"


def cell(text, limit):
    # Truncate before escaping: escaping first risks a cut that splits "\|" and
    # breaks the table on the one row you most wanted to read.
    text = str(text)
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text.replace("|", "\\|")


def table(header, rows):
    emit("| " + " | ".join(header) + " |")
    emit("|" + "|".join([" --- "] * len(header)) + "|")
    for cells in rows:
        emit("| " + " | ".join(cells) + " |")


def print_metric_table(before, after, before_rows, after_rows):
    # Rows come from the caller, not from before["rows"], so these numbers are
    # over exactly the questions the per-question table lists.
    before_hits, before_n = hit_counts(before_rows)
    after_hits, after_n = hit_counts(after_rows)

    if before_n and after_n:
        rate_delta = f"{after_hits / after_n - before_hits / before_n:+.3f}"
    else:
        rate_delta = "n/a"

    before_ms = p50_latency(before, before_rows)
    after_ms = p50_latency(after, after_rows)
    if before_ms is None or after_ms is None:
        ms_delta = "n/a"
    else:
        ms_delta = f"{after_ms - before_ms:+.2f}"

    if before_ms and after_ms:
        ratio_before, ratio_after = "1.00x", f"{after_ms / before_ms:.2f}x"
    else:
        ratio_before, ratio_after = "n/a", "n/a"

    table(
        ["metric", "before", "after", "delta"],
        [
            [
                "hit-rate@3",
                fmt_rate(before_hits, before_n),
                fmt_rate(after_hits, after_n),
                rate_delta,
            ],
            ["p50 latency (ms)", fmt_ms(before_ms), fmt_ms(after_ms), ms_delta],
            ["p50 latency ratio", ratio_before, ratio_after, ratio_verdict(before_ms, after_ms)],
        ],
    )


def pick(before_row, after_row, key):
    """Question-level fields belong to the golden set, not to a run -- read them
    from the before run, falling back if an older run file predates the field."""
    value = before_row.get(key)
    return after_row.get(key) if value is None else value


def build_question_rows(before_rows, after_rows):
    rows = []
    counts = {"FIXED": 0, "REGRESSED": 0, "still hit": 0, "still broken": 0}
    for question_id in before_rows:
        before_row = before_rows[question_id]
        after_row = after_rows[question_id]
        status = STATUS[(bool(before_row.get("hit")), bool(after_row.get("hit")))]
        counts[status] += 1
        before_ms = fmt_latency(before_row.get("latency_ms"))
        after_ms = fmt_latency(after_row.get("latency_ms"))
        rows.append(
            [
                cell(question_id, 16),
                "y" if pick(before_row, after_row, "exact_token") else "-",
                cell(pick(before_row, after_row, "question") or "", QUESTION_WIDTH),
                cell(fmt_gold(pick(before_row, after_row, "gold")), GOLD_WIDTH),
                fmt_rank(before_row.get("rank")),
                fmt_rank(after_row.get("rank")),
                f"{before_ms} -> {after_ms}",
                status,
            ]
        )
    return rows, counts


def main():
    parser = argparse.ArgumentParser(
        description="Compare two eval runs in eval/runs/ (hit-rate@3 and p50 latency)."
    )
    parser.add_argument("before", help="label of the baseline run, e.g. baseline")
    parser.add_argument("after", help="label of the changed run, e.g. after_bm25")
    args = parser.parse_args()

    before = load_run(args.before)
    after = load_run(args.after)
    before_rows = rows_by_id(before)
    after_rows = rows_by_id(after)
    check_same_ids(before_rows, after_rows, args.before, args.after)

    emit(f"# compare: {args.before} -> {args.after}")
    emit("")
    warn_mismatches(before, after, args.before, args.after)

    print_metric_table(before, after, before_rows.values(), after_rows.values())
    emit("")

    rows, counts = build_question_rows(before_rows, after_rows)
    table(
        ["id", "tok", "question", "gold", "rank before", "rank after", "latency", "status"],
        rows,
    )
    emit("")
    emit(
        f"tally: {counts['FIXED']} FIXED, {counts['REGRESSED']} REGRESSED, "
        f"{counts['still hit']} still hit, {counts['still broken']} still broken"
        f"  (n={len(rows)})"
    )


if __name__ == "__main__":
    main()
