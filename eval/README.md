# eval/ - retrieval eval harness (Week 4, Task Set C)

Measures retrieval on an HR policy corpus: ingest the corpus, build a 12-question
golden set, get a baseline hit-rate@3, label every miss, make exactly one
retrieval change chosen from the failure tally, then re-measure and compare.

**The rule that makes it safe:** every script imports `common.py` first, which sets
`PINECONE_NAMESPACE` from `$EVAL_NAMESPACE` (default `"hr"`). Everything the harness
ingests, searches and deletes stays inside that namespace, so the running app's
default namespace (`""`) is never touched. Never pass a namespace argument yourself -
let `common.py` set it. The one exception is the teardown `delete` under
[Smoke test](#smoke-test), where naming the namespace is the safety.

## Setup (once per shell)

Run everything from the `rag_app` directory.

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:EVAL_NAMESPACE = "hr"
```

Git Bash:

```bash
source .venv/Scripts/activate
export EVAL_NAMESPACE=hr
```

`EVAL_NAMESPACE` is optional - `common.py` defaults to `"hr"`. Set it when you want a
throwaway namespace (see [Smoke test](#smoke-test)).

Always invoke the scripts as `python eval/<script>.py` from `rag_app`. That puts
`eval/` on `sys.path[0]`, which is how `import common` resolves.

---

## Runbook

### 1. Corpus

Drop the HR policy files into `eval/corpus/`. Supported: `.pdf`, `.docx`, `.txt`, `.md`.

`.docx` needs `python-docx`, which is not installed - either `pip install python-docx`
or save the file as `.txt` first. The other three types need nothing extra.

Name files so the stem is meaningful and distinct: it becomes the chunk id prefix
(step 2), and two names that slug to the same id collide - ingest warns and skips the
second one rather than letting two documents share a chunk id.

### 2. Ingest

```powershell
python eval/ingest_hr.py
```

**Why this exists instead of the app's upload endpoint:** the endpoint assigns a random
uuid per document, so chunk ids change on every re-ingest and a golden set cannot
reference them. `ingest_hr.py` slugs the file stem into a stable `document_id`:

```
"HR-207 Leave Policy.pdf"  ->  document_id  hr_207_leave_policy
                           ->  chunk ids    hr_207_leave_policy_chunk_0, _chunk_1, ...
```

Those ids survive re-ingests, so `golden_set.jsonl` keeps pointing at the right chunks.
Re-running is safe: ids are stable and upserts overwrite in place. Chunks are also
written to `eval/chunks.jsonl` for `find.py` to search.

Add `--dry-run` to chunk and write `chunks.jsonl` without embedding or upserting - no
network, useful for checking that the files parse and the slugs come out right.
`--corpus DIR` points at a different folder.

### 3. Find gold chunks and write the golden set

```powershell
python eval/find.py "carry over" HR-207
```

`find.py` prints every chunk in `chunks.jsonl` containing **all** the given terms
(case-insensitive substrings; quote multi-word terms), with its chunk id and a snippet.
`--context N` widens the snippet, `--limit N` caps the output. Use it to get the gold
chunk id for each question, then write 12 questions into `eval/golden_set.jsonl`, one
JSON object per line:

```json
{"id":"q01","question":"How many unused leave days carry over into the next year?","gold_chunk_id":"hr_207_leave_policy_chunk_3","exact_token":true,"expected_answer":"Up to 10 days, and they expire on 31 March."}
```

- `gold_chunk_ids` (a list) is accepted instead of `gold_chunk_id` when an answer
  legitimately spans two chunks - retrieving either one counts as a hit.
- `id` must be present and unique. It is what `run_eval.py` writes into every run row
  and what `compare.py` matches the two runs on, question by question; `compare.py`
  refuses to compare runs whose id sets differ.
- `exact_token` is optional: left out, `run_eval.py` derives it from
  `common.EXACT_TOKEN_RE`. Set it by hand only to overrule the regex.
- `expected_answer` is optional, but write it anyway: at step 5 it is what makes the
  R-vs-G call quick instead of a re-reading exercise.

**Mix the questions deliberately:** roughly half exact-token (policy numbers, section
refs, dates, region codes - the things `common.EXACT_TOKEN_RE` matches) and half
paraphrase, with no literal overlap with the source wording. That mix is the whole
point of the tally at step 6: exact-token misses and paraphrase misses fail for
different reasons, and if the set is all one kind you cannot tell a lexical-recall
problem from a ranking problem.

### 4. Baseline

```powershell
python eval/run_eval.py --label baseline
```

Writes `eval/runs/baseline.json`. `--repeats N` sets how many timings per query the
median is taken over (default 5); the run file records it as `summary.repeats`.

### 5. Inspect and label every miss

```powershell
python eval/inspect_view.py --label baseline --misses
```

Generates `eval/runs/baseline_inspect_misses.md` - each miss with its question, its gold
chunk text, what retrieval actually returned, and the app's real answer (that last one is
what the R-vs-G call turns on). Go through it and label **every** miss with exactly one
of:

| Label | Meaning |
| --- | --- |
| **R** | Retrieval. The gold chunk exists and answers the question, but retrieval did not put it in the top 3. |
| **G** | Generation. Retrieval found it; the generator still answered wrong. |
| **Not-In-Corpus** | No chunk actually answers the question. Fix the question or the corpus - this is not a retrieval bug and must not count toward the tally. |

Drop `--misses` to review all 12 rows, and do that at least once: `--misses` filters on
`hit`, and a **G** by definition has the gold chunk in the top 3, so a G never appears in
the misses-only view - it hides inside a question the run counted as a hit.

### 6. Pick the one change - from the tally, not from taste

Tally the **R** misses only, then read `dense_rank_top25` for each of them in
`runs/baseline.json`. That field is the evidence, and it answers one question: could a
reranker over the dense top-25 even *see* the gold chunk?

| What the tally shows | The change |
| --- | --- |
| Mostly R misses on exact-token questions, gold often **not** in the dense top-25 | **BM25 + RRF (k=60)** - the gold chunk is out of reach entirely, so this is a lexical recall problem and no reranker can fix it. |
| Mostly R misses where the gold **is** in the dense top-25 but ranked 4-25 | **Cross-encoder rerank over top-25** - recall is fine, ordering is not. |

Make **exactly one** of them. The point of the assignment is a change traceable to
evidence, so write the tally down before you touch any code.

This harness implements neither one - it only measures. The change lands in
`app/retrieval/`, where both legs are currently placeholders: `hybrid_search.py`
returns dense results only, and `reranker.py` truncates without reordering.

### 7. Re-measure and compare

```powershell
python eval/run_eval.py --label after
python eval/compare.py baseline after
```

`compare.py` reports hit-rate@3 **and** p50 latency side by side, plus which questions
flipped in each direction. Both numbers matter: the reranker option usually buys
accuracy with latency, and a change that adds 300 ms for one extra hit is a result you
need to see, not hide.

Illustrative shape only - fill in from your own `runs/*.json`. No number in this README
is measured:

| run | n | hit-rate@3 | p50 latency |
| --- | --- | --- | --- |
| baseline | 12 | 7/12 (0.583) | -- ms |
| after | 12 | --/12 | -- ms |

---

## Files in this folder

| Path | What it is |
| --- | --- |
| `common.py` | Shared setup: puts `rag_app` on `sys.path`, loads `.env`, pins `PINECONE_NAMESPACE`, and holds the paths, the exact-token regex and the jsonl/rank helpers. Import it first in every script. |
| `ingest_hr.py` | Ingests `corpus/` into the eval namespace with stable, slug-based chunk ids. |
| `find.py` | Searches the ingested chunks for a phrase; used to pick gold chunk ids. |
| `run_eval.py` | Runs the golden set through the retriever and writes `runs/<label>.json`. |
| `inspect_view.py` | Renders a run as a reviewable markdown file for R / G / Not-In-Corpus labelling. |
| `compare.py` | Diffs two runs on hit-rate@3 and p50 latency. |
| `corpus/` | Your HR policy source files; the `.gitkeep` keeps the folder. `__pycache__/` is the only thing under `eval/` that `.gitignore` catches - a corpus dropped in here is not, so check before committing one you did not mean to publish. |
| `chunks.jsonl` | Every chunk written at ingest - `find.py` searches it, and `inspect_view.py` pulls the gold chunk text from it. |
| `golden_set.jsonl` | The 12 questions, one JSON object per line. |
| `runs/` | One set of output files per label. |

## What lands in runs/

| File | Written by |
| --- | --- |
| `<label>.json` | `run_eval.py` - the run file (summary + one row per question). |
| `<label>_inspect.md` / `<label>_inspect_misses.md` | `inspect_view.py` - the human-readable review doc; `_misses` when `--misses` is passed. |
| `<label>_inspect.json` / `<label>_inspect_misses.json` | The same content as data. |

## Metric definitions

Stated once here; `run_eval.py`, `inspect_view.py` and `compare.py` all use exactly these.

`runs/<label>.json` carries `label`, `namespace`, a `summary` (`n`, `hits`,
`hit_rate_at_3`, `p50_latency_ms`, `repeats`) and one `rows` entry per question
(`id`, `question`, `gold`, `exact_token`, `hit`, `rank`, `dense_rank_top25`,
`top3` as `chunk_id`/`score` pairs, `latency_ms`). `run_eval.py` writes those names;
`inspect_view.py` and `compare.py` read them, so do not rename one without the others.

- **`rank`** - the gold chunk's 1-based rank in the **full** retriever output.
  `retrieval_service.retrieve()` returns `final_k=4`, so rank is 1-4, or `null` if the
  gold chunk is absent.
- **`hit`** - `rank` is not null and `rank <= 3`. This is hit-rate@3.
- **`hit_rate_at_3`** - hits divided by number of questions.
- **`latency_ms`** - the median of `--repeats` timings of `retrieve()` for that question.
- **`p50_latency_ms`** - the median of those per-query medians.
- **`dense_rank_top25`** - the gold chunk's 1-based rank in
  `perform_vector_search(q, top_k=25)`, or `null`. Diagnostic only and never timed: it
  is the evidence for step 6, telling you whether a reranker could reach the gold chunk
  at all.

## Smoke test

Before a real run, exercise the whole pipeline against a throwaway namespace and a
2-file dummy corpus - it catches a bad `.env` or a broken chunk id without writing
anything into `hr`. Keep the dummy files in their own folder and point `--corpus` at it,
so `eval/corpus/` stays the real corpus:

```powershell
$env:EVAL_NAMESPACE = "smoke"
python eval/ingest_hr.py --corpus eval/smoke_corpus
python eval/run_eval.py --label smoke
```

`run_eval.py` needs `golden_set.jsonl` to exist, and against a dummy corpus every
question misses: what the smoke run proves is that the pipeline runs end to end, not
anything about the numbers. Note that ingest overwrites `chunks.jsonl` with the dummy
chunks (dry run or not), so re-run the real `python eval/ingest_hr.py` afterwards or
`find.py` and `inspect_view.py` will be reading the wrong corpus.

Then tear the namespace down. Run it from `rag_app`; `python -c` puts the current
directory on `sys.path`, not `eval/`, so `common` has to be pointed at:

```powershell
python -c "import sys; sys.path.insert(0, 'eval'); import common; from app.vectorstore.pinecone_store import pinecone_store; pinecone_store.index.delete(delete_all=True, namespace='smoke')"
```

**Warning:** `delete_all=True` is irreversible and the `namespace` argument is the only
thing standing between it and your data. Never run it against the default namespace
(`""`, i.e. omitting `namespace=`) - that is the live app's index content, not the
harness's. Type the namespace explicitly every time, and re-set
`$env:EVAL_NAMESPACE = "hr"` afterwards.
