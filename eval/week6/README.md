# eval/week6 — validated policy-answer judge (Week 6, Task Set C)

One command scores every change to the HR assistant, per failure mode, with a judge
whose agreement with a human was measured before its number was trusted.

## The one command

```powershell
.\.venv\Scripts\Activate.ps1
python eval/week6/run_eval.py --label <name>            # asks the app, asserts, judges (judge v2), prints pass rate by mode
```

Useful variants:

| Command | What it does |
| --- | --- |
| `run_eval.py --label X --judge none` | assertions only (the only mode allowed before labels are committed) |
| `run_eval.py --label X --answers-from Y --judge v1` | re-score the frozen answers of run `Y` (no app calls) |
| `label.py --run baseline --labeler NAME` | blind hand-labelling of the 25 judged answers → `labels_25.json` |
| `agreement.py --run X [--exclude a,b]` | judge-vs-human agreement on the labelled answers |

The app is called in-process (`RAGService`), reading the same `.env` and Pinecone
namespace (`solstice`) as the running API. Every answer is also a normal trace in
`traces/traces.jsonl`, so any row can be replayed. A Groq rate limit mid-run is
survivable: answers are checkpointed to `runs/<label>.partial.json` and re-running
the same command resumes.

## Cases — `cases.jsonl` (29)

Each case is tagged with exactly one mode from the Week 3/5 taxonomy
(`data/error_analysis_report.md`):

| mode | cases | what it probes |
| --- | --- | --- |
| `contamination` | 8 (4 answer, 4 must-refuse) | foreign handbook content leaking in; out-of-jurisdiction / not-in-handbook questions must take the refusal path |
| `stale_addendum` | 5 | 2026 addendum figures (20/10 weeks, 600/35 USD) must beat the superseded ones |
| `false_refusal` | 6 | answerable questions the app refused in real traces |
| `chunk_severed` | 5 | answers spanning cross-references / sections |
| `no_attribution` | 5 | answers where the source matters (e.g. Peloton: Wellness Benefit vs gym exclusion) |

**12 regression cases** (`regression_of` = the failed trace) replay real failures
verbatim — `run_eval.py` asserts each question is byte-identical to the trace's query
before it runs. They include the Week 3 "3-month notice" and "Internal Complaints
Committee" contamination failures, the stale "16 weeks", and six Week 5 false refusals.

## Assertions vs judge — 5 assertions, 1 judged criterion

Everything an `if` can decide was moved out of the judge (`judge_v0.txt` → `judge_v1.txt`):

| Criterion | Where | How |
| --- | --- | --- |
| Out-of-jurisdiction / not-in-handbook question triggers the refusal path | assertion `refusal_path` | refusal regex |
| Answerable question is not refused | assertion `no_false_refusal` | refusal regex |
| Amount / duration / notice-period figure is numeric | assertion `figure_numeric` | digit+unit regex |
| A policy section reference is present and resolves to a real section | assertion `section_ref_resolves` | "Section N" + nearest policy name → lookup table parsed from `data/hr_policies` (`policy_index.py`) |
| The handbook version is cited | assertion `version_cited` | an effective date that belongs to a cited policy |
| **Is the stated policy correct?** | **judge** (binary PASS/FAIL, `openai/gpt-oss-120b`, temperature 0) | needs reading comprehension |

A case passes only if all applicable assertions pass **and** the judge says PASS.

## Blind protocol (enforced, not promised)

1. `run_eval.py --label baseline --judge none` — app answers frozen in `runs/baseline.json`.
2. `label.py` — 25 answers labelled blind (question + reference + answer only, seeded shuffle).
3. `labels_25.json` committed.
4. Only now can the judge run: `judge.py` refuses unless `labels_25.json` is committed and
   unchanged since, and every judged run records that commit hash + time
   (`labels_commit` in `runs/*.json`).

## Results

_Filled in as measured._
