# Build Week — Error Analysis & Ranked Taxonomy
**Track C: HR Policy** | Solstice Analytics HR bot (fictional corpus, real RAG pipeline) | 2026-09-08

## Setup

- **App under test:** this repo's actual FastAPI RAG service (`app/services/rag_service.py`), unmodified except for adding trace logging (see `app/services/trace_logger.py`). Retrieval, chunking, embeddings, reranking, and generation all ran for real — nothing here is simulated.
- **Corpus:** 10 synthetic HR policy documents for a fictional company ("Solstice Analytics") covering leave, remote work, code of conduct, expenses, benefits, performance reviews, parental leave, offboarding, overtime, and a 2026 policy addendum that updates two of the older documents. Ingested via `POST /api/v1/documents/upload` → 27 chunks in Pinecone.
- **Sampling:** 22 questions written to cover every document at least once, phrased the way an employee would actually ask (not phrased to target known weak spots), plus a few deliberately out-of-scope questions (pet insurance, wifi-adjacent conflict-of-interest scenario) to check refusal behavior. Not cherry-picked from a larger run — this is the full, unfiltered batch.
- **Traces:** every query's question, retrieved chunks (text + source + score), full prompt, and answer were logged to `traces/traces.jsonl` — complete enough to replay any answer exactly. `data/run_results.json` has the question→trace_id index.

## Unplanned finding, disclosed up front

While reading the traces, **every retrieval also pulled chunks from documents that were already sitting in this Pinecone index from an unrelated, earlier project** — real PDF employee handbooks for a different actual company, with real staff names and email addresses. Nothing in this exercise uploaded those; they were already there because the Pinecone index has no per-project/collection scoping. This is a genuine, severe finding in its own right (see Problem #1 below) and also a data-hygiene issue worth cleaning up in Pinecone independent of this assignment — recommend purging or namespacing that index.

---

## Open-coding notes (written per-trace, before any grouping)

| # | Question (trace_id) | Note |
|---|---|---|
| 1 | PTO days & rollover (`trace_43f638dabe`) | Correct — 18 days, 5-day rollover, matches source exactly. No failure. |
| 2 | Doctor's note after 4 sick days (`trace_1011e7f555`) | Correct answer (3+ day threshold), but 2 of 4 retrieved chunks were from the unrelated company's handbook (48-hr notice / "Leave Balance" jargon that doesn't exist in our policy) — noise that happened not to leak into the answer this time. |
| 3 | Remote equipment stipend "right now" (`trace_c558d40eea`) | Correct — retrieved both the base policy (400 USD) and the 2026 addendum (600 USD) and correctly reported the superseding value. This is the one addendum case that worked. |
| 4 | Hybrid office days/week (`trace_4c42cb60fd`) | Correct (3 days/week). One contamination chunk retrieved (unrelated company's shift-timing rules) but not used. |
| 5 | Working from home country for a month (`trace_471205cb64`) | **Failure** — answered "I cannot answer this based on the provided documents" even though the exact answer (15-day threshold, needs Legal/People Ops approval) was sitting in the retrieved context. Two of the four retrieved chunks were irrelevant contamination about generic WFH conduct rules from the unrelated company, which likely crowded out the model's attention from the one correct chunk. |
| 6 | Who to report harassment to (`trace_761b18eec5`) | **Failure** — our own policy chunk (report to manager/People Ops/Ethics Hotline) was retrieved and used, but the model also pulled in "the Internal Complaints Committee (ICC)" from the unrelated company's real handbook and presented it as if it were Solstice's own reporting channel. Fabricated-sounding but actually copied from someone else's real document. |
| 7 | Fired on the spot for Code of Conduct violation (`trace_2a475b782f`) | Correct — matches our disciplinary-process section. One contamination chunk retrieved (unrelated company's misconduct list) but not used. |
| 8 | Expense submission window (`trace_bb80cb7e56`) | Correct (30 days). No contamination retrieved — clean trace. |
| 9 | Daily meal limit while traveling (`trace_850a855653`) | Correct ($75, itemized). One contamination chunk retrieved (unrelated company's team-lunch INR policy) but not used. |
| 10 | 401(k) match at 6% contribution (`trace_17bec77d11`) | Correct (5% match). No contamination retrieved. |
| 11 | When health insurance starts (`trace_f8f559ab97`) | Correct (first of month following start, no gap). No contamination retrieved. |
| 12 | Performance review cadence & raise impact (`trace_741533f7e1`) | Core answer correct, but the model appended "(June for Solstice Analytics, April or October for other companies)" — an unrelated company's real appraisal-month policy got pulled in and stated alongside ours, which is confusing even though it didn't make our own numbers wrong. |
| 13 | "Needs Improvement" rating outcome (`trace_f31f9c54f1`) | Correct and well-synthesized (PIP process, timeline, termination risk). One contamination chunk retrieved but unused. |
| 14 | Birthing-parent leave weeks, asked "this year" (`trace_5a94a26b63`) | **Failure** — answered 16 weeks (the pre-2026 figure). It's 2026-09-08 "today," so the correct current figure per the addendum is 20 weeks. The addendum chunk wasn't retrieved at all — its slot was taken by two duplicate contamination chunks about a real, unrelated statutory maternity-leave law. The one real chunk that *was* retrieved even contains a pointer sentence to the addendum, but it's truncated mid-sentence by the chunker right before the useful part. |
| 15 | Adoption parental leave, asked "right now" (`trace_1650e61840`) | **Failure** — same bug as #14: answered 8 weeks (stale) instead of the current 10 weeks; same root cause (addendum crowded out by contamination, cross-reference sentence truncated). |
| 16 | Resignation notice for an individual contributor (`trace_e0db8013fc`) | **Failure, most severe of the batch** — answered "3-month working notice." That figure belongs entirely to the unrelated company's real handbook. Our actual policy (2 weeks for ICs, 4 for managers) wasn't retrieved at all — the two duplicate contamination chunks outscored it and took all the top slots. Stated with full confidence, no hedge, no source note. |
| 17 | Health benefits end date on departure (`trace_e80a1a4ad1`) | Correct (last day of month, COBRA, 14-day info window). Two contamination chunks retrieved (unrelated company's full-and-final-settlement text) but not used. |
| 18 | Overtime eligibility for a salaried exempt employee (`trace_4695183e2d`) | **Failure, cleanest example of the batch** — answered "I cannot answer this based on the provided documents" even though the single highest-scored retrieved chunk (rank 1, score 0.74) *opens with the literal answer*: "Exempt employees... are not eligible for overtime pay." No contamination even needed to explain this one — it's a plain refusal-on-answerable-question failure in generation, not retrieval. |
| 19 | On-call stipend this week (`trace_1a5b275421`) | Correct ($100/week). No contamination retrieved — clean trace, addendum chunk retrieved but correctly irrelevant to this specific number. |
| 20 | Expensing a Peloton subscription (`trace_47e1234bae`) | Correct — reasonably inferred Peloton counts as a "gym service" under the non-reimbursable list. One contamination chunk retrieved but unused. |
| 21 | Pet insurance (`trace_501f52a72a`) | Correct refusal — genuinely not covered anywhere, and the model correctly said so. No contamination retrieved. This is the control case showing refusal behavior *can* work correctly. |
| 22 | Now-manager dating a direct report after a reorg (`trace_2da750f79d`) | **Failure** — the reasoning (disclose the conflict of interest) is correct and sourced from our own Code of Conduct, but the answer's closing line tells the user this "ensures compliance with **Soft Suave's** policies" — naming a different, real company outright, sourced from that company's real handbook intro/mission statement that got pulled into context. |

**Raw tally:** 22 traces read, 7 clean answers, 15 traces that pulled in contaminating chunks (7 harmlessly, 6 to the point of corrupting the answer, plus 2 unrelated pure-refusal failures where contamination wasn't even the cause).

---

## Named problem taxonomy

### 1. Cross-tenant document contamination
Retrieval pulls chunks from an unrelated, pre-existing document set (a different real company's employee handbook) because the vector index has no per-project/collection scoping — every query searches the entire index regardless of what was just ingested for this task.
*Seen in:* 15/22 traces retrieved at least one contaminating chunk; it visibly corrupted the final answer in 6 (#6, #12, #14, #15, #16, #22).

### 2. Stale policy values override the current addendum
When a newer document (the 2026 addendum) updates a number in an older document, the app doesn't reliably surface the newer value — it depends on the addendum chunk winning a slot in the top-4, which it lost twice out of three relevant questions.
*Seen in:* 2/3 addendum-relevant questions (#14, #15); #3 succeeded.

### 3. False refusal on answerable questions
The model says "I cannot answer this based on the provided documents" even when the correct answer is plainly present in the retrieved context — in one case it's the literal first sentence of the top-ranked chunk.
*Seen in:* 2/22 (#5, #18).

### 4. Chunker severs structure and cross-references mid-sentence
The fixed 1000-character sliding-window chunker (combined with whitespace-collapsing during cleaning, which destroys headings and list structure) routinely cuts chunks mid-clause. Most of the time the model recovers via overlap, but at least once it cut off the exact sentence that points to the addendum, right before the useful part.
*Seen in:* structurally present in nearly every multi-chunk retrieval; decisive (not just cosmetic) in #14/#15.

### 5. No source attribution or contradiction signaling in answers
The API returns a `sources` list, but the answer text itself never cites which document a number came from and never flags when two retrieved chunks disagree. This doesn't cause wrong answers by itself, but it's why problems #1 and #2 are invisible to a real user — a source label ("per the [Unrelated Company] handbook...") would have made #16 and #22 obviously wrong at a glance.
*Seen in:* effectively all 22 answers.

---

## Ranked by frequency × severity

| Rank | Problem | Frequency | Severity | Why this rank |
|---|---|---|---|---|
| **1** | Cross-tenant document contamination | High (15/22) | High (6 corrupted answers, incl. a wrong company name stated to the user; also a data-governance/privacy issue) | Most frequent *and* most damaging category; also the root retrieval-competition cause behind #2. |
| **2** | False refusal on answerable questions | Low (2/22) | High (complete failure to help on a common, plainly answerable question) | Rare but total failure per occurrence — for a real HR bot this is as trust-damaging as a wrong answer. |
| **3** | Stale policy values override current addendum | Low (2/3 relevant) | High (confident wrong compliance-relevant numbers, e.g. parental leave duration) | Narrow trigger surface but serious consequences when it fires; downstream of #1. |
| **4** | Chunker severs structure/cross-references | High (structural) | Low–Medium (usually absorbed by overlap; decisive only via #2) | Frequent but rarely decisive on its own in this batch. |
| **5** | No source attribution / no contradiction signaling | High (~22/22) | Low standalone / High as an amplifier | Never itself produced a wrong fact, but is why #1 and #3 go unnoticed by a real user. |

---

## Chosen fix target and prediction

**Target: #1, cross-tenant document contamination.** It's simultaneously the most frequent problem, the one with the most severe individual failures (a wrong company name told to the user, someone else's real policy number stated as fact), the cheapest to fix relative to impact (scope Pinecone queries to a `collection_id`/namespace instead of searching the whole index), and it's the upstream cause competing for retrieval slots in problem #3.

**Prediction (written before implementing the fix):** After adding collection scoping to ingestion and retrieval —
1. Contamination chunks will disappear from all 22 questions' retrieved context.
2. The two answers directly corrupted by contamination with a clean root cause (#16 notice period, #22 wrong company name) will become correct.
3. Both addendum failures (#14, #15) will very likely also self-resolve, since the addendum chunks will no longer compete against 2 duplicate irrelevant chunks for the last two of four context slots — expect 2/2 to flip to correct.
4. This fix will **not** resolve the false-refusal problem (#5, #18) — those happened with correct context already present, so refusal behavior needs a separate fix (rank #2 next).

If (3) doesn't hold after the fix, that means chunking/truncation (#4) is a bigger independent factor than estimated here and should be promoted.

---

## Post-fix verification (2026-09-22)

### What changed — and what didn't
- **Code:** `app/vectorstore/pinecone_store.py` now scopes `upsert`, `query` and `delete` to a namespace read from `PINECONE_NAMESPACE` (empty keeps the old, unscoped behaviour).
- **Config:** `.env` sets `PINECONE_NAMESPACE=solstice`; documented in `.env.example`.
- **Data:** the same 10 files, re-uploaded through the same `POST /api/v1/documents/upload` endpoint, into namespace `solstice` → **28 chunks**. The Setup section's "27" looks like a miscount: the same files through the same, unchanged code produce 28, and the files were last modified at 13:18 UTC on 2026-09-08, before the first trace at 13:20.
- **Nothing deleted:** the default namespace still holds the same 244 vectors, contamination included. The app just no longer reads it.

Unchanged: the 22 questions, retrieval code, chunker, prompt and model. Re-run with the same script — `python scripts/run_traces.py data/run_results_after_fix.json` — so the Week 3 index in `data/run_results.json` is preserved; the new traces are appended to `traces/traces.jsonl`.

### Prediction check

| # | Prediction (written before the fix) | Result | Verdict |
|---|---|---|---|
| 1 | Contamination disappears from all 22 | Questions retrieving any foreign chunk: **15/22 → 0/22** | **Held** |
| 2 | #16 and #22 become correct | #16: "at least 2 weeks' written notice" (was "3-month notice"). #22: same correct advice, no other company named | **Held** |
| 3 | #14 and #15 self-resolve (2/2) | The addendum chunk now ranks 3rd for both — the mechanism held. #15 correct (10 weeks). #14 no longer answers the stale 16 weeks, but refuses 3 times in 8 | **Partly held** |
| 4 | The fix will **not** resolve the false refusals (#5, #18) | #5: refusals fell from 4/8 to 1/8 — contamination *was* a cause, as the #5 open-coding note itself suspected. #18: 1/8 before and after — no effect, because the Week 3 refusal was a 1-in-8 draw | **Half held** |

### Why single traces weren't enough
`app/generation/llm.py` sets no temperature, so every trace is one sample of what the model might say. For the questions whose single after-fix trace contradicted a prediction, plus #20 (below), the exact stored prompt was re-sent to the same model 8 times — once with the Week 3 context, once with the after-fix context. Retrieval is deterministic; only generation varies.

| # | Week 3 context (8 replays) | After-fix context (8 replays) |
|---|---|---|
| 5 | 4 answer, 4 refuse | 7 answer, 1 refuse |
| 14 | 7 stale "16 weeks", 1 refuse | 5 "20 weeks" (3 hedged by year), 3 refuse |
| 18 | 7 answer, 1 refuse | 7 answer, 1 refuse |
| 20 | 5 "No", 3 refuse | 8 "Yes, up to $50/month" |

Raw samples and tallies: `data/replay_results.json`. `python scripts/replay_prompts.py` re-runs the replay (new samples will differ slightly).

### A Week 3 grading error the fix exposed
**#20 (Peloton) was wrong in Week 3, not correct.** The corpus contradicts itself: the Expense Policy (§6) excludes "spa or gym services", but the Benefits Overview (§4, Wellness Benefit) allows "up to 50 USD per month toward gym memberships, fitness classes, or wellness apps". The specific rule wins, so the answer is yes, up to $50/month. In Week 3 the Wellness chunk lost its slot to a contaminating chunk, so "No" was a retrieval failure that happened to sound right. After the fix the Wellness chunk ranks 4th, and all 8 replays answer correctly.

### Scorecard (one pass over all 22, same method as Week 3)
- Correct answers: **13/22 → 21/22** (Week 3 reported 14; #20 regraded as wrong).
- The one remaining failure is #14 — now a generation failure: the right chunk is in context and the model refuses.

### Updated ranking — what to fix next
1. ~~Cross-tenant document contamination~~ — **fixed** (0/22).
2. **False refusal on answerable questions — now the top problem.** With clean context the model still refuses answerable questions: #14 3/8, #5 1/8, #18 1/8, each with the answer in context. A generation problem; better retrieval cannot fix it.
3. **Stale values vs the addendum** — the retrieval half is fixed (the addendum now reaches the context). What's left is that "this year" / "right now" questions make the model guess today's date, which the prompt never states — 3 of #14's 5 correct answers hedge by year, and the refusals are likely the same uncertainty. Folds into #2: put the current date in the prompt.
4. **Chunker (#4) is not promoted.** The report said to promote it if (3) failed; (3) failed only through refusals, and the addendum chunk carries the whole "20 weeks" sentence — nothing was truncated.

**Still open, outside this fix:** the contaminated vectors (the unrelated handbook, uploaded twice) are still in the default namespace. The app no longer reads them, but they should be purged.
