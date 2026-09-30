# eval/week7 — Race the HR agent against a fixed workflow (Week 7, Task Set C)

Entitlement questions ("what notice must the company give E1004?") answered two ways over the
same 10 questions, same model (`openai/gpt-oss-20b` on Groq, `reasoning_effort=low`, the app's
pinned Week 5 params), same three tools, same output contract.

## One command each

```powershell
.\.venv\Scripts\Activate.ps1
python -m app.agent.agent    "The company is eliminating E1004's role. What is the minimum notice it must give E1004?"
python -m app.agent.workflow "The company is eliminating E1004's role. What is the minimum notice it must give E1004?"
python eval/week7/race.py                 # the race -> race.csv, race_by_question.csv, runs/race.json
python eval/week7/test_budgets.py         # offline: each of the 4 budgets stops a spinning agent (no API calls)
python eval/week7/tool_diff.py            # -> third_tool.diff
```

The agent CLI takes `--max-iters / --max-tokens / --max-cost / --max-wall` to tighten any budget,
and `--save run.json` to keep every step.

| File | What it is |
| --- | --- |
| `app/agent/agent.py` | the loop (~100 lines of logic), every lap logged, 4 budgets enforced |
| `app/agent/workflow.py` | the fixed 6-step workflow — no loop |
| `app/agent/tools.py` | the 3 tools + specs + one dispatcher both systems call |
| `app/agent/contract.py` | shared policy rules, output contract, and the one parser both systems' answers go through |
| `app/agent/model.py` | one accounted chat call: tokens, cost, latency, 429 waits (kept out of latency) |
| `app/agent/handbook.py` | section-level BM25 over `data/hr_policies` (local, deterministic) |
| `data/hr_agent/` | synthetic employee records + the Local Jurisdiction Supplement |
| `eval/week7/questions.jsonl` | the 10 questions, expected value/unit/citation, and why |

## Results — race.csv (10 questions × 2 trials = 20 runs per system)

| system | pass rate | p50 latency | total tokens (10 questions) | cost / question |
| --- | --- | --- | --- | --- |
| **agent** | **80%** (16/20) | **2.84 s** | **39,738** | **$0.000307** |
| **workflow** | **70%** (14/20) | **1.26 s** | **13,610** | **$0.000147** |

Tokens are summed over **every** model call in a run. The agent re-sends the whole message list on
every lap: its prompt grows from ~700 to ~1,600 tokens by lap 5. Counting only the final call would
report 27,699 agent tokens over the 20 runs instead of 79,475, an understatement of 2.9×. "Total tokens" is one pass over the 10 questions (the mean of the 2 trials).
Latency excludes time spent waiting on Groq 429s, which came to **0 s** in all 40 runs: the harness
waits for the token bucket to refill *between* runs, outside the timed region.

The same numbers by input class — this is where the race is decided:

| class (questions) | what varies | agent pass | workflow pass | agent tok/q | workflow tok/q | agent p50 | workflow p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_lookup (q01–q03) | nothing | 5/6 | 6/6 | 3,308 | 1,290 | | |
| param_branch (q04–q07) | an *argument*: tenure band / jurisdiction override | 6/8 | 8/8 | 3,274 | 1,328 | | |
| **q01–q07 together** | | **11/14** | **14/14** | 3,288 | 1,312 | 2.24 s | 1.17 s |
| **path_branch (q08–q10)** | **the *steps*: record links to another record / jurisdiction** | **5/6** | **0/6** | 5,573 | 1,476 | 3.75 s | 1.59 s |

Each trial on its own gives the same pass count: agent 8/10 and workflow 7/10 in both.

Why each run failed (`race_by_question.csv` has every row):

| question | system | trials | what happened |
| --- | --- | --- | --- |
| q08, q10 (rehire) | workflow | t1, t2 | abstained 3 times, and once (q10 t2) treated the 2-month current stint as probation-era service → "4 weeks" instead of 4 months. It sees `tenure_note: not computable ... prior record E0877` and has no step to fetch E0877. |
| q09 (UK→DE transfer) | workflow | t1, t2 | 24 days: used the company-wide 18 days for the UK months. It never looked up the UK rule — there is no step for it. |
| q05 (UK, 9 years) | agent | t1, t2 | 112 weeks: read `complete_months_of_service` (112) instead of years. The workflow read the same record correctly. |
| q09 (transfer) | agent | t1 | right path (both jurisdiction rules fetched), wrong arithmetic: 11 UK months instead of 6 → 25.42 |
| q02 (parental) | agent | t2 | **spin:** 6 rejected tool calls in a row, then `max_iters` fired (see the budget log below) |

## Verdict (decision rule: does the path vary by input?)

For q01–q07 it does not. Tenure and jurisdiction change an *argument* to step 4, never the steps.
On those seven the workflow went 14/14 to the agent's 11/14, at 40% of the tokens and half the p50
latency. The path does vary for one class: **the employee record links to another entity** — a
rehire whose tenure lives on a prior record (q08, q10), or a mid-year transfer that needs a second
jurisdiction's rule (q09). There the workflow scored 0/6 and the agent 5/6. That class alone is the
agent's whole 80% vs 70% lead. Ship the workflow as the default and route only records carrying
`service_continuity` or `transfer_history` to the agent. Or store a continuous-service date on the
record, which turns that class back into a workflow case.

*(125 words.)*

## Budgets — enforced in code, and one real termination

All four are checked in `run_agent` before **every** lap (`gate()` in `app/agent/agent.py`):

- **max_iters = 8.** Counts model calls, including rejected ones.
- **max_tokens = 20,000.** Prompt + completion tokens summed over laps. The next lap's re-sent
  prompt is projected, so a lap that *would* cross the limit is never started. Each lap's
  completion is also capped to what the budget has left.
- **max_cost_usd = $0.003.** Same projection, priced at $0.075/M in and $0.30/M out (Groq's
  gpt-oss-20b rates).
- **max_wall_s = 90 s.** Also passed as the request timeout, and caps any 429 sleep.

When a budget fires, the loop returns a well-formed `cannot_answer` contract naming the budget. It
never raises.

**Real run, not staged:** `logs/budget_termination.log` (full run in `logs/budget_termination_run.json`).
The same spin happened inside the race (q02, trial 2).

```
[agent   0.00s] question: E1009 is adopting a child. How many weeks of paid parental leave is E1009 entitled to?
[agent   0.00s] budgets: iters=8 tokens=20,000 cost=$0.003 wall=90s
[agent   0.97s] lap 1: prompt 713 + completion 32 tok (0.97s) | run total 745/20,000 tok, $0.00004/$0.003
[agent   0.97s]   -> get_employee_record({"employee_id": "E1009"})
[agent   1.54s] lap 2: prompt 895 + completion 44 tok (0.57s) | run total 1,684/20,000 tok, $0.00012/$0.003
[agent   1.54s]   -> get_jurisdiction_rules({"jurisdiction": "US-CA", "topic": "parental_leave"})
[agent   2.07s] lap 3: model produced an invalid tool call (Error code: 400 - ... attempted to call too...) - retrying
[agent   3.96s] lap 4: model produced an invalid tool call (...) - retrying
[agent   4.46s] lap 5: model produced an invalid tool call (...) - retrying
[agent   5.41s] lap 6: model produced an invalid tool call (...) - retrying
[agent   5.90s] lap 7: model produced an invalid tool call (...) - retrying
[agent   6.45s] lap 8: prompt 1,013 + completion 33 tok (0.55s) | run total 2,730/20,000 tok, $0.00021/$0.003
[agent   6.45s]   -> search_handbook({"query": "parental leave adoption weeks US-CA"})
[agent   6.45s] STOP: budget 'max_iters' fired - 8 laps done, limit 8
[agent   6.45s] terminated cleanly after 8 laps; returning cannot_answer
```

What was rejected: gpt-oss leaked its channel marker into the tool name
(`attempted to call tool 'search_handbook<|channel|>commentary'`, Groq `tool_use_failed`).
**Only `max_iters` (or wall-clock) could catch this spin.** Rejected calls report no usage, so the
token and cost counters did not move for laps 3–7. The loop also retries the *identical*
request, which is why the same rejection repeats. Next fix, deliberately not applied (it would change
the system that was raced): feed the rejection back as a message so the retry differs.

The other three budgets are shown firing in `logs/test_budgets.log` (a stub model that never stops
calling tools, one budget tightened per run):

```
PASS  max_iters     stopped_by=max_iters     laps=3 tokens=6,660 cost=$0.00058 wall=0.00s
      STOP: budget 'max_iters' fired - 3 laps done, limit 3
PASS  max_tokens    stopped_by=max_tokens    laps=3 tokens=6,660 cost=$0.00058 wall=0.00s
      STOP: budget 'max_tokens' fired - spent 6,660 + next lap >= 3,130 (prompt ~2,874 re-sent) = 9,790 > limit 8,000
PASS  max_cost_usd  stopped_by=max_cost_usd  laps=3 tokens=6,660 cost=$0.00058 wall=0.00s
      STOP: budget 'max_cost_usd' fired - spent $0.00058 + next lap >= $0.00029 = $0.00087 > limit $0.00080
PASS  max_wall_s    stopped_by=max_wall_s    laps=3 tokens=6,660 cost=$0.00058 wall=0.90s
      STOP: budget 'max_wall_s' fired - stub call would run past the deadline

4/4 budgets fired as expected
```

## Third tool — `get_jurisdiction_rules` (full diff: `third_tool.diff`)

```diff
+ get_jurisdiction_rules(jurisdiction: enum["US-CA","US-NY","UK","DE","IN-KA"],
+                        topic:        enum["notice_period","annual_leave","sick_leave","parental_leave"])
+   "Return one jurisdiction's own rule for one topic - its PTO days per year, notice by tenure,
+    sick pay or parental leave - wherever it replaces or adds to the company-wide handbook.
+    Call once per jurisdiction you need a figure for. Exact lookup by enum - no free-text search,
+    no employee data."
- search_handbook:     "Search the HR handbook for any HR policy question: leave, PTO, notice periods,
-                       parental leave, benefits, local rules and employee entitlements."
+ search_handbook:     "... company-wide HR handbook: the default policies that apply in every location ...
+                       Holds no employee data and no jurisdiction's own figures - one country's or
+                       state's number is get_jurisdiction_rules' job, not this tool's."
- get_employee_record: "Look up an employee and their entitlements. ..."
+ get_employee_record: "Return one employee's HR record by employee ID ... When a fact depends on a linked
+                       record, call this again with that record's ID. Holds facts about the person
+                       only - no policy rules and no entitlement figures."
```

One job per tool: facts about a person / company-wide policy text / one jurisdiction's override.
In the Week 7 base, "entitlements" was claimed by both existing descriptions, and "local rules" was
folded into the handbook search with no tool of its own.
Every free parameter now has an enum or a pattern (`^E\d{4}$`).

The number that moved: the first cut of the third tool (`tool_specs/v2a_first_cut.json`) still
let q09 thrash. The agent called `search_handbook` twice for "annual leave UK PTO" and priced the
UK months at the company-wide 18 days (24 days, wrong). After sharpening the two *descriptions* —
no system-prompt change, no model change — it called `get_jurisdiction_rules("UK", "annual_leave")`
and answered 27.5. For the rehire questions (q08, q10), the `tenure_note` fix (below) turned a confident wrong "1 week"
into an abstention. Then one added sentence in `get_employee_record`'s description ("When a fact
depends on a linked record, call this again with that record's ID") made the agent fetch the prior
record. It passed both questions in both race trials (4/4).

## The workflow does the same task

- **Same inputs:** the identical question string.
- **Same model and params:** both call `app.agent.model.chat`.
- **Same tools:** `run_tool` from `app/agent/tools.py`.
- **Same output contract:** the same `OUTPUT_CONTRACT` text, and the same `parse_contract` parser
  for both final answers.
- **No loop:** six steps, each run exactly once, in fixed order (see the `workflow.py` docstring).
  Step 4's *arguments* depend on step 3's record; the step sequence never changes. Its one extra
  LLM call is a JSON-mode router that picks the topic enum and the handbook keywords.

## What changed during the work (so the numbers can be trusted)

- **No Week 7 base agent existed in the repo.** The base loop and its two tools were built here too;
  `tool_specs/week7_base.json` is that two-tool starting point, written for this diff.
- **Data fix before the race counted.** The first race run was stopped after q01–q02 because *both*
  systems answered q01 with 0 weeks. The US-NY/US-CA notice rules said "no notice required from either
  party" without saying that resignation notice is still set by the handbook (the UK rule already said
  so). Both US rules were fixed and the race restarted from zero.
- **Tool fix before the race.** For rehires, `get_employee_record` used to return
  `complete_years_of_service: 0` computed from the current stint alone — a wrong fact that both
  systems trusted. It now returns `null` plus a `tenure_note` naming the prior record.
- **Grading fix after trial 1.** q03 required citing `HB-01-2`, but the US-NY rule restates the same
  5-day carry-over cap, and both systems cited that rule with the correct figure. `must_cite` now
  accepts either. Every row is re-graded from the frozen answers, so trial 1 counts under the same
  grader (that is why `logs/race_trial1_stdout.log` still shows q03 as FAIL).
- **Limits:** n = 10 questions × 2 trials. One question is 5 points of pass rate. The records and
  jurisdiction rules are synthetic (loosely modelled on local statute, not legal advice). Handbook
  search is local BM25, not the Pinecone index, so the race measures the loop rather than network
  jitter or the Week 3/5 namespace contamination.
