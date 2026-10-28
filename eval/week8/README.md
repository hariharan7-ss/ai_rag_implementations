# eval/week8 — Outcome vs trajectory for the HR agent (Week 8, Task Set C)

```powershell
python eval/week8/trajectory_eval.py --score before                    # offline: score the frozen Week-7 agent runs
python eval/week8/trajectory_eval.py --run after --trials 2            # agent over the 10 questions x 2 trials
python eval/week8/trajectory_eval.py --score after --compare before    # tables + per-mode regression + price
python eval/week8/fault_injection.py                                   # forced-failure A/B of the mitigation
```

Files: `expected_paths.json` (10 expected sequences), `trajectory_eval.py`, `fault_injection.py`,
`mitigation.diff`, `trajectory_results_{before,after}.csv`, `runs/{before,after,fault}.json`.

## 1. Expected sequences (`expected_paths.json`)

Tokens: `R:<id>` get_employee_record, `J:<jur>:<topic>` get_jurisdiction_rules, `H` search_handbook.
Each case lists every accepted core path. A run matches when its non-H calls equal a core path in
order and it made at least `h_min` handbook searches. **H position is never asserted** and extra
H calls don't fail the path (charged to step efficiency instead), so reading the handbook before
or after the record is not scored as an error.

| case | accepted paths | why more than one |
| --- | --- | --- |
| q01, q02 | R+H, or R+J+H | the US override only defers to the handbook; checking it first is legitimate |
| q03 | R+H, or R+J | the 5-day cap is stated in both the handbook and the US-NY rule |
| q08, q10 | R(E1012/E1015), then {R(prior), J} in either order | prior record and rule are independent |
| q09 | R, then {J:DE, J:UK} in either order | both rules needed |
| q04–q07 | single path | the rule exists in one place only |

## 2. Trajectory numbers

Same agent code, same 10 questions, 2 trials each (20 runs per column).

| | before (Week 7 code, 20 runs) | after (mitigation, 20 runs) |
| --- | --- | --- |
| tool-choice accuracy | 95% (19/20) | 100% (20/20) |
| argument validity (real **and** grounded) | 100% (55/55) | 100% (61/61) |
| step efficiency (taken / needed) | 1.19 mean, 1.50 max | 1.33 mean, 1.50 max |
| cost / question | **p50 $0.000302, max $0.000507** | **p50 $0.000358, max $0.000548** |
| tokens / question | p50 4,116, max 6,390 | p50 4,370, max 6,220 |
| latency / question | p50 2.84 s, max 6.18 s | p50 2.45 s, max 4.40 s |

Argument validity counts a call as valid only if the employee id exists **and** appeared in the
question or an earlier tool result, and the jurisdiction came from a fetched record. Both columns
are 100%: the agent never made up an id, section or jurisdiction.

## 3. The gap (outcome pass rate − trajectory pass rate)

| | outcome pass | trajectory pass | **gap** |
| --- | --- | --- | --- |
| before | 80% (16/20) | 90% (18/20) | **−10 pts** |
| after | 100% (20/20) | 100% (20/20) | **0 pts** |

The Week-7 agent's gap is **negative**: the trajectory eval is more forgiving than the outcome
eval here. The two evals fail on different runs, so neither replaces the other. q05 passes
trajectory (the correct `R:E1005 → J:UK:notice_period`) but fails outcome twice by reading
`complete_months_of_service` (112) as weeks.

**Right answer, wrong path (before, trial 1, q02):** answered the correct 10 weeks via
`R:E1009 → J:US-CA:parental_leave → [rejected] → [rejected] → H`. Outcome PASS, trajectory FAIL:
two laps were rejected by the API (`tool 'search_handbook<|channel|>commentary' was not in
request.tools`) before the handbook search finally went through. The same question on trial 2
spun until `max_iters` fired. A pass here is luck on a path that sometimes does not finish.

## 4. The one mitigation: re-planning on a rejected tool call

Top mode: `rejected_tool_call` (8 rejected laps in 2 of 20 runs; one run hit `max_iters`).
Change (`mitigation.diff`): on a rejected call, append the rejection to the messages so the retry
differs. Week 7 re-sent a byte-identical request, and with a pinned seed the identical bad
generation came back.

**The natural re-run does not prove it.** Rejections went 8 → 0, but the feedback branch never
executed in the 20 "after" runs. The failure is intermittent (it hit q02 only), and it did not
recur, so that 8 → 0 is chance and not evidence. The outcome jump (80% → 100%) is q05 no longer
misreading months as weeks, which this change does not touch. Treat both as unattributed run-to-run variation.

**So the mitigation was measured under forced failure** (`fault_injection.py`): the 2nd model call
of every run is rejected, and a retry with identical messages is rejected again (the
Week-7 pinned-seed behaviour). Any retry with different messages goes to the real model. The
injection is the only simulated part, and it makes the without-feedback arm unable to recover by
construction. The with-feedback arm is the real test.

| 10 questions, 1 rejection injected per run | without feedback (Week 7) | with feedback (mitigation) |
| --- | --- | --- |
| runs ending in `budget_hit` (max_iters) | **10 / 10** | **0 / 10** |
| `rejected_tool_call` laps | 70 | 10 (the injected one only) |
| runs that returned an answer | 0 / 10 | 10 / 10 |
| correct answers | 0 / 10 | 7 / 10 |
| path matches an accepted sequence | 0 / 10 | 10 / 10 |

**Top mode before → after: 10 runs spinning to max_iters → 0; 70 rejected laps → 10.**

**Price paid, measured:**

| | without feedback | with feedback | price |
| --- | --- | --- | --- |
| tokens / question p50 (max) | 745 (759) | 4,611 (6,357) | +3,866 tokens p50 (but the first arm returns no answer) |
| cost / question p50 (max) | $0.000063 ($0.000064) | $0.000342 ($0.000505) | +$0.000279 p50 |
| latency p50 (max) | 0.63 s (0.93 s) | 2.51 s (3.92 s) | +1.88 s p50 |
| laps (mean) | 8.0 | 4.8 | |

The honest comparison for cost is against a run that works. Against the natural "after" column
(p50 $0.000358, 2.45 s) a recovered run costs about the same, since recovery adds one lap. The
without-feedback arm looks cheap only because it fails fast.

**The other price is accuracy.** Recovered runs got **3 of 10 wrong figures**: q02 gave 8 weeks
(skipped the 2026 addendum and used the original policy), q08 gave 7 weeks, and q10 gave 4 *weeks*
instead of 4 *months*. All three still followed an accepted path. After the interruption the model
was less careful with figures and the sequence eval cannot see that. An abstention is safer than a
confident wrong figure for HR, so ship this behind a check on the figure, not alone. n = 10 and one
fault pattern, so the 7/10 is a rough figure.

## 5. Regression check: per-mode counts, natural runs before → after

| mode | before | after | verdict |
| --- | --- | --- | --- |
| rejected_tool_call | 8 | 0 | improved (chance, see §4) |
| wrong_path | 1 | 0 | improved (that one run was the q02 spin) |
| budget_hit | 1 | 0 | improved (same run) |
| ungrounded_args | 0 | 0 | checked, still zero |
| duplicate_call | 0 | 0 | checked, still zero |

Nothing got worse in the natural runs, and no new mode appeared there. Under forced failure the
mitigation did create one new problem: **confident wrong figures after recovery** (3 of 10).
That mode is not in the scorer's taxonomy yet; I'd name it `figure_after_recovery` and add it next.
Step efficiency drifted 1.19 → 1.33 in the natural runs: q03, q05 and q10 now take an extra
handbook search. It is not a failure but it is a small cost, and it is not explained by the
mitigation, since the branch never ran.

## Limits

- 20 runs per column, and the failure seen before occurred in 2 runs. Pass-rate differences of
  ±10 pts are 2 runs.
- The Groq key was replaced between the "before" and "after" runs. The model backend may also have
  changed (q05 flipped, and several runs gained a handbook search), so before/after in §2–§3 is not a controlled comparison.
- `feedback_on_reject=False` stays in `agent.py` only so the fault A/B can run; the default is on.
- The bonus (injection) was not attempted.
