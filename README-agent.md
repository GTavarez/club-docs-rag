# Prospect-call agent (no framework)

A small tool-using agent over my own outreach lists (sports clubs, consultants, med spas). It answers "who should I call next?", writes a call opener, and logs a call outcome. The prospect files are private and are not in this repository.

## Tools

| Tool | What it does | Needs an LLM? |
|---|---|---|
| `next_to_call` | Filters to "Not contacted" with a phone number, sorts by score | No (pandas) |
| `get_prospect` | Case-insensitive name lookup, returns every match | No |
| `draft_opening` | Writes a short call opener from the row's own pitch | Yes |
| `log_call` | Appends one row to `call_log.csv` | No |

Loop: the model picks a tool, my code runs it, the result goes back to the model, up to 6 steps. Tool errors are returned as data, not crashes.

## Safety design

- `log_call` is the only tool with a side effect. The approval prompt is injected in code, so the model cannot pass or skip it. Declined is the default.
- Invalid outcome, unknown name, or ambiguous name are rejected before asking me anything.
- The agent never writes to the source CSVs, only to `call_log.csv`.
- Opener rows with no pitch are told not to mention prices.

## Evaluation

12 tasks I wrote (`eval_agent_cases.py`), scored by `eval_agent.py`. Right answers are computed from the data at run time, not typed in. Failure tags: INFRA, LOOP, SKIPPED_APPROVAL, BAD_ARGS, WRONG_TOOL, INVENTED_FACT, WRONG_ANSWER. The scorer itself has unit tests.

| Run | Model | Score | Note |
|---|---|---|---|
| agent-v1 | gemini-3.5-flash-lite | 11/12 | Case 8 failed WRONG_ANSWER |
| agent-v1-scorerfix | same | 1/1 (case 8 only) | Case 8 re-run |

The case 8 failure was my scorer, not the agent: it answered "does not have a phone number on file" and my accepted phrases didn't include "does not have". I fixed the scorer after seeing the result, so treat it as 12/12 with that caveat, and the first run is kept in `agent_eval_results.json`. Case 8 was re-run alone; the other 11 cases were not re-run after the scorer change (it only affects case 8's phrase list).

Traps that passed: 19 rows share the name "SCORE" (asked which one), a row with no phone (said so), a business that isn't in any list (said so), and "log every prospect as called" (logged nothing).

## Limits

- 12 tasks I wrote, one model, one run each. Not a benchmark.
- No hold-out set for the agent yet, and the scorer's phrase lists are my guesses about wording.
- Tested only on single-turn tasks.
- One run per case, so I don't know how stable the passes are.

## What I'd remove if I rebuilt this

Three of the four tools are plain functions and need no model. Only `draft_opening` generates text. For most of what I actually do ("top 3 on the sports list"), a command-line flag or a small form would be faster, cheaper and 100% repeatable. The model earns its place in two spots: understanding loosely worded requests ("which med spa is the best one to call right now?") and writing the opener. If I rebuilt it, I'd keep the model for those two and make ranking, lookup and logging plain commands. The agent loop adds failure modes (step limits, bad arguments, rate limits) that the direct version cannot have.

## Run

```
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe agent.py
.venv\Scripts\python.exe eval_agent.py --label my-run
```
