# Agent + Evals Sprint (3 days, about 4-5 hours total)

Goal (doc 06, project 2.2): a raw-Python agent loop (no framework) with 4+ tools and a human
approval step, plus an agent eval that shows where it fails. Outcome by Sunday: it completes
8 of 10 tasks on real inputs, failures are visible (not silent), and a paragraph:
"what I'd remove if I rebuilt this."

Rule 3 (you go first): YOU write each tool and the loop; Claude reviews after your attempt.

## The agent: a call-list assistant
Your three lists (sports, consultants, med spas/clinics) all have: a score, a business name, a
phone, a call hook / pitch, STATUS and NOTES. None has dates. So the job is "who do I call next,
what do I say, and log what happened" - not "overdue follow-ups".

Tools (plain functions; each gets a pytest test that needs no LLM):
  1. next_to_call(list_name, n=5, type=None)  read-only. Highest score with status "Not contacted".
  2. get_prospect(name)                        read-only. Returns the row; handles duplicates (see below).
  3. draft_opening(name)                       LLM makes a 2-3 sentence call opener from the row's hook
                                               and pitch. Saves nothing. Must not invent facts.
  4. log_call(name, outcome, note)             ** CONSEQUENTIAL ** appends to call_log.csv ONLY after
                                               you type y. NEVER edits the source lists.
  5. search_club_docs(question)  (optional)    wraps ask_json() from the club app. Use case: "show a
                                               prospect an example of the AI assistant answering a
                                               refund question".

## Real data traps already in your files (use them as test cases)
  - "SCORE" appears 15+ times in consultant_prospects.csv -> get_prospect("SCORE") is ambiguous.
    Decide: return all matches, or ask for a city. Never silently pick one.
  - Placeholders instead of blanks: "Check website", "Not found", "NO WEBSITE", "N/A" (rating).
  - Blank phones, and a malformed email ("%20info@lenserfgroup.com").
  - Score columns have different names in each file; the sports file has an extra SITE QUALITY column.
  - The sports pitch says "Empire State Huskies, 80 players, $2,500 + $399/month". A draft must
    use only facts in the row. Anything else is an invention -> WRONG_ANSWER.
  - McKinsey/BCG sit in the consultant list at low scores. Do you want those ever called?

## Safety first (2 minutes)
  Put the 3 CSVs in claude-quickstart\data_private\ (not in the repo):
    mkdir data_private
    (move/copy your 3 files in; keep their names)
  Add to .gitignore:  data_private/   call_log.csv   users_log.csv   drafts.csv   followup.csv
  git checkout -b agent-sprint        # experiments live on a branch; main stays safe

## Day 1 - tools + tests (about 1.5 h)  -> tools.py, test_tools.py
  1. Load all three lists into one common shape: list, name, score, phone, hook, pitch, status, notes.
     Decide how placeholders ("Not found", "N/A", "Check website") are represented (None).
  2. Write the 4 tools. Write down what each returns on bad input (unknown name, n=0, empty list).
  3. log_call asks via approve(text) -> bool, a separate function so tests can fake it.
  4. 6+ pytest tests, no LLM, including the ambiguous "SCORE" case.
  Stop when: pytest passes and you can call each tool by hand.

## Day 2 - the agent loop (about 1.5 h)  -> agent.py
  1. Loop: messages + tool schemas -> model returns tool_calls -> run tool -> append result ->
     repeat. Hard cap: 6 steps (loops must end, loudly).
  2. Use rag.chat() with tools=[...] (Gemini's OpenAI-compatible endpoint supports it).
  3. Print every step: tool name, arguments, result.
  4. Try 5 tasks by hand and write down what breaks before fixing anything.
  Stop when: "Who should I call next on the sports list?" works end to end.

## Day 3 - agent evals (about 1.5-2 h)  -> eval_agent_cases.py, eval_agent.py
  1. Write 12 tasks yourself: 4 call-next, 2 opener, 2 data-trap (SCORE, missing phone),
     1 mixed-list, 1 club-docs demo, 1 must-not-act ("log everything as called"), 1 impossible.
  2. Each case: task text, expected tools (set), must_approve (bool), facts the answer must
     contain, and facts it must NOT contain (invented prices, names).
  3. Tag failures: WRONG_TOOL, SKIPPED_APPROVAL, LOOP (hit 6 steps), BAD_ARGS, WRONG_ANSWER,
     INVENTED_FACT, INFRA.
  4. Run ONCE (about 4 LLM calls per task, 5 requests/min on flash-lite: expect 10+ minutes).
  5. Write the paragraph in README-agent.md: what I'd remove, which steps didn't need an LLM.

## Rules for the 3 days
  - One change at a time, re-score after each. Commit before every experiment.
  - If you only have 3 hours: do Day 1 and Day 2 and write 6 eval tasks. Shrink, never skip.
  - Don't touch the club-docs app or its evals this week.
  - The data is real business information. It stays out of the public repo, and the agent must
    never write to the source CSVs.
