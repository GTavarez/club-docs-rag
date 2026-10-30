r"""Day 3 - score the agent against eval_agent_cases.py.

Run (about 10 minutes on the free tier; run it ONCE, don't loop):
  .venv\Scripts\python.exe eval_agent.py --label agent-v1
  .venv\Scripts\python.exe eval_agent.py --label agent-v1 --only 7      # one case

Failure tags (the first that applies, most serious first):
  INFRA              the model call never completed (rate limit/outage) -> run is NOT a valid measurement
  LOOP               hit the step limit without answering
  SKIPPED_APPROVAL   a call was logged without a human approving it
  BAD_ARGS           a tool returned an error (unknown tool, bad or missing arguments)
  WRONG_TOOL         a required tool was not used, a forbidden one was, or a tool was used too often
  INVENTED_FACT      the answer has a phone number or $ amount that no tool returned (or banned text)
  WRONG_ANSWER       a required fact is missing from the answer
"""
import argparse
import json
import os
import re
import tempfile
import time
from datetime import datetime

from agent import run_agent
from eval_agent_cases import CASES
from tools import load_lists, next_to_call

PHONE_RE = re.compile(r"\(\d{3}\)\s?\d{3}-\d{4}")
MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
PRIORITY = ["INFRA", "LOOP", "SKIPPED_APPROVAL", "BAD_ARGS", "WRONG_TOOL", "INVENTED_FACT", "WRONG_ANSWER"]


class Approver:
    """Stands in for the human. Records every approval request and answers as the case says."""
    def __init__(self, answer):
        self.answer, self.asked = answer, []

    def __call__(self, text):
        self.asked.append(text)
        return self.answer


def truth_values(df):
    """Right answers computed from the data, so the eval can't drift from the files."""
    def names(**kw):
        return [r["name"] for r in next_to_call(df, **kw)]
    nophone = df[df["phone"].isna()]
    return {
        **{f"sports{i + 1}": n for i, n in enumerate(names(list_name="sports", n=3))},
        **{f"consultants{i + 1}": n for i, n in enumerate(names(list_name="consultants", n=2))},
        "medspa1": names(list_name="medspa", n=1)[0],
        "cricket1": names(type="Cricket", n=1)[0],
        "softball1": names(type="Softball", n=1)[0],
        "nophone1": nophone.iloc[0]["name"] if len(nophone) else "NO-SUCH-ROW",
    }


def fill(text, truth):
    for key, value in truth.items():
        text = text.replace("{" + key + "}", value)
    return text


def has_fact(answer, fact):
    return any(alt.strip().lower() in answer.lower() for alt in fact.split("|"))


def check_run(case, run, approver, rows_written, truth):
    """Returns (tag or None, [problems]). Pure function: easy to test without any API."""
    steps, answer = run["steps"], run["answer"] or ""
    used = [s["tool"] for s in steps]
    tool_text = json.dumps([s["result"] for s in steps], default=str)
    problems = {}                                  # tag -> list of messages

    def add(tag, msg):
        problems.setdefault(tag, []).append(msg)

    if run["status"] == "max_steps":
        add("LOOP", f"hit the step limit after {len(steps)} tool calls")
    logged = [s for s in steps if s["tool"] == "log_call" and isinstance(s["result"], dict)
              and s["result"].get("status") == "logged"]
    if len(logged) > len(approver.asked) or rows_written > sum(1 for _ in logged):
        add("SKIPPED_APPROVAL", f"{len(logged)} logged, {len(approver.asked)} approvals asked, {rows_written} rows written")
    for s in steps:
        if isinstance(s["result"], dict) and "error" in s["result"]:
            add("BAD_ARGS", f"{s['tool']}: {s['result']['error']}")
    for tool in case["expected_tools"]:
        if tool not in used:
            add("WRONG_TOOL", f"never used {tool}")
    for tool in case["forbidden"]:
        if tool in used:
            add("WRONG_TOOL", f"used forbidden tool {tool}")
    for tool, cap in case.get("max_calls", {}).items():
        if used.count(tool) > cap:
            add("WRONG_TOOL", f"{tool} used {used.count(tool)}x (max {cap})")

    for phone in PHONE_RE.findall(answer):
        if phone not in tool_text:
            add("INVENTED_FACT", f"phone number {phone} was never returned by a tool")
    for money in MONEY_RE.findall(answer):
        if money.replace(" ", "") not in tool_text.replace(" ", ""):
            add("INVENTED_FACT", f"amount {money} was never returned by a tool")
    for banned in case["must_not_say"]:
        if banned.lower() in answer.lower():
            add("INVENTED_FACT", f"answer contains banned text {banned!r}")
    for fact in case["must_say"]:
        if not has_fact(answer, fill(fact, truth)):
            add("WRONG_ANSWER", f"answer is missing {fill(fact, truth)!r}")

    for tag in PRIORITY:
        if tag in problems:
            return tag, [m for t in PRIORITY for m in problems.get(t, [])]
    return None, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="agent-v1")
    ap.add_argument("--only", type=int, help="run just this case number (1-based)")
    ap.add_argument("--pause", type=int, default=10, help="seconds between cases (free-tier rate limit)")
    args = ap.parse_args()

    from rag import CHAT_MODEL, PROVIDER, RETRYABLE      # imported late so tests never need the RAG app

    df, truth = load_lists(), None
    truth = truth_values(df)
    results = []
    todo = [(i, c) for i, c in enumerate(CASES, 1) if args.only in (None, i)]
    for n_done, (i, case) in enumerate(todo):
        task = fill(case["task"], truth)
        approver = Approver(case["approve"])
        log_path = os.path.join(tempfile.mkdtemp(), "call_log.csv")     # never your real call_log.csv
        try:
            run = run_agent(task, df=df, approve=approver, verbose=False, log_path=log_path)
            written = max(0, sum(1 for _ in open(log_path, encoding="utf-8")) - 1) if os.path.exists(log_path) else 0
            tag, problems = check_run(case, run, approver, written, truth)
        except RETRYABLE as e:                                           # quota/outage: NOT a model failure
            run, tag, problems = {"status": "infra", "answer": None, "steps": []}, "INFRA", [f"{type(e).__name__}: request never completed"]
        except Exception as e:
            run, tag, problems = {"status": "crash", "answer": None, "steps": []}, "BAD_ARGS", [f"crashed: {type(e).__name__}: {e}"]
        ok = tag is None
        results.append({"n": i, "task": task, "pass": ok, "tag": tag, "problems": problems,
                        "tools": [s["tool"] for s in run["steps"]], "answer": run["answer"],
                        "approvals_asked": len(approver.asked)})
        print(f"{'PASS' if ok else 'FAIL'}  {i:>2}. {task}")
        if not ok:
            print(f"        [{tag}] " + "; ".join(problems))
            print(f"        tools: {[s['tool'] for s in run['steps']]}")
            print(f"        answer: {(run['answer'] or '')[:200]!r}")
        if n_done < len(todo) - 1:
            time.sleep(args.pause)

    n = sum(r["pass"] for r in results)
    counts = {t: sum(r["tag"] == t for r in results) for t in PRIORITY}
    if counts["INFRA"]:
        print(f"\n*** {counts['INFRA']} case(s) never reached the model. This run is NOT a valid measurement. ***")
    print(f"\nScore: {n}/{len(results)}   {({t: c for t, c in counts.items() if c})}   (label={args.label}, provider={PROVIDER}, model={CHAT_MODEL})")

    try:
        history = json.load(open("agent_eval_results.json", encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    history.append({"when": datetime.now().isoformat(timespec="seconds"), "label": args.label,
                    "provider": PROVIDER, "chat_model": CHAT_MODEL, "score": n, "total": len(results),
                    "tags": counts, "valid": counts["INFRA"] == 0, "only": args.only, "cases": results})
    json.dump(history, open("agent_eval_results.json", "w", encoding="utf-8"), indent=2)
    print("Saved to agent_eval_results.json")


if __name__ == "__main__":
    main()
