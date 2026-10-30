"""Day 2 - the agent loop. Raw Python, no framework.

The model never touches your data directly. Each turn it either:
  (a) asks to run a tool  -> we run it, show it, and feed the result back, or
  (b) answers in plain text -> we stop.
It is capped at MAX_STEPS tool rounds, so a loop always ends, loudly.

Run:  .venv\\Scripts\\python.exe agent.py "Who should I call next on the sports list?"
"""
import json
import sys

from tools import (approve_cli, draft_opening, get_prospect, load_lists, log_call,
                   next_to_call, OUTCOMES)

MAX_STEPS = 6
MAX_N = 10          # never hand the model more than 10 prospects at once

SYSTEM_PROMPT = (
    "You are a call-list assistant for Gisell. You help her decide who to call next, "
    "prepare what to say, and record the result of calls.\n"
    "Rules:\n"
    "- Use the tools for every fact about a prospect. Never guess phone numbers, scores, pitches or status.\n"
    "- If a tool says a name is ambiguous or not found, tell Gisell and ask; do not pick one yourself.\n"
    "- Only call log_call when Gisell clearly asks you to record a call outcome. Gisell is asked to approve "
    "each one, so never say a call was logged unless log_call returned status 'logged'.\n"
    "- Quote openings exactly as draft_opening returned them. Do not add facts, prices or promises.\n"
    "- Keep answers short."
)

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "next_to_call",
        "description": "Highest-scoring prospects not yet contacted that have a phone number.",
        "parameters": {"type": "object", "properties": {
            "list_name": {"type": "string", "enum": ["sports", "consultants", "medspa"],
                          "description": "Which list. Omit to search all lists."},
            "n": {"type": "integer", "description": "How many to return (1-10). Default 5."},
            "type": {"type": "string", "description": "Optional exact type, e.g. 'Softball' or 'Accountant/CPA'."},
        }}}},
    {"type": "function", "function": {
        "name": "get_prospect",
        "description": "Look up one business by exact name. Returns every match (several = ambiguous).",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "draft_opening",
        "description": "Write a short spoken call opening for one prospect, from its own hook and pitch only.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "log_call",
        "description": "Record the outcome of a call in call_log.csv. Gisell must approve each entry. "
                       "Only use when she asks to record a result.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "outcome": {"type": "string", "enum": sorted(OUTCOMES)},
            "note": {"type": "string"},
        }, "required": ["name", "outcome"]}}},
]


def run_tool(name, args, df, approve, llm=None, log_path=None):
    """Run one tool call. NEVER raises: any problem comes back as {"error": ...} so the model
    (and the printed trace) can see it. The approve function is injected here, so the model
    cannot choose it, skip it, or pass its own."""
    if not isinstance(args, dict):
        return {"error": "arguments must be a JSON object"}
    try:
        if name == "next_to_call":
            n = args.get("n", 5)
            if not isinstance(n, int):
                return {"error": "n must be an integer"}
            return next_to_call(df, list_name=args.get("list_name"), n=min(max(n, 1), MAX_N),
                                type=args.get("type"))
        if name == "get_prospect":
            return get_prospect(df, args["name"])
        if name == "draft_opening":
            return draft_opening(df, args["name"], llm=llm)
        if name == "log_call":
            kwargs = {"path": log_path} if log_path else {}
            return log_call(df, args["name"], args["outcome"], args.get("note", ""),
                            approve=approve, **kwargs)
        return {"error": f"unknown tool {name!r}. Available: next_to_call, get_prospect, draft_opening, log_call"}
    except KeyError as e:
        return {"error": f"missing required argument {e}"}
    except Exception as e:                                     # a tool crash is data, not a stop
        return {"error": f"{type(e).__name__}: {e}"}


def default_chat(messages, tools):
    """One model call through the same OpenAI-compatible client the RAG app uses."""
    from rag import CHAT_MODEL, chat
    r = chat(model=CHAT_MODEL, temperature=0, messages=messages, tools=tools)
    m = r.choices[0].message
    return {
        "content": m.content,
        "tool_calls": [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
                       for c in (m.tool_calls or [])],
        "raw": m.model_dump(exclude_none=True),     # keep exactly what the model sent (incl. extras)
    }


def _short(obj, limit=300):
    text = json.dumps(obj, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


def run_agent(task, df=None, chat_fn=None, approve=approve_cli, llm=None,
              max_steps=MAX_STEPS, verbose=True, log_path=None):
    """Returns {"status": "done" | "max_steps", "answer": str | None, "steps": [...]}."""
    df = load_lists() if df is None else df
    chat_fn = chat_fn or default_chat
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task}]
    steps = []

    for _ in range(max_steps):
        reply = chat_fn(messages, TOOL_SCHEMAS)
        if not reply["tool_calls"]:                                  # model answered: finished
            if verbose:
                print(f"\nANSWER: {reply['content']}")
            return {"status": "done", "answer": reply["content"], "steps": steps}

        messages.append(reply["raw"])
        for call in reply["tool_calls"]:
            try:
                args = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                args, result = None, {"error": "arguments were not valid JSON"}
            else:
                result = run_tool(call["name"], args, df, approve, llm=llm, log_path=log_path)
            steps.append({"tool": call["name"], "args": args, "result": result})
            if verbose:
                print(f"STEP {len(steps)}: {call['name']} {_short(args)}\n   -> {_short(result)}")
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(result, default=str)})

    if verbose:
        print(f"\nSTOPPED: hit the {max_steps}-step limit without a final answer.")
    return {"status": "max_steps", "answer": None, "steps": steps}


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or input("Task: ")
    run_agent(task)
