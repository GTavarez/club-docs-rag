r"""Tests for the agent loop. The 'model' is a script, so no API calls are made.

Run:  .venv\Scripts\python.exe -m pytest test_agent.py -v
"""
import json

import pandas as pd
import pytest

from agent import TOOL_SCHEMAS, run_agent, run_tool


# ---- a scripted fake model -------------------------------------------------
def call(name, args, id="c1"):
    """A model reply that asks for one tool."""
    arguments = args if isinstance(args, str) else json.dumps(args)
    return {"content": None,
            "tool_calls": [{"id": id, "name": name, "arguments": arguments}],
            "raw": {"role": "assistant", "content": None, "tool_calls": [
                {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}]}}


def say(text):
    """A model reply that is a final answer."""
    return {"content": text, "tool_calls": [], "raw": {"role": "assistant", "content": text}}


class Script:
    """chat_fn that replays replies in order and remembers what it was sent."""
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)


class FakeLLM:
    def __call__(self, prompt):
        return "Hi, this is Gisell. Quick question?"


@pytest.fixture
def df():
    cols = ["list", "name", "type", "score", "phone", "address", "hook", "pitch", "status", "notes"]
    rows = [("sports", f"Club {i}", "Softball", 100 - i, f"({i:03d}) 000-0000", "Somewhere",
             "No website", "Pitch", "Not contacted", None) for i in range(30)]
    rows.append(("medspa", "Spa One", None, 50, "(999) 999-9999", "Elsewhere", "No booking", None, "Not contacted", None))
    return pd.DataFrame(rows, columns=cols)


def deny(text):
    return False


def allow(text):
    return True


# ---- the loop --------------------------------------------------------------
def test_tool_then_final_answer(df):
    model = Script(call("next_to_call", {"list_name": "sports", "n": 2}), say("Call Club 0 first."))
    out = run_agent("who next?", df=df, chat_fn=model, approve=deny, verbose=False)
    assert out["status"] == "done" and out["answer"] == "Call Club 0 first."
    assert [s["tool"] for s in out["steps"]] == ["next_to_call"]
    assert [r["name"] for r in out["steps"][0]["result"]] == ["Club 0", "Club 1"]


def test_tool_result_is_fed_back_to_the_model(df):
    model = Script(call("get_prospect", {"name": "Club 3"}), say("ok"))
    run_agent("find club 3", df=df, chat_fn=model, approve=deny, verbose=False)
    last_messages = model.seen[1]                         # what the model saw on its second turn
    assert last_messages[-1]["role"] == "tool" and "Club 3" in last_messages[-1]["content"]


def test_loop_stops_at_the_step_limit(df):
    model = Script(*[call("next_to_call", {}, id=f"c{i}") for i in range(10)])
    out = run_agent("go", df=df, chat_fn=model, approve=deny, max_steps=3, verbose=False)
    assert out["status"] == "max_steps" and out["answer"] is None and len(out["steps"]) == 3


def test_unknown_tool_is_reported_and_the_loop_continues(df):
    model = Script(call("delete_everything", {}), say("sorry"))
    out = run_agent("x", df=df, chat_fn=model, approve=deny, verbose=False)
    assert "unknown tool" in out["steps"][0]["result"]["error"] and out["status"] == "done"


def test_bad_json_arguments_are_reported_not_raised(df):
    model = Script(call("get_prospect", "{not json"), say("sorry"))
    out = run_agent("x", df=df, chat_fn=model, approve=deny, verbose=False)
    assert "not valid JSON" in out["steps"][0]["result"]["error"] and out["status"] == "done"


def test_missing_required_argument_is_reported(df):
    result = run_tool("get_prospect", {}, df, deny)
    assert "missing required argument" in result["error"]


def test_wrong_argument_type_is_reported(df):
    assert "integer" in run_tool("next_to_call", {"n": "five"}, df, deny)["error"]


def test_n_is_capped_at_ten(df):
    assert len(run_tool("next_to_call", {"n": 999}, df, deny)) == 10


def test_unknown_list_comes_back_as_an_error_not_a_crash(df):
    assert "Unknown list" in run_tool("next_to_call", {"list_name": "nope"}, df, deny)["error"]


def test_draft_opening_uses_the_injected_llm(df):
    out = run_tool("draft_opening", {"name": "Club 1"}, df, deny, llm=FakeLLM())
    assert out["status"] == "ok" and out["opening"].startswith("Hi, this is Gisell")


# ---- approval: the model must not be able to get around it ------------------
def test_log_call_declined_by_the_human_writes_nothing(df, tmp_path):
    path = tmp_path / "log.csv"
    out = run_tool("log_call", {"name": "Club 1", "outcome": "spoke"}, df, deny, log_path=str(path))
    assert out["status"] == "declined" and not path.exists()


def test_model_cannot_pass_its_own_approval(df, tmp_path):
    path = tmp_path / "log.csv"
    sneaky = {"name": "Club 1", "outcome": "spoke", "approve": True, "approved": True}
    out = run_tool("log_call", sneaky, df, deny, log_path=str(path))
    assert out["status"] == "declined" and not path.exists()      # the human function still decided


def test_approved_log_call_writes_the_row(df, tmp_path):
    path = tmp_path / "log.csv"
    out = run_tool("log_call", {"name": "Club 1", "outcome": "interested", "note": "send info"}, df, allow,
                   log_path=str(path))
    assert out["status"] == "logged" and "Club 1" in path.read_text(encoding="utf-8")


def test_invalid_outcome_never_reaches_the_human(df, tmp_path):
    asked = []
    out = run_tool("log_call", {"name": "Club 1", "outcome": "maybe"}, df,
                   lambda t: asked.append(t) or True, log_path=str(tmp_path / "log.csv"))
    assert out["status"] == "invalid_outcome" and asked == []


def test_full_run_with_a_declined_log_call_never_writes(df, tmp_path):
    path = tmp_path / "log.csv"
    model = Script(call("log_call", {"name": "Club 1", "outcome": "spoke"}), say("It was not logged."))
    out = run_agent("log it", df=df, chat_fn=model, approve=deny, verbose=False, log_path=str(path))
    assert out["steps"][0]["result"]["status"] == "declined" and not path.exists()


def test_tool_schemas_never_expose_an_approve_parameter():
    for schema in TOOL_SCHEMAS:
        assert "approve" not in schema["function"]["parameters"]["properties"]
