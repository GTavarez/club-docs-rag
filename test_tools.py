r"""Tests for the call-list tools. Uses tiny FAKE tables, not your real data.

Run:  .venv\Scripts\python.exe -m pytest test_tools.py -v
"""
import pandas as pd
import pytest

from tools import (build_opening_prompt, draft_opening, get_prospect, log_call,
                   next_to_call)


@pytest.fixture
def df():
    rows = [
        # list,         name,     type,                score, phone,           status
        ("sports",      "Alpha",  "Softball",          100,   "(111) 111-1111", "Not contacted"),
        ("sports",      "Bravo",  "Cricket",           90,    None,             "Not contacted"),  # no phone
        ("sports",      "Charlie", "Softball",         80,    "(333) 333-3333", "Called"),         # already contacted
        ("consultants", "Delta",  "Business Consultant", 70,  "(444) 444-4444", "Not contacted"),
        ("consultants", "SCORE",  "SCORE/SBA Mentor",  60,    "(555) 555-5555", "Not contacted"),
        ("consultants", "SCORE",  "SCORE/SBA Mentor",  50,    "(666) 666-6666", "Not contacted"),
        ("medspa",      "Echo",   None,                95,    "(777) 777-7777", "Not contacted"),
    ]
    return pd.DataFrame(rows, columns=["list", "name", "type", "score", "phone", "status"])


# ---- next_to_call: who should I call next? -------------------------------
def test_highest_score_first_and_only_not_contacted(df):
    names = [r["name"] for r in next_to_call(df)]
    assert names == ["Alpha", "Echo", "Delta", "SCORE", "SCORE"]   # Bravo (no phone), Charlie (called) excluded


def test_n_limits_the_results(df):
    assert [r["name"] for r in next_to_call(df, n=2)] == ["Alpha", "Echo"]


def test_n_zero_returns_empty_list(df):
    assert next_to_call(df, n=0) == []


def test_filter_by_list(df):
    assert [r["name"] for r in next_to_call(df, list_name="consultants")] == ["Delta", "SCORE", "SCORE"]


def test_filter_by_type(df):
    assert [r["name"] for r in next_to_call(df, type="Softball")] == ["Alpha"]


def test_unknown_list_raises_a_clear_error(df):
    with pytest.raises(ValueError) as err:
        next_to_call(df, list_name="nonsense")
    assert "sports" in str(err.value)          # the message must tell you the valid names


def test_results_are_plain_dicts_with_the_useful_fields(df):
    first = next_to_call(df, n=1)[0]
    assert isinstance(first, dict)
    assert {"name", "list", "score", "phone"} <= set(first)


def test_does_not_modify_the_table(df):
    before = df.copy()
    next_to_call(df)
    pd.testing.assert_frame_equal(df, before)


# ---- get_prospect: look up one business -----------------------------------
def test_get_prospect_finds_by_name_ignoring_case_and_spaces(df):
    found = get_prospect(df, "  alpha ")
    assert [r["name"] for r in found] == ["Alpha"]


def test_get_prospect_returns_every_match_when_ambiguous(df):
    found = get_prospect(df, "SCORE")
    assert len(found) == 2                      # never silently pick one of the two


def test_get_prospect_unknown_returns_empty_list(df):
    assert get_prospect(df, "Nobody Inc") == []


# ---- draft_opening and log_call use a fuller fake table --------------------
@pytest.fixture
def full():
    cols = ["list", "name", "type", "score", "phone", "address", "hook", "pitch", "status", "notes"]
    rows = [
        ("sports", "Alpha", "Softball", 100, "(111) 111-1111", "1 Main St", "No website", "Pitch A: $1,000 setup", "Not contacted", None),
        ("medspa", "Spa One", None, 90, "(222) 222-2222", "2 Oak Ave", "No online booking", None, "Not contacted", None),
        ("medspa", "Blank Co", None, 80, "(333) 333-3333", "3 Elm Rd", None, None, "Not contacted", None),
        ("consultants", "SCORE", "Mentor", 60, "(555) 555-5555", "Newark NJ", None, "Referral pitch", "Not contacted", None),
        ("consultants", "SCORE", "Mentor", 50, "(666) 666-6666", "Denver CO", None, "Referral pitch", "Not contacted", None),
    ]
    return pd.DataFrame(rows, columns=cols)


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return "Hi, this is Gisell. Quick question?"


# ---- draft_opening --------------------------------------------------------
def test_draft_opening_ok_uses_the_llm_and_returns_its_text(full):
    llm = FakeLLM()
    out = draft_opening(full, "Alpha", llm=llm)
    assert out["status"] == "ok" and out["opening"] == "Hi, this is Gisell. Quick question?"
    assert len(llm.prompts) == 1


def test_prompt_contains_the_row_facts_and_a_no_invention_rule(full):
    prompt = build_opening_prompt(get_prospect(full, "Alpha")[0])
    assert "No website" in prompt and "$1,000 setup" in prompt
    assert "Never invent" in prompt


def test_draft_opening_unknown_name_does_not_call_the_llm(full):
    llm = FakeLLM()
    assert draft_opening(full, "Nobody", llm=llm)["status"] == "not_found"
    assert llm.prompts == []


def test_draft_opening_ambiguous_name_does_not_call_the_llm(full):
    llm = FakeLLM()
    out = draft_opening(full, "SCORE", llm=llm)
    assert out["status"] == "ambiguous" and len(out["matches"]) == 2
    assert llm.prompts == []


def test_draft_opening_with_hook_but_no_pitch_forbids_offers(full):
    llm = FakeLLM()
    assert draft_opening(full, "Spa One", llm=llm)["status"] == "ok"
    assert "do NOT mention any price" in llm.prompts[0]


def test_draft_opening_with_nothing_to_say_does_not_call_the_llm(full):
    llm = FakeLLM()
    assert draft_opening(full, "Blank Co", llm=llm)["status"] == "no_material"
    assert llm.prompts == []


# ---- log_call: nothing is written without approval ------------------------
def test_log_call_without_approval_function_writes_nothing(full, tmp_path):
    path = tmp_path / "log.csv"
    assert log_call(full, "Alpha", "spoke", approve=None, path=str(path))["status"] == "declined"
    assert not path.exists()


def test_log_call_declined_writes_nothing(full, tmp_path):
    path = tmp_path / "log.csv"
    assert log_call(full, "Alpha", "spoke", approve=lambda text: False, path=str(path))["status"] == "declined"
    assert not path.exists()


def test_log_call_approved_writes_header_and_row_then_appends(full, tmp_path):
    path = tmp_path / "log.csv"
    assert log_call(full, "Alpha", "spoke", "asked for a quote", approve=lambda t: True, path=str(path))["status"] == "logged"
    log_call(full, "Spa One", "no_answer", approve=lambda t: True, path=str(path))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "timestamp,list,name,phone,outcome,note"
    assert len(lines) == 3 and "Alpha" in lines[1] and "no_answer" in lines[2]


def test_log_call_invalid_outcome_never_asks_for_approval(full, tmp_path):
    asked = []
    out = log_call(full, "Alpha", "maybe", approve=lambda t: asked.append(t) or True, path=str(tmp_path / "log.csv"))
    assert out["status"] == "invalid_outcome" and asked == []


def test_log_call_ambiguous_name_never_asks_and_writes_nothing(full, tmp_path):
    asked, path = [], tmp_path / "log.csv"
    out = log_call(full, "SCORE", "spoke", approve=lambda t: asked.append(t) or True, path=str(path))
    assert out["status"] == "ambiguous" and asked == [] and not path.exists()


def test_log_call_approval_text_names_the_prospect_and_outcome(full, tmp_path):
    seen = []
    log_call(full, "Alpha", "interested", "hot lead", approve=lambda t: seen.append(t) or False, path=str(tmp_path / "x.csv"))
    assert "Alpha" in seen[0] and "interested" in seen[0] and "hot lead" in seen[0]


def test_log_call_does_not_modify_the_table(full, tmp_path):
    before = full.copy()
    log_call(full, "Alpha", "spoke", approve=lambda t: True, path=str(tmp_path / "x.csv"))
    pd.testing.assert_frame_equal(full, before)
