r"""Tests for the call-list tools. Uses a tiny FAKE table, not your real data.

Run:  .venv\Scripts\python.exe -m pytest test_tools.py -v

Write next_to_call() and get_prospect() in tools.py until these pass.
Both take the table as their first argument (so tests can pass a fake one).
Both return a list of plain dicts (so the agent can turn them into JSON).
"""
import pandas as pd
import pytest

from tools import get_prospect, next_to_call


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
