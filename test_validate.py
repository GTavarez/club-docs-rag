r"""Tests for validate() in drill_json.py. Harness is done; you write validate().

Run:  .venv\Scripts\python.exe -m pytest test_validate.py -v
"""
from drill_json import REFUSAL, validate

RETRIEVED = {"faq_05_refunds.txt", "handbook_02_subs.txt", "faq_01_joining.txt"}


def good(**changes):
    out = {"found": True, "answer": "Yes, pro-rata.", "sources": ["faq_05_refunds.txt"]}
    out.update(changes)
    return out


# --- 1. a good answer comes back unchanged -------------------------------
def test_good_answer_passes_through():
    assert validate(good(), RETRIEVED) == good()


# --- 2. wrong shape -> None (caller will retry) --------------------------
def test_not_a_dict():
    assert validate(["found", True], RETRIEVED) is None


def test_missing_key():
    out = good()
    del out["sources"]
    assert validate(out, RETRIEVED) is None


def test_extra_key():
    assert validate(good(confidence=0.9), RETRIEVED) is None


def test_found_is_a_string_not_bool():
    assert validate(good(found="true"), RETRIEVED) is None


def test_sources_is_a_string_not_list():
    assert validate(good(sources="faq_05_refunds.txt"), RETRIEVED) is None


# --- 3. invented sources get dropped -------------------------------------
def test_invented_source_is_removed():
    out = good(sources=["faq_05_refunds.txt", "secret.txt"])
    assert validate(out, RETRIEVED)["sources"] == ["faq_05_refunds.txt"]


def test_answer_with_only_invented_sources_is_rejected():
    # claims an answer but cites nothing real -> can't trust it
    assert validate(good(sources=["secret.txt"]), RETRIEVED) is None


# --- 4. refusals are normalized ------------------------------------------
def test_refusal_is_normalized():
    out = {"found": False, "answer": "Sorry, not sure!", "sources": ["faq_05_refunds.txt"]}
    assert validate(out, RETRIEVED) == REFUSAL
