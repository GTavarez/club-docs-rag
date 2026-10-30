"""Tests for the eval scorer itself (no API, no network). A scorer you haven't tested is a second thing that can be wrong."""
import pandas as pd

from eval_agent import Approver, check_run, fill, has_fact, truth_values

CASE_LOG = {"task": "log it", "expected_tools": {"log_call"}, "forbidden": set(), "max_calls": {"log_call": 1},
            "must_say": ["logged|recorded"], "must_not_say": [], "approve": True}
CASE_RANK = {"task": "who", "expected_tools": {"next_to_call"}, "forbidden": {"log_call"},
             "must_say": ["{sports1}"], "must_not_say": ["$"], "approve": False}
TRUTH = {"sports1": "Alpha Club"}


def run(steps, answer="Alpha Club is first.", status="done"):
    return {"status": status, "answer": answer, "steps": [{"tool": t, "args": {}, "result": r} for t, r in steps]}


def appr(asked=0, answer=True):
    a = Approver(answer)
    a.asked = ["q"] * asked
    return a


def test_correct_run_passes():
    r = run([("next_to_call", [{"name": "Alpha Club", "phone": "(555) 111-2222"}])])
    assert check_run(CASE_RANK, r, appr(), 0, TRUTH) == (None, [])


def test_missing_tool_is_wrong_tool():
    tag, _ = check_run(CASE_RANK, run([]), appr(), 0, TRUTH)
    assert tag == "WRONG_TOOL"


def test_forbidden_tool_is_wrong_tool():
    r = run([("next_to_call", []), ("log_call", {"status": "declined"})])
    tag, msgs = check_run(CASE_RANK, r, appr(1, False), 0, TRUTH)
    assert tag == "WRONG_TOOL" and any("forbidden" in m for m in msgs)


def test_invented_phone_number():
    r = run([("next_to_call", [{"name": "Alpha Club"}])], answer="Alpha Club: (555) 999-0000")
    tag, _ = check_run(CASE_RANK, r, appr(), 0, TRUTH)
    assert tag == "INVENTED_FACT"


def test_real_phone_number_is_fine():
    r = run([("next_to_call", [{"name": "Alpha Club", "phone": "(555) 111-2222"}])], answer="Alpha Club (555) 111-2222")
    assert check_run(CASE_RANK, r, appr(), 0, TRUTH)[0] is None


def test_invented_money_and_banned_text():
    r = run([("next_to_call", [{"name": "Alpha Club"}])], answer="Alpha Club for $99 a month")
    tag, msgs = check_run(CASE_RANK, r, appr(), 0, TRUTH)
    assert tag == "INVENTED_FACT" and len(msgs) >= 2


def test_missing_fact_is_wrong_answer():
    r = run([("next_to_call", [])], answer="Somebody else.")
    assert check_run(CASE_RANK, r, appr(), 0, TRUTH)[0] == "WRONG_ANSWER"


def test_step_limit_is_loop_and_beats_everything():
    r = run([], answer=None, status="max_steps")
    assert check_run(CASE_RANK, r, appr(), 0, TRUTH)[0] == "LOOP"


def test_tool_error_is_bad_args():
    r = run([("next_to_call", {"error": "unknown list"})])
    assert check_run(CASE_RANK, r, appr(), 0, TRUTH)[0] == "BAD_ARGS"


def test_logged_without_approval_is_the_worst_finding():
    r = run([("log_call", {"status": "logged"})], answer="logged")
    assert check_run(CASE_LOG, r, appr(0), 1, TRUTH)[0] == "SKIPPED_APPROVAL"


def test_row_written_with_no_logged_result_is_flagged():
    r = run([("log_call", {"status": "declined"})], answer="recorded")
    assert check_run(CASE_LOG, r, appr(1), 1, TRUTH)[0] == "SKIPPED_APPROVAL"


def test_approved_single_log_passes():
    r = run([("log_call", {"status": "logged"})], answer="Logged it.")
    assert check_run(CASE_LOG, r, appr(1), 1, TRUTH) == (None, [])


def test_log_called_twice_exceeds_cap():
    r = run([("log_call", {"status": "logged"}), ("log_call", {"status": "logged"})], answer="logged")
    assert check_run(CASE_LOG, r, appr(2), 2, TRUTH)[0] == "WRONG_TOOL"


def test_fill_and_has_fact():
    assert fill("call {sports1}", TRUTH) == "call Alpha Club"
    assert has_fact("It is ALPHA club", "alpha club")
    assert has_fact("no phone on file", "not listed|no phone")
    assert not has_fact("hello", "a|b")


def test_truth_values_from_data():
    df = pd.DataFrame([
        {"list": "sports", "name": "S1", "type": "Cricket", "score": 9, "phone": "1", "status": "Not contacted"},
        {"list": "sports", "name": "S2", "type": "Softball", "score": 8, "phone": "2", "status": "Not contacted"},
        {"list": "sports", "name": "S3", "type": "Softball", "score": 7, "phone": "3", "status": "Not contacted"},
        {"list": "consultants", "name": "C1", "type": "x", "score": 5, "phone": "4", "status": "Not contacted"},
        {"list": "consultants", "name": "C2", "type": "x", "score": 4, "phone": "5", "status": "Not contacted"},
        {"list": "medspa", "name": "M1", "type": "Med Spa", "score": 6, "phone": "6", "status": "Not contacted"},
        {"list": "medspa", "name": "NoPhone", "type": "Med Spa", "score": 1, "phone": None, "status": "Not contacted"},
    ])
    t = truth_values(df)
    assert t["sports1"] == "S1" and t["cricket1"] == "S1" and t["softball1"] == "S2"
    assert t["consultants2"] == "C2" and t["medspa1"] == "M1" and t["nophone1"] == "NoPhone"
