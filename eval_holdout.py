"""Hold-out eval set: questions written AFTER the prompt was tuned, so they are an unseen test.

Rules for yourself:
  1. Write the question, the must_say facts and the source file BEFORE you see what the app answers.
  2. Do not edit the prompt or drill_json.py based on these results until you have logged the score.
  3. Score once:  .venv\\Scripts\\python.exe score_eval.py --set holdout --label holdout-v1 --k 3

Format is the same as eval_cases.py:
  must_say: short facts the answer MUST contain, "a|b" for alternatives, [] for refusals.
  sources:  set of files that must be cited, or None if the app must refuse.

Target mix (10): 4 plain, 3 multi-part (only some parts answerable), 2 must-refuse, 1 adversarial.
Check the set itself (no API needed):  .venv\\Scripts\\python.exe eval_holdout.py
"""
from eval_cases import check_set

EVAL = [
    # {
    #     "q": "Your question here?",
    #     "must_say": ["fact one", "a|b"],
    #     "sources": {"faq_01_joining.txt"},
    # },
    # {
    #     "q": "A question the documents do not cover?",
    #     "must_say": [],
    #     "sources": None,
    # },
]

if __name__ == "__main__":
    check_set(cases=EVAL)
