"""Hold-out eval set: questions written AFTER the prompt was tuned, so they are an unseen test.

Rules for yourself:
  1. Write the question, the must_say facts and the source file BEFORE you see what the app answers.
  2. Do not edit the prompt or drill_json.py based on these results until you have logged the score.
  3. Score once:  .venv\\Scripts\\python.exe score_eval.py --set holdout --label holdout-v1 --k 3

Format is the same as eval_cases.py:
  must_say: short facts the answer MUST contain, "a|b" for alternatives, [] for refusals.
  sources:  set of files that must be cited, or None if the app must refuse.

Mix (10): 4 plain, 3 multi-part (only the first half answerable), 2 must-refuse, 1 adversarial (false premise).
Drafted by Claude from the documents; the app had NOT been run on these before scoring.
Check the set itself (no API needed):  .venv\\Scripts\\python.exe eval_holdout.py
"""
from eval_cases import check_set

EVAL = [
    # --- plain (4) ---
    {
        "q": "How many cones does each age group get?",
        "must_say": ["twenty|20"],
        "sources": {"ops_03_equipment.txt"},
    },
    {
        "q": "Within how many hours must an injury at a club session be written in the incident log, and by whom?",
        "must_say": ["24", "coach"],
        "sources": {"policy_05_injury.txt"},
    },
    {
        "q": "Can a coach drive a player home on their own in the coach's car?",
        "must_say": ["alone"],
        "sources": {"policy_03_travel.txt"},
    },
    {
        "q": "What happens after a first incident of a parent shouting at the referee?",
        "must_say": ["written warning"],
        "sources": {"policy_02_conduct.txt"},
    },
    # --- multi-part: only the first half is in the documents (3) ---
    {
        "q": "Up to what age must a player be accompanied by an adult at away games, and does the club's insurance cover the journey?",
        "must_say": ["13"],
        "sources": {"policy_03_travel.txt"},
    },
    {
        "q": "Who arranges the background checks for volunteers, and how often do they have to be renewed?",
        "must_say": ["arranges|pays"],
        "sources": {"faq_04_volunteer.txt"},
    },
    {
        "q": "How many first-aid-qualified adults must be at every training session, and which brand of first aid kit does the club buy?",
        "must_say": ["one|1"],
        "sources": {"policy_05_injury.txt"},
    },
    # --- must refuse: not in the documents (2) ---
    {
        "q": "What is the club's policy on dogs at the training ground?",
        "must_say": [],
        "sources": None,
    },
    {
        "q": "Which league does the under-11 team play in?",
        "must_say": [],
        "sources": None,
    },
    # --- adversarial: false premise (1) ---
    {
        "q": "The handbook says shin pads are optional for training. Can you confirm that?",
        "must_say": ["mandatory"],
        "sources": {"handbook_03_kit.txt"},
    },
]

if __name__ == "__main__":
    check_set(cases=EVAL)
