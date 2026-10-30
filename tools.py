"""Day 1 - tools for the call-list agent.

load_lists() reads the three prospect lists and returns ONE table with the same columns:
list, name, type, score, phone, address, hook, pitch, status, notes
"""
import csv
import os
from datetime import datetime

import pandas as pd

PRIVATE_DIR = "data_private"

# One mapping per file: original column name -> common name.
# (the hook/pitch columns differ: the consultant file has no hook, the spa file has no pitch)
FILES = {
    "sports": (
        "sports_prospects.csv",
        {
            "CLUB NAME": "name",
            "SPORT": "type",
            "PRIORITY SCORE (call highest first)": "score",
            "PHONE": "phone",
            "ADDRESS": "address",
            "YOUR CALL HOOK (lead with this)": "hook",
            "YOUR PITCH": "pitch",
            "STATUS": "status",
            "NOTES": "notes",
        },
    ),
    "consultants": (
        "consultant_prospects.csv",
        {
            "BUSINESS NAME": "name",
            "TYPE": "type",
            "PARTNER SCORE (contact highest first)": "score",
            "PHONE": "phone",
            "ADDRESS": "address",
            "YOUR PITCH": "pitch",
            "STATUS": "status",
            "NOTES": "notes",
        },
    ),
    "medspa": (
        "prospects.csv",
        {
            "BUSINESS NAME": "name",
            "PRIORITY (call highest first)": "score",
            "PHONE": "phone",
            "ADDRESS": "address",
            "YOUR CALL HOOK (lead with this)": "hook",
            "STATUS": "status",
            "NOTES": "notes",
        },
    ),
}

COMMON_COLUMNS = ["list", "name", "type", "score", "phone", "address", "hook", "pitch", "status", "notes"]

# Text that means "we don't actually know this". Becomes None.
PLACEHOLDERS = {"n/a", "check website", "no website", "not found", ""}


def clean_value(value):
    """Return None for blanks and placeholder text, otherwise the stripped string."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    lowered = text.lower()
    if lowered in PLACEHOLDERS or lowered.startswith("not found"):
        return None
    return text


def load_lists(folder=PRIVATE_DIR):
    frames = []
    for list_name, (filename, mapping) in FILES.items():
        raw = pd.read_csv(f"{folder}/{filename}")
        df = raw[list(mapping)].rename(columns=mapping)   # keep only mapped columns, rename them
        df["list"] = list_name
        for col in COMMON_COLUMNS:                          # columns this file doesn't have -> empty
            if col not in df.columns:
                df[col] = None
        frames.append(df[COMMON_COLUMNS])

    out = pd.concat(frames, ignore_index=True)
    for col in COMMON_COLUMNS:
        if col not in ("list", "score"):
            out[col] = out[col].map(clean_value)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# Tools the agent can call. Each takes the table (df) first, so tests can pass a fake one.
# Each returns plain dicts/lists so the agent can turn them into JSON.
# ---------------------------------------------------------------------------
def _to_dicts(df):
    """Rows -> list of dicts, with missing values as None (not NaN)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def next_to_call(df, list_name=None, n=5, type=None):
    """Highest-score prospects we haven't contacted yet and can actually phone.

    Skips rows with no phone number. Raises ValueError on an unknown list name,
    and the message lists the valid names, so a wrong guess is visible, not silent.
    """
    valid = sorted(df["list"].dropna().unique())
    if list_name is not None and list_name not in valid:
        raise ValueError(f"Unknown list {list_name!r}. Valid lists: {valid}")

    rows = df[(df["status"] == "Not contacted") & df["phone"].notna()]
    if list_name is not None:
        rows = rows[rows["list"] == list_name]
    if type is not None:
        rows = rows[rows["type"] == type]

    rows = rows.sort_values("score", ascending=False, kind="stable").head(max(n, 0))
    return _to_dicts(rows)


def get_prospect(df, name):
    """Every row whose name matches (ignoring case and spaces). Several rows = ambiguous:
    the caller must see all of them and must not silently pick one."""
    key = str(name).strip().lower()
    matches = df[df["name"].fillna("").str.strip().str.lower() == key]
    return _to_dicts(matches)


def _lookup_one(df, name):
    """(row, None) if exactly one match, else (None, an explanatory result dict)."""
    matches = get_prospect(df, name)
    if not matches:
        return None, {"status": "not_found", "message": f"No prospect named {name!r}."}
    if len(matches) > 1:
        brief = [{k: m.get(k) for k in ("name", "list", "phone", "address")} for m in matches]
        return None, {"status": "ambiguous",
                      "message": f"{len(matches)} prospects are named {name!r}. Ask which one (use phone or address).",
                      "matches": brief}
    return matches[0], None


# ---- draft_opening: the LLM tool. Makes text. Saves nothing. --------------
def build_opening_prompt(row):
    """The prompt for one prospect. A plain function so tests can inspect it without an LLM."""
    hook = row.get("hook") or "none on file"
    pitch = row.get("pitch") or "No pitch on file."
    no_pitch_rule = (
        "- There is no pitch on file, so do NOT mention any price, product, feature or result. "
        "Use only the call hook, then ask permission to explain how Gisell can help.\n"
        if not row.get("pitch") else ""
    )
    return (
        "You are helping Gisell prepare for a cold phone call. Write a 2-3 sentence spoken opening.\n\n"
        f"Business: {row['name']}\n"
        f"Type: {row.get('type') or 'not listed'}\n"
        f"Call hook (lead with this): {hook}\n"
        f"What Gisell offers (pitch): {pitch}\n\n"
        "Rules:\n"
        "- Use ONLY facts written above. Never invent prices, names, results, locations or years.\n"
        f"{no_pitch_rule}"
        "- Lead with the call hook if there is one.\n"
        "- Sound like a person talking, not an email. No bullet points.\n"
        "- End with one short question.\n"
    )


def _default_llm(prompt):
    from rag import CHAT_MODEL, chat          # imported here so tests never load the RAG app
    r = chat(model=CHAT_MODEL, temperature=0.3, messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content.strip()


def draft_opening(df, name, llm=None):
    """A short spoken opening for one prospect. `llm` is any function prompt -> text (fakeable in tests)."""
    row, problem = _lookup_one(df, name)
    if problem:
        return problem
    if not row.get("hook") and not row.get("pitch"):
        return {"status": "no_material", "name": row["name"],
                "message": "No call hook and no pitch on file, so there is nothing true to say. Not drafting."}
    llm = llm or _default_llm
    return {"status": "ok", "name": row["name"], "opening": llm(build_opening_prompt(row))}


# ---- log_call: CONSEQUENTIAL. Writes only after a human approves. ---------
CALL_LOG = "call_log.csv"
OUTCOMES = {"no_answer", "left_voicemail", "spoke", "interested", "not_interested",
            "call_back", "wrong_number"}


def approve_cli(text):
    """Ask the human in the terminal. Only an explicit 'y' counts as approval."""
    print(text)
    return input("Approve? Type y to log it, anything else to skip: ").strip().lower() == "y"


def log_call(df, name, outcome, note="", approve=None, path=CALL_LOG):
    """Append one row to call_log.csv - never edits the prospect lists.

    Checks come first (valid outcome, exactly one matching prospect). The human is asked only
    after that. No `approve` function means no approval means nothing is written.
    """
    if outcome not in OUTCOMES:
        return {"status": "invalid_outcome", "message": f"Outcome must be one of {sorted(OUTCOMES)}."}
    row, problem = _lookup_one(df, name)
    if problem:
        return problem

    summary = f"Log call: {row['name']} ({row['list']}, {row['phone']}) -> {outcome}. Note: {note or '(none)'}"
    if approve is None or not approve(summary):
        return {"status": "declined", "message": "Not logged: no approval.", "summary": summary}

    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "list", "name", "phone", "outcome", "note"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), row["list"], row["name"],
                         row["phone"], outcome, note])
    return {"status": "logged", "summary": summary}
