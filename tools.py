"""Day 1 - tools for the call-list agent.

load_lists() reads the three prospect lists and returns ONE table with the same columns:
list, name, type, score, phone, hook, pitch, status, notes
"""
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
            "YOUR CALL HOOK (lead with this)": "hook",
            "STATUS": "status",
            "NOTES": "notes",
        },
    ),
}

COMMON_COLUMNS = ["list", "name", "type", "score", "phone", "hook", "pitch", "status", "notes"]

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
