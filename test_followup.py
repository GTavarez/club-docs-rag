import pandas as pd
from followup import find_overdue


def make_df(dates):
    return pd.DataFrame({
        "Club/Practice": [f"Club {i}" for i in range(len(dates))],
        "Follow Up Date": dates,
        "Status": ["Sent"] * len(dates),
    })


def test_overdue_row_is_found():
    assert len(find_overdue(make_df(["Aug 26"]), days=7)) == 1


def test_unparseable_date_is_excluded():
    assert len(find_overdue(make_df(["Keep warm"]), days=7)) == 0


def test_future_date_is_excluded():
    assert len(find_overdue(make_df(["Dec 31"]), days=7)) == 0