import pandas as pd
df = pd.read_csv("followup.csv")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(df.columns.tolist())
print(df.dtypes)
print(df.head())
print(df.shape)

# 1. glue the year onto the text, then parse it into real dates
df["followup_parsed"] = pd.to_datetime(df["Follow Up Date"] + " 2026", format="%b %d %Y", errors="coerce")

# 2. what is today?
today = pd.Timestamp.today()

# 3. keep only the rows whose parsed date is before today
overdue = df[df["followup_parsed"] < today]

print(overdue[["Club/Practice", "Follow Up Date", "Status"]])