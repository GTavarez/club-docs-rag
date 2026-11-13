import argparse
import sys
import pandas as pd
from google import genai


BUSINESS = "GT Digital builds custom all-in-one platforms for youth sports clubs and community organizations, with online registration, payments, roster and team management, and a parent portal, all built and hosted for you. You want it because it takes the endless spreadsheets, group texts, and payment chasing off your volunteers' plates and puts everything in one place that just works, so your board spends time running the club instead of running admin."


def find_overdue(df, days):
    df = df.copy()
    df["followup_parsed"] = pd.to_datetime(
        df["Follow Up Date"] + " 2026", format="%b %d %Y", errors="coerce"
    )
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
    return df[df["followup_parsed"] < cutoff]


def draft_message(client, row):
    prompt = f"""Business context: {BUSINESS}

Club: {row['Club/Practice']}
Sport: {row['Sport/Specialty']}
Contact: {row['Contact']}
Status of last outreach: {row['Status']}
Last contacted: {row['Date']} 2026. Today is {pd.Timestamp.today():%b %d} 2026.
Channel used last time: {row['Status']}
Notes from last contact: {row['Notes']}

Write a follow-up message of two or three sentences.

Rules:
- Reference the specific last interaction.
- Give one concrete reason this club in particular would care, drawn from the business context.
- End with one specific ask — a question or a proposed next step.
- Never write a message whose only content is "checking in."
- Do not invent facts not listed above.
- Never state or imply how long ago the last contact was unless you can compute it from the dates given.
- Match the medium: an Instagram DM is short and has no formal sign-off. An email can be longer and signs off "Best, Gisell". A voicemail follow-up should reference the voicemail.
- The reason this club would care must be specific to their sport or their situation, not generic.
- Sign off as Gisell."""
    result = client.interactions.create(model="gemini-3.7-flash", input=prompt)
    return result.output_text



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--file", default="followup.csv")
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"Could not find '{args.file}'. Check the filename and that you're in the right folder.")
        sys.exit(1)

    overdue = find_overdue(df, args.days)
    print(overdue[["Club/Practice", "Follow Up Date", "Status"]])

    if args.draft:
        client = genai.Client()
        for _, row in overdue.head(args.limit).iterrows():
            print("=" * 60)
            print(row["Club/Practice"])
            print(draft_message(client, row))


if __name__ == "__main__":
    main()