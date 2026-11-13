
today = pd.Timestamp.today()
overdue = df[df['follow_up_date'] < today]
print(overdue[['name', 'company', 'follow_up_date', 'status']])

today = pd.Timestamp.today()
overdue = df[df['follow_up_date'] < today]
print(overdue[['name', 'company', 'follow_up_date', 'status']])
