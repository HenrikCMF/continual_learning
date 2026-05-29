import csv
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def parse_ts(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        m = re.search(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?', str(s))
        if m:
            return pd.to_datetime(m.group())
        return None


interval_minutes = 10000

with open('episodes.csv', 'r') as f:
    rows = [(int(row[0]), parse_ts(row[1])) for row in csv.reader(f) if len(row) >= 2]

first_one = next((i for i, (e, _) in enumerate(rows) if e == 1), None)
if first_one is not None:
    rows = rows[first_one + 1:]

df = pd.DataFrame(rows, columns=['label', 'timestamp'])
df['bucket'] = df['timestamp'].dt.floor(f'{interval_minutes}min')
grouped = df.groupby('bucket')['label'].apply(lambda g: (g == 0).sum()).reset_index()
grouped.columns = ['time', 'fp_count']

fault_times = df.loc[df['label'] == 1, 'timestamp']

plt.figure()
plt.plot(grouped['time'], grouped['fp_count'], marker='o')
for ts in fault_times:
    plt.axvline(x=ts, color='red', linewidth=0.8, zorder=5)
plt.xlabel('Time')
plt.ylabel(f'False positives per {interval_minutes}-min interval')
plt.title('False positives over time')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig('episodes_plot.png')
plt.show()
