"""
Converts datasets/Data.csv into a format compatible with the existing pipeline.

Transformations:
- Renames 'time' -> 'timestamp'
- Renames 'label' -> 'machine_status'
- Maps 0 -> 'NORMAL', any other value -> 'BROKEN'
- Drops metadata columns: faultNumber, simulationRun, sample
- Writes result to datasets/Data_processed.csv

After running this script, swap configs_Data.json in as configs.json to use this dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
INPUT = "datasets/HAI.csv"
OUTPUT = "datasets/Hai_processed.csv"
DROP_COLS = ["faultNumber", "simulationRun", "sample"]

print(f"Loading {INPUT}...")
df = pd.read_csv(INPUT)
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

df = df.drop(columns=DROP_COLS)
df = df.rename(columns={"time": "timestamp", "label": "machine_status"})

print(df.head())
print(df["machine_status"].unique())
plt.plot(df["machine_status"])
plt.show()
exit()

df["machine_status"] = df["machine_status"].apply(lambda x: "NORMAL" if x == 0 else "BROKEN")

print(f"Label distribution:\n{df['machine_status'].value_counts()}")
print(f"Writing to {OUTPUT}...")

df.to_csv(OUTPUT, index=False)
print("Done.")
