"""
Benchmark AVRO package size as a function of NUM_BUF_SAMPLES.

For each value of NUM_BUF_SAMPLES, draws N_TRIALS random windows of length
2*NUM_BUF_SAMPLES+1 from the dataset, serialises each with AVRO.save_AVRO_default
(deflate codec, same as get_important_important_batch), and reports the average
resulting file size.
"""

import json
import os
import tempfile
import numpy as np
import pandas as pd
import bin.AVRO as AVRO
from bin.utils import generate_avro_schema

# ── Config ────────────────────────────────────────────────────────────────────
N_TRIALS = 50          # random windows per NUM_BUF_SAMPLES value
CODEC = 'deflate'
ACCURACY = 10

NUM_BUF_SAMPLES_VALUES = [10, 20, 40, 60, 80, 100, 120, 150, 200]

# ── Load dataset (mirrors IoT_device.__init__) ────────────────────────────────
with open("configs.json", "r") as f:
    configs = json.load(f)
cfg = configs["string_configs"]

df = pd.read_csv(cfg["file_paths"]["dataset_path"],
                 sep=cfg["data_columns"]["delimiter"])
df = df.drop(columns=cfg["data_columns"]["sensors_to_drop"])

sensor_cols = df.columns[df.isnull().any()].tolist()
df[sensor_cols] = df[sensor_cols].interpolate(method='linear')
df[sensor_cols] = df[sensor_cols].fillna(method='ffill')
df[sensor_cols] = df[sensor_cols].fillna(method='bfill')

timestamps_series = df[cfg["data_columns"]["timestamp_column"]]
data = df.drop(columns=[cfg["data_columns"]["timestamp_column"]])

n_sensors = data.shape[1]
n_rows    = data.shape[0]

# ── Generate schema once ──────────────────────────────────────────────────────
schema_path = os.path.join(cfg["file_paths"]["test_files_dir"],
                           f"avro_{n_sensors}.avsc")
os.makedirs(cfg["file_paths"]["test_files_dir"], exist_ok=True)
generate_avro_schema(n_sensors, schema_path)

# ── Benchmark ─────────────────────────────────────────────────────────────────
print(f"{'NUM_BUF_SAMPLES':>16}  {'window_len':>10}  {'avg_size_bytes':>14}  {'avg_size_KB':>10}")
print("-" * 58)

results = {}
with tempfile.TemporaryDirectory() as tmpdir:
    for nbs in NUM_BUF_SAMPLES_VALUES:
        window_len = 2 * nbs + 1
        if window_len > n_rows:
            print(f"{nbs:>16}  {window_len:>10}  (dataset too short)")
            continue

        sizes = []
        rng = np.random.default_rng(seed=42)
        starts = rng.integers(0, n_rows - window_len, size=N_TRIALS)

        for start in starts:
            idx = range(start, start + window_len)
            # mirror get_important_important_batch: buffer is converted to numpy
            # before being passed to save_AVRO_default, giving integer column names
            snippet = np.array(data.iloc[idx].values.tolist())
            ts      = timestamps_series.iloc[idx].tolist()

            avro_path = os.path.join(tmpdir, "tmp.avro")
            AVRO.save_AVRO_default(snippet, ts, schema_path,
                                   accuracy=ACCURACY,
                                   path=avro_path,
                                   original_size=1,
                                   codec=CODEC)
            sizes.append(os.path.getsize(avro_path))

        avg = np.mean(sizes)
        results[nbs] = avg
        print(f"{nbs:>16}  {window_len:>10}  {avg:>14.0f}  {avg/1024:>10.2f}")

print()
print("Done.")
