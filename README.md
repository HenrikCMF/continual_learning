# ACORD — Link-Aware Energy-Frugal Continual Learning for Fault Detection in IoT Networks

Reference implementation and experimental testbed for:

> H. C. M. Frederiksen, J. Shiraishi, Č. Stefanović, H. V. Cheng, and S. R. Pandey,
> **"Link-Aware Energy-Frugal Continual Learning for Fault Detection in IoT Networks."**

**ACORD** (Adaptive Compression Online Resource-aware fault Detection) is an
event-driven communication framework that integrates **continual learning (CL)**
into an IoT fault-detection (FD) loop. A battery-powered **IoT device** runs a
lightweight autoencoder FD model on-device; when it observes a potential fault it
ships a small window of context data to an **edge server (ES)**, which relabels the
data, updates the model with experience-replay CL, and compresses the model
(pruning + quantization) to fit the *measured link conditions and remaining energy
budget* before sending it back. The aim is to maximize inference recall under a
hard energy constraint `E_th` (Eq. 4 in the paper).

The two nodes run on **separate machines** connected over TCP; the wireless link is
emulated with Linux `tc` so the energy/bandwidth sweeps are reproducible on a LAN.

```
   ┌────────────────────────┐   Uplink: Avro-compressed context window Q^i (2W+1 samples)
   │  IoT device            │  ───────────────────────────────────────▶  ┌───────────────────────┐
   │  IoT_device.py         │                                            │  Edge Server (ES)     │
   │                        │   Downlink: compressed FD model M^{i+1}    │  ES.py                │
   │  FD phase: monitor with│  ◀───────────────────────────────────────  │                       │
   │  model M^i, threshold  │       tc-emulated Wi-Fi bottleneck         │  relabel + CL update  │
   │  τ_th; on fault buffer │     (effective rate R_UL / R_DL)           │  prune P_L, quant Q_L │
   │  context window W      │                                            │                       │
   └────────────────────────┘                                            └───────────────────────┘
```

---

## 1. The FD-round loop

Time is divided into **fault-detection rounds**. In round *i* the device uses FD
model `M^i`; each round has the four phases of the paper (Sec. II-A), which map
directly onto the two programs:

| Paper phase | Where it runs | What happens |
| --- | --- | --- |
| **1. FD phase** | `IoT_device.py` → `get_important_important_batch()` / `analyze_samples()` | Stream samples through the autoencoder; flag a sample as a potential fault when its reconstruction error exceeds the decision threshold **τ_th**. |
| **2. Uplink transmission** | `IoT_device.py` (Avro encode + `send_file`) | On a flagged sample, accumulate the **context window** of `2W+1` samples (`Q^i`), losslessly compress with Avro, and transmit to the ES. |
| **3. Training + link-aware compression** | `ES.py` → `IoT_model.improve_model()` | Relabel the batch (fault / normal), update `M^i` with experience-replay CL, then prune (**P_L**) and quantize (**Q_L**) the model to fit the target downlink time. |
| **4. Downlink model transmission** | `ES.py` → `distribute_model()` | Zip the compressed `M^{i+1}` and send it back; the device loads it and resumes the FD phase. |

The device tracks cumulative energy `E_total`; once it reaches the budget `E_th` it
stops updating the model (matching the energy constraint in Eq. 4a).

### Node entry points
- **`ES.py`** (`ES_station`) — edge server: trains the initial model, waits for the
  device, then per round receives data, runs the CL update + compression, and
  redistributes the model. `run()` returns `(TP, FP, avg_throughput)`.
- **`IoT_device.py`** (`iot_device`) — IoT device: streams the dataset through the FD
  model, buffers context windows on detection, transmits them, and receives model
  updates. `run()` returns timing, byte counts, inference count, throughput, and
  total energy.

> **Start order:** launch `ES.py` **first** (it trains and waits for a client), then
> `IoT_device.py` on the other machine — it performs a `READY` handshake before
> streaming.

---

## 2. Requirements & setup

- Python 3.10 (a conda env named `py310` is used in development).
- `pip install -r requirements.txt`
- **Linux + `tc` + sudo.** Link emulation calls `sudo tc ...` (`bin/network_control.py`);
  the sudo password is read from the config (`sudo_password`). Set
  `use_config_network_control: false` to disable emulation.
- TensorFlow Lite is used for model deployment (Sec. IV-A); models live in `models/`.
- Datasets are fetched separately — see [§4](#4-datasets).

---

## 3. The two nodes in detail

### Edge server — `ES.py`
1. On a fresh start, builds the initial training set from normal-only data
   (`make_initial_data`) and trains the initial autoencoder (`train_initial_model`) —
   matching the paper's assumption that the round-0 model sees only normal data.
2. Applies the emulated link conditions via `tc`, waits for the device, distributes `M^0`.
3. **Per round** (`run`): receives an Avro package, splits it into batches, and for each:
   - relabels it using the dataset's fault label (the "fault reporter" of Fig. 1);
   - calls `improve_model(...)` — experience-replay CL update, then pruning/quantization
     scaled to the measured throughput and the download time budget `t_DL`;
   - appends the samples to `initial_data.csv` (normal) or `faulty_data.csv` (fault),
     forming the rehearsal memory `R^i`;
   - re-distributes the compressed `M^{i+1}`.
4. On `DONE`, prints summary stats and writes `episodes.csv`.

### IoT device — `IoT_device.py`
1. Builds its sample stream from the dataset and an Avro schema sized to the sensor count.
2. Applies the emulated link conditions via `tc`.
3. **Per round** (`run`): receives `M^{i+1}` (loads it when CL is enabled), then
   `get_important_important_batch()`:
   - scores each sample by autoencoder reconstruction MSE against **τ_th**
     (`select_threshold` when `vary_th` is on, else fixed);
   - on a flagged sample, collects `W` past + `W` future samples (the `2W+1` window,
     `NUM_BUF_SAMPLES` = `W`), Avro-encodes it, and transmits;
   - accumulates inference / Tx / Rx energy; stops updating once `E_total ≥ E_th`.
4. When the dataset is exhausted, sends `DONE` and writes `test_files/full_IoTresult.csv`.

---

## 4. Datasets

The paper evaluates on the **Kaggle pump-sensor dataset** (Sec. IV-A; ref. [10]):
~220k samples, 50 features, labels `{NORMAL, BROKEN, RECOVERING}` where `BROKEN` is
the fault state. This is the **default** configuration (`configs.json`). A second
dataset, **HAI**, is also supported via `configs_HAI.json`.

The CSVs are hosted as assets on the
[`datasets-v1` release](https://github.com/HenrikCMF/ACORD/releases/tag/datasets-v1)
and downloaded into `datasets/` with:

```bash
python bin/download_data.py          # fetches HAI.csv and sensor.csv
python bin/download_data.py --force  # re-download even if present
```

| File | Config | Source |
| --- | --- | --- |
| `datasets/sensor.csv` | `configs.json` (default) | Kaggle pump-sensor dataset — **the paper's dataset** |
| `datasets/HAI.csv` | `configs_HAI.json` | HAI ICS security dataset (additional) |

`datasets/` is git-ignored, so downloaded files are never committed by accident.

---

## 5. Configuration — `configs.json`

All runtime behavior is driven by `configs.json` (read by both nodes and
`bin/utils.py`). The default `configs.json` **is** the paper's pump-sensor
configuration, so no changes are needed to reproduce the paper. To switch to the
HAI dataset instead, copy that profile over it:

```bash
cp configs_HAI.json configs.json
```
(The code always reads the file literally named `configs.json`.)

### Network & scenario keys ↔ paper notation

| Key | Paper symbol | Meaning |
| --- | --- | --- |
| `baseline_energy` | `E_Ref` | Reference energy (paper: **60 J** for `R_Ref = 1 Mbps`) — default `configs.json` uses `60`. |
| `baseline_rate` | `R_Ref` | Reference link rate (paper **1 Mbps**; config in kbps → `1000`). |
| `base_tdl` | `τ_DL` | Reference downlink transmission time, used to derive the target `t*_DL` (Eq. 5). |
| `iot_device_tul` | `τ_UL` | Reference uplink transmission time, used to derive `t*_UL` (Eq. 5). |
| `bandwidth_limit_kbps`, `buffering_latency_ms`, `packet_loss_pct`, … | — | `tc` shaping of the emulated link (`R_UL` / `R_DL`). |
| `iot_device_ip`, `ESip`, ports, `*NET_INTERFACE` | — | LAN addresses / shaped interface per machine. |
| `use_config_network_control` | — | Apply `tc` shaping on startup. |
| `sudo_password` | — | Used for `sudo tc`. |

> The constructors take `bandwidth=` and `energy_budget=`(**`E_th`**) arguments. The
> code computes `energy_ratio = E_th / E_Ref` and scales the target transmission
> times `t_DL = energy_ratio · τ_DL`, `t_UL = energy_ratio · τ_UL` (Eq. 5), which in
> turn drive the context-window size `W` and the model compression `{P_L, Q_L}`.

### Dataset keys — `string_configs.data_columns` (change per dataset)

| Key | Meaning | pump (paper) | HAI |
| --- | --- | --- | --- |
| `dataset_label` | label column | `machine_status` | `attack` |
| `fault_label` | value marking a fault | `"BROKEN"` | `1` |
| `timestamp_column` | timestamp column | `timestamp` | `time` |
| `sensors_to_drop` | columns dropped before training | `["Unnamed: 0", …]` | `["attack_P1", …]` |
| `delimiter` | CSV delimiter | `,` | `;` |

---

## 6. Reproducing the paper's schemes

The `string_configs.ablation_settings` flags select ACORD vs. the three baselines
of Sec. IV-A. **ACORD** uses full link-aware adaptation + CL; **Hawk** uses fixed
`P_L = 0, Q_L = 32, W = 200` with no link/energy awareness; **Hawk+DeepIoT** adds
DeepIoT-style compression; **Hawk w/o CL** never updates the model.

| Flag | ACORD | Hawk | Hawk+DeepIoT | Hawk w/o CL |
| --- | :--: | :--: | :--: | :--: |
| `CL_enabled` | ✅ | ✅ | ✅ | ❌ |
| `use_DeepIoT` | ❌ | ❌ | ✅ | ❌ |
| `Link_adaptation_parts.IoT_side_adaptation_enabled` (adapts **W**) | ✅ | ❌ | ❌ | ❌ |
| `Link_adaptation_parts.Pruning_enabled` (**P_L**) | ✅ | ❌ | ❌ | ❌ |
| `Link_adaptation_parts.Quantization_enabled` (**Q_L**) | ✅ | ❌ | ❌ | ❌ |
| `Link_adaptation_parts.vary_th` (adapts **τ_th**) | ✅ | ❌ | ❌ | ❌ |

With link adaptation off, the device falls back to `W = NUM_BUF_SAMPLES = 200` and a
fixed `τ_th`, exactly the Hawk configuration.

### Running a sweep
`run_ES.py` / `run_IoT.py` reproduce the two curves of Fig. 2 by sweeping either the
**energy threshold `E_th`** (Fig. 2 left, fixed bandwidth) or the **bandwidth**
(Fig. 2 right, fixed `E_th = 60 J`). Run them as a matched pair (ES first):

```bash
# server machine
python run_ES.py
# device machine
python run_IoT.py
```

Edit the top of each file to configure the sweep:

| Variable | Meaning |
| --- | --- |
| `screen` | `"energy"` (sweep `E_th`) or `"bandwidth"` (sweep link rate) |
| `bandwidths` / `energy_budget` | swept values, and the fixed value of the other axis |
| `results_file` | output CSV |

`run_ES.py` logs `[value, TP, FP, avg_throughput]` per run; `run_IoT.py` logs energy,
timing, byte counts, inferences, and average throughput — together these give the
**recall** (Eq. 3) plotted in Fig. 2.

---

## 7. What to change, and where

| Paper component | File / location |
| --- | --- |
| Autoencoder layers `{128,64,32,8,32,64,128}` (Sec. IV-A) | `bin/IoT_model.py` → `design_model_architecture()` |
| Masked AE loss (Eq. 10, faults excluded from reconstruction) | `bin/IoT_model.py` → `train_model()` (`mse_loss`, `invert_loss`) |
| Training epochs `L_epoch = 2000/(2W+1)` | `bin/IoT_model.py` → `train_model()` (`num_epochs`) |
| Experience-replay CL (rehearsal memory `R^i`) | `bin/IoT_model.py` → `combine_new_with_random_old`, `combine_faulty_with_random_old` |
| Pruning `P_L` & quantization `Q_L` vs. throughput (Eq. 6–7) | `bin/IoT_model.py` → `improve_model()` |
| TFLite quantization levels `Q_L ∈ {8,32}` | `bin/_quantize_model.py` |
| Context-window `W` adaptation (Eq. 8) | `IoT_device.py` → `get_important_important_batch()` (`NUM_BUF_SAMPLES`) |
| Decision-threshold `τ_th` adaptation (Eq. 9) | `IoT_device.py` → `select_threshold()` / fixed `trigger_threshold` |
| Energy model: `ξ_rx=0.33`, `ξ_tx=0.79`, `N_s=34298`, `N_c=33792`, `A_s=506`, `E_inf` (Sec. II-B, IV-A) | `bin/IoT_energy.py` |
| Reference values `E_Ref`, `R_Ref`, `τ_DL`, `τ_UL` and budget `E_th` | `configs.json` + `__main__` of `ES.py`/`IoT_device.py` |
| Link emulation (`R_UL`/`R_DL`, latency, loss) | `bin/network_control.py` (`tc` TBF/netem) |
| TCP transport, throughput & RTT measurement | `bin/TCP_code.py` |
| Lossless data compression of `Q^i` (Avro) | `bin/AVRO.py`, `generate_avro_schema()` in `bin/utils.py` |

---

## 8. Outputs

| File | Produced by | Contents |
| --- | --- | --- |
| `episodes.csv` | `ES.py` | per-package `(is_fault, mid_timestamp)` — used to compute recall (Eq. 3) |
| `test_files/full_IoTresult.csv` | `IoT_device.py` | per-sample `mse`, `energy`, configured & measured `throughput` |
| `<results_file>.csv` | `run_ES.py` / `run_IoT.py` | one row per swept `E_th` / bandwidth point (Fig. 2 data) |
| `models/autoencoder.{h5,tflite,tflite.zip}`, `models/scaler.pkl` | `bin/IoT_model.py` | FD model artifacts and fitted scaler |
| Console summary | both nodes | elapsed time, bytes sent, TP/FP, total energy |

---

## 9. Repository layout

```
ES.py                 # Edge server (run on the server machine)
IoT_device.py         # IoT device (run on the device machine)
run_ES.py / run_IoT.py# Sweep drivers (E_th / bandwidth → recall)
configs.json          # Active configuration — the paper's Kaggle pump-sensor setup
configs_HAI.json      # Alternate configuration (HAI ICS dataset)
requirements.txt
datasets/             # Source CSVs — fetched via bin/download_data.py (git-ignored)
models/               # Trained FD model + scaler artifacts
test_files/           # Working dir: staged data, Avro schemas, results
received/             # Working dir: incoming files
bin/
  TCP_code.py         # TCP/UDP transport, throughput & RTT measurement
  network_control.py  # tc-based link emulation (R_UL / R_DL, latency, loss)
  AVRO.py             # Lossless Avro (de)serialization of context windows Q^i
  IoT_model.py        # Autoencoder FD model: CL update, pruning P_L, quantization Q_L
  DeepIoT_model.py    # DeepIoT-style model (Hawk+DeepIoT baseline, use_DeepIoT=true)
  _quantize_model.py  # TFLite conversion / quantization (Q_L)
  IoT_energy.py       # Energy model (ξ_tx, ξ_rx, N_s, N_c, A_s, E_inf)
  download_data.py    # Fetch dataset CSVs from the GitHub release
  utils.py            # Config loader, dataset prep, Avro schema gen, decorators
```
