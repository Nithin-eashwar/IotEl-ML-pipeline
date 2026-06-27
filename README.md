# NILM Motor Load Classifier

Non-Intrusive Load Monitoring (NILM) pipeline for predictive maintenance of NEMA 17 stepper motors. The ML pipeline classifies motor operating states — **unloaded**, **light load**, **heavy load**, **stall** — from current, power, and vibration sensor data using windowed feature extraction and Random Forest classification. The trained model is exported to C and deployed on an ESP32 running bare-metal FreeRTOS firmware that publishes predictions to Azure IoT Hub over MQTT/TLS.

---

## Project Structure

```
├── ml/
│   ├── nilm_pipeline.py          # Main training pipeline (run this first)
│   ├── compare_nilm_features.py  # Validates C feature extraction vs Python
│   └── model.pkl                 # Saved model (generated, not committed)
│
├── data/                         # CSV recordings (gitignored — get from team drive)
├── results/                      # Generated plots & reports (gitignored)
│
├── firmware/motor_telemetry/     # Production ESP32 firmware (ESP-IDF v6)
│   ├── main/
│   │   ├── motor_telemetry.c     # FreeRTOS application
│   │   ├── nilm_model.c/.h       # Exported Random Forest (generated)
│   │   └── CMakeLists.txt        # Component config + env var injection
│   ├── .env                      # Credentials (gitignored)
│   ├── build.sh                  # Docker build wrapper
│   ├── idf_component.yml         # ESP-IDF component dependencies
│   └── sdkconfig                 # Project config (committed)
│
├── firmware/test_ml/             # Standalone ML test firmware
├── firmware/nilm_validate_features.c  # Feature validation helper
├── firmware/iot_testbed/         # Arduino data-collection sketch (Adafruit libs)
│
├── docs/                         # Design notes
├── test_azure_connection.py      # Azure IoT Hub diagnostic
├── requirements.txt
└── README.md
```

---

## ML Pipeline

The ML pipeline is the core of this project. It trains a classifier from raw sensor data, evaluates it rigorously, and exports the model for embedded deployment.

### Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ml/nilm_pipeline.py
```

Place the 4 training CSV files in `data/` before running.

### Sensor Data Format

Each CSV has **no header row**. Columns in fixed order, sampled at **100 Hz** (10 ms per row):

| # | Column | Unit | Description |
|---|--------|------|-------------|
| 0 | `timestamp_ms` | ms | Millisecond timestamp |
| 1 | `accel_x` | m/s² | X-axis acceleration |
| 2 | `accel_y` | m/s² | Y-axis acceleration |
| 3 | `accel_z` | m/s² | Z-axis acceleration |
| 4 | `accel_rms` | m/s² | RMS of all 3 axes |
| 5 | `tmp117_c` | °C | Precision digital temperature |
| 6 | `ntc_c` | °C | NTC thermistor temperature |
| 7 | `ina_a_ma` | mA | Motor current (channel A) |
| 8 | `ina_a_mw` | mW | Motor power (channel A) |
| 9 | `ina_b_ma` | mA | Channel B (dropped — not used for training) |
| 10 | `ina_b_mw` | mW | Channel B (dropped) |
| 11 | `motor_rpm` | RPM | Motor speed (dropped) |
| 12 | `load_class` | — | Class label 1–4 (read but labels are hardcoded per file) |

### Training Data

Four Session-A recordings, all collected at **150 RPM** with INA219-A, ADXL345, TMP117, and NTC thermistor:

| File | Label | Rows |
|------|-------|------|
| `class1_unloaded__1_.csv` | 1 (unloaded) | ~295K |
| `class2_light_load__1_.csv` | 2 (light_load) | ~185K |
| `class3_heavy_load__1_.csv` | 3 (heavy_load) | ~136K |
| `class4_stall.csv` | 4 (stall) | ~149K |

Labels are **hardcoded per file** in `nilm_pipeline.py`, not read from the `load_class` CSV column.

### Pipeline Steps

1. **Load & clean** — drop dead-current rows (`|current| < 2 mA`), trim class 2 to largest contiguous segment (timestamp-reset artifact)
2. **Chronological split** — 80/20 raw-row split per class with 128-sample purge gap to prevent window-boundary leakage
3. **Windowed feature extraction** — 128 samples (1.28 s), 64-sample step (50% overlap), 20 features
4. **5-fold purged blocked CV** — feature ablation across 5 variants, Random Forest and KNN evaluated
5. **Cross-session evaluation** — train on Session A, test on Session B (the honest benchmark)
6. **Export** — model saved to `ml/model.pkl`, C Random Forest arrays generated in `firmware/*/nilm_model.c`

### Feature Set (20 features)

The winning CV variant is **"No temp or accel FFT"** — 20 features from 6 raw sensor series:

| Series | Features | Source Column | Units |
|--------|----------|---------------|-------|
| Current | `cur_mean`, `cur_std`, `cur_min`, `cur_max`, `cur_range`, `cur_rms`, `cur_p25`, `cur_p75`, `cur_iqr` | `ina_a_ma` | mA |
| Power | `pwr_mean`, `pwr_std`, `pwr_range` | `ina_a_mw` | mW |
| Accel X | `accel_x_std`, `accel_x_range` | `accel_x` | m/s² |
| Accel Y | `accel_y_std`, `accel_y_range` | `accel_y` | m/s² |
| Accel Z | `accel_z_std`, `accel_z_range` | `accel_z` | m/s² |
| Accel RMS | `accel_rms_mean`, `accel_rms_std` | `accel_rms` | m/s² |

Only INA219-A is used for features. INA219-B, motor RPM, and both temperature columns (tmp117, ntc) are dropped at data-load time. The winning variant also drops all FFT spectral energy bins.

### Model Details

- **Algorithm:** Random Forest — 200 trees, max_depth=12, min_samples_leaf=5, class_weight="balanced"
- **Prediction:** Soft voting (average per-tree class probabilities, pick argmax)
- **Export:** `export_random_forest_firmware()` walks each tree and writes flat C arrays with threshold splits and per-node class probabilities

### Key Results

| Evaluation | Accuracy | Macro-F1 |
|---|---|---|
| Within-session (same recording) | 100.0% | 100.0% |
| Cross-session (Train A → Test B) | **76.8%** | **71.0%** |

Within-session 100% is expected (model memorizes the session's noise floor). Cross-session is the real benchmark. The model correctly identifies `unloaded` and `stall` with 100% cross-session accuracy. `heavy_load` confusion with `light_load` is due to vibration baseline drift between sessions — a hardware/physics issue, not a code bug.

### Saved Model (`ml/model.pkl`)

After every run the best model is serialized. Contains:

| Key | Type | Description |
|---|---|---|
| `model` | scikit-learn Pipeline | Fitted classifier (RF or KNN + scaler) |
| `feature_cols` | list[str] | Feature columns the model expects |
| `class_names` | dict | `{1: "unloaded", 2: "light_load", 3: "heavy_load", 4: "stall"}` |
| `window_size` | int | Samples per window (128) |
| `step_size` | int | Window step (64) |
| `variant` | str | Feature variant name selected by CV |
| `model_name` | str | "Random Forest" or "KNN" |

Usage:

```python
import pickle
from ml.nilm_pipeline import extract_window_features
import pandas as pd

with open("ml/model.pkl", "rb") as f:
    saved = pickle.load(f)

model        = saved["model"]
feature_cols = saved["feature_cols"]
class_names  = saved["class_names"]

feats = extract_window_features(window_df)  # window_df: 128 rows
X     = pd.DataFrame([feats])[feature_cols]
pred  = model.predict(X)[0]
print(class_names[pred])  # e.g. "heavy_load"
```

### Pipeline Configuration Constants

Located near the top of `ml/nilm_pipeline.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `WINDOW_SIZE` | 128 | Samples per feature window (1.28 s @ 100 Hz) |
| `STEP_SIZE` | 64 | Window step (50% overlap) |
| `SAMPLE_RATE` | 100 | Sensor sample rate (Hz) |
| `DEAD_CURRENT_THRESHOLD` | 2.0 mA | Motor-off threshold |
| `DEVELOPMENT_FRACTION` | 0.80 | Train/dev split ratio |
| `N_CV_SPLITS` | 5 | Cross-validation folds |
| `PURGE_GAP` | 128 | Leakage-prevention gap |

---

## ESP32 Firmware (Deployment)

The exported Random Forest runs on an ESP32 with bare-metal FreeRTOS firmware. Sensor data feeds the model at 100 Hz, predictions are published to Azure IoT Hub every 5 seconds.

### Prerequisites

- Docker with `espressif/idf:v6.0.2` image (ESP-IDF v6)
- ESP32 dev board with sensors attached

### Build & Flash

```bash
cd firmware/motor_telemetry

# 1. Edit .env with your credentials (gitignored, never committed)
#    WIFI_SSID, WIFI_PASS, IOT_HUB_HOST, DEVICE_ID, PRIMARY_KEY

# 2. Build
./build.sh build

# 3. Flash + monitor
./build.sh flash monitor
```

`build.sh` sources `.env`, passes secrets to Docker via `-e`, and runs `idf.py`. CMake reads `$ENV{...}` and injects them as `-D` compiler defines — **no credentials in source files**.

### Architecture

5 FreeRTOS tasks + SAS background refresh:

| Task | Priority | Stack | Role |
|---|---|---|---|
| `vMotorTask` | 2 | 2048 | RMT-driven A4988 step pulses, 150 RPM |
| `vSensorTask` | 3 | 16384 | I2C sensors @ 100 Hz, NILM feature extraction, ml_infer() |
| `vAzureTask` | 1 | 8192 | MQTT+TLS to Azure IoT Hub, SAS auth, 5 s publish |
| `sas_refresh_task` | 0 | 4096 | HMAC-SHA256 SAS token generator, 55 min cycle |
| `vWatchdogTask` | 1 | 2048 | TWDT health check, stuck detection, esp_restart() |

Sensors (bare I2C drivers, no Adafruit libraries):

| Sensor | I2C Addr | Role |
|---|---|---|
| ADXL345 | 0x53 | 3-axis vibration, ±16g → m/s² |
| INA219-A | 0x40 | Motor current (mA) + power (mW) — used for ML |
| INA219-B | 0x41 | Diagnostic only — not used for ML features |
| TMP117 | 0x48 | Ambient temperature |
| NTC 10k | ADC GPIO34 | Motor contact temperature |

Hardware: NEMA 17 stepper + A4988 driver (RMT GPIO25, DIR GPIO26, ENABLE GPIO27). 0.1 Ω shunt, INA219 cal=4096 → 100 µA/LSB current, 2 mW/LSB power.

### NILM Inference

Runs inside `vSensorTask` at 100 Hz. A rolling 128-sample window accumulates readings (dead-current filter skips `|current| < 2 mA`). Every 64 samples, 20 features are computed using the same C code path as the Python training pipeline. The exported 200-tree Random Forest predicts a class via soft voting. Result appears in Azure telemetry as `status` and `status_message`.

### C ↔ Python Consistency Guarantees

Python's `extract_window_features()` and C's `nilm_feature_value()` must stay in sync. Verification tools:

| Check | Tool |
|---|---|
| Sensor units match (m/s², mA, mW) | `compare_nilm_features.py` |
| Feature values match | `nilm_validate_features.c` → JSON dump → `compare_nilm_features.py` |
| Model export is valid C | `gcc -fsyntax-only -std=c11 firmware/*/nilm_model.c` |

### Reconnection System (3-layer)

| Layer | Mechanism | Behavior |
|---|---|---|
| WiFi | `wifi_event_handler` | Permanent failures (AUTH_FAIL, NO_AP_FOUND, ASSOC_LEAVE) abort retry. All others: exponential backoff `MIN(1000 << n, 60000)` ms, counter reset on `GOT_IP`. |
| MQTT | `vAzureTask` | Auto-reconnect disabled. Publish failures tracked; at 5 consecutive fails, forces watchdog restart. Start reconnects use exponential backoff, destroy/recreate client handle each attempt. |
| Watchdog | `vWatchdogTask` | TWDT (60 s timeout). Checks `g_sas_valid && g_connected` every 30 s. 10 consecutive failures triggers `esp_restart()`. |

### TLS Configuration

Uses ESP x509 certificate bundle with proper PKI verification:

```c
#include "esp_crt_bundle.h"
.broker.verification.crt_bundle_attach = esp_crt_bundle_attach,
```

Enables `CONFIG_MBEDTLS_CERTIFICATE_BUNDLE=y` + `CONFIG_MBEDTLS_CERTIFICATE_BUNDLE_DEFAULT_FULL=y` in sdkconfig. The full default bundle includes DigiCert Global Root G2 (Azure IoT Hub's CA). No `skip_cert_common_name_check` — that bypasses verification entirely.

### SAS Token Format

```
SharedAccessSignature sig={URLENCODED_BASE64_HMAC}&se={EPOCH_EXPIRY}&sr={URLENCODED_RESOURCE_URI}
```

- **sr** = `{iothub-hostname}/devices/{device-id}` (URL-encoded, NOT just the hostname)
- **sig** = URL-encoded Base64 of HMAC-SHA256 over `{URLENCODED_SR}\n{EXPIRY}`
- **se** = UNIX epoch expiry (3600 s TTL)
- Tokens refreshed every 55 min (5 min before expiry)
- Epoch guard (`MIN_VALID_EPOCH = Jan 1 2024`) blocks generation before NTP syncs

---

## Requirements

- Python 3.10+ — `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`
- Docker — `espressif/idf:latest` for ESP-IDF v6 firmware builds
- ESP32 dev board + sensors for deployment
