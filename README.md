# NILM Motor Load Classifier

A leakage-resistant machine-learning pipeline for **predictive maintenance of motors** using Non-Intrusive Load Monitoring (NILM) techniques.

The pipeline classifies motor operating states — `unloaded`, `light_load`, `heavy_load`, and `stall` — from raw sensor data (current, power, vibration, temperature) using a windowed feature-extraction approach and chronological cross-validation.

---

## Project Structure

```
IotEl/
├── nilm_pipeline.py              # Main pipeline (run this)
├── class1_unloaded__1_.csv       # Session A — unloaded motor recording
├── class2_light_load__1_.csv     # Session A — light load recording
├── class3_heavy_load__1_.csv     # Session A — heavy load recording
├── class4_stall.csv              # Session A — stall recording
├── class3_heavy_load_test.csv    # Session B — cross-session heavy load test
├── requirements.txt              # Python dependencies
└── README.md
```

> **Note:** CSV data files are excluded from version control (see `.gitignore`).  
> Ask a teammate or check your shared drive for the dataset files.

---

## Sensor Columns (CSV format)

Each CSV has **no header row**. Columns are in this fixed order:

| # | Column | Unit | Description |
|---|--------|------|-------------|
| 0 | `timestamp_ms` | ms | Millisecond timestamp |
| 1 | `accel_x` | m/s² | X-axis acceleration |
| 2 | `accel_y` | m/s² | Y-axis acceleration |
| 3 | `accel_z` | m/s² | Z-axis acceleration |
| 4 | `accel_rms` | m/s² | RMS of all 3 accel axes |
| 5 | `tmp117_c` | °C | Precision digital temperature |
| 6 | `ntc_c` | °C | NTC thermistor temperature |
| 7 | `ina_a_ma` | mA | Motor current (channel A) |
| 8 | `ina_a_mw` | mW | Motor power (channel A) |
| 9 | `ina_b_ma` | mA | Current channel B (unused) |
| 10 | `ina_b_mw` | mW | Power channel B (unused) |
| 11 | `motor_rpm` | RPM | Motor speed (unused) |
| 12 | `load_class` | — | Class label (1–4) |

Sampling rate: **100 Hz** (10 ms per sample).

---

## Setup

### Prerequisites

- Python 3.10 or newer
- pip

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd IotEl
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset CSVs

Place the CSV files in the project root (same folder as `nilm_pipeline.py`).  
The required files are:

```
class1_unloaded__1_.csv
class2_light_load__1_.csv
class3_heavy_load__1_.csv
class4_stall.csv
class3_heavy_load_test.csv        ← cross-session test (utf-16 encoded)
```

---

## Running the Pipeline

```bash
python nilm_pipeline.py
```

The pipeline runs end-to-end and prints all results to the terminal. No arguments needed.

---

## What the Pipeline Does

```
[1]  Load & clean raw CSVs
       • Dead-current rows removed (|current| < 2 mA)
       • Class 2 trimmed to largest contiguous segment
         (removes a known timestamp reset / concatenation artefact)

[2]  Chronological 80/20 raw-row split per class
       • A 128-sample purge gap separates development from test
         to prevent window-boundary leakage

[3]  Windowed feature extraction (128-sample windows, 50% overlap)
       • Current & power statistics (mean, std, min, max, RMS, IQR, percentiles)
       • Per-axis accel statistics (std, range) + FFT spectral energy (8 bins)
       • accel_rms statistics (mean, std)
       • Temperature means (NTC + TMP117)

[4]  Purged blocked 5-fold cross-validation on the development set
       • Feature ablation across 5 variants (all features → current+power only)
       • Models: Random Forest (200 trees) and KNN (k=7, distance-weighted)
       • Best model selected by macro-F1, then accuracy, then fewest features

[5]  Within-session held-out test  ← later rows from the same recording
[6]  Detailed classification report + confusion matrix for the best model
[7]  Top-15 feature importances (Random Forest only)

[A]  Load cross-session test data  ← separate recording / different session
[B]  Per-class window accuracy on the cross-session data
[C]  Detailed classification report for the cross-session data
[D]  Raw signal statistics: session A vs cross-session comparison
```

> **Interpreting results:** Within-session accuracy is expected to be very high
> (the motor is in a stable operating point). The cross-session sections
> **[B–D]** are what really show whether the model generalises.

---

## Adding More Cross-Session Data

To add new-session recordings for other classes, edit the `CROSS_SESSION_MAP`
dictionary near the top of `nilm_pipeline.py`:

```python
CROSS_SESSION_MAP = {
    1: ("class1_unloaded_session2.csv",   "utf-8"),
    2: ("class2_light_load_session2.csv", "utf-8"),
    3: ("class3_heavy_load_test.csv",     "utf-16"),   # already included
    4: ("class4_stall_session2.csv",      "utf-8"),
}
```

The encoding for most new files will be `"utf-8"`. Use `"utf-16"` only if
the file was saved that way (the current class 3 test file is utf-16).

---

## Key Configuration Constants

All tunable parameters are at the top of `nilm_pipeline.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `WINDOW_SIZE` | 128 | Samples per feature window (1.28 s at 100 Hz) |
| `STEP_SIZE` | 64 | Window step size (50% overlap) |
| `SAMPLE_RATE` | 100 | Sensor sample rate (Hz) |
| `DEAD_CURRENT_THRESHOLD` | 2.0 mA | Rows below this are treated as motor-off |
| `DEVELOPMENT_FRACTION` | 0.80 | Fraction of each class used for training/CV |
| `N_CV_SPLITS` | 5 | Number of cross-validation folds |
| `PURGE_GAP` | 128 | Raw rows removed at the train/test boundary |
| `SEGMENT_GAP_MS` | 500 | Timestamp gap (ms) that marks a sub-recording break |

---

## Requirements

See `requirements.txt`. Core dependencies:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
