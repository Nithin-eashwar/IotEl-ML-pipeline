# NILM Motor Load Classifier

A leakage-resistant machine-learning pipeline for **predictive maintenance of motors** using Non-Intrusive Load Monitoring (NILM) techniques.

The pipeline classifies motor operating states — `unloaded`, `light_load`, `heavy_load`, and `stall` — from raw sensor data (current, power, vibration) using a windowed feature-extraction approach and rigorous cross-session evaluation.

---

## Project Structure

```
IotEl/
├── ml/
│   └── nilm_pipeline.py          # Main ML pipeline (run this)
│
├── data/                         # CSV dataset files (NOT in git — see below)
│   ├── class1_unloaded__1_.csv
│   ├── class2_light_load__1_.csv
│   ├── class3_heavy_load__1_.csv
│   ├── class4_stall.csv
│   ├── 1_unloaded_test.csv
│   ├── 2_light_load_test.csv
│   ├── 3_heavy_load_test.csv
│   └── 4_stall_test.csv
│
├── firmware/                     # IoT sensor firmware
├── docs/                         # Additional documentation
├── results/                      # Generated output — auto-created, NOT in git
├── NILM_Cross_Session_Report.md  # Full analysis report
├── requirements.txt
└── README.md
```

> **Note:** `data/` and `results/` are excluded from version control (see `.gitignore`).
> Get dataset files from the shared drive or ask a teammate.

---

## Sensor Columns (CSV format)

Each CSV has **no header row**. Columns in fixed order:

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
| 9 | `ina_b_ma` | mA | Channel B (unused) |
| 10 | `ina_b_mw` | mW | Channel B (unused) |
| 11 | `motor_rpm` | RPM | Motor speed (unused) |
| 12 | `load_class` | — | Class label (1–4) |

Sampling rate: **100 Hz** (10 ms per sample).

---

## Setup

### Prerequisites
- Python 3.10+
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
Place all 8 CSV files in the `data/` directory (create it if it doesn't exist).

---

## Running the Pipeline

The pipeline supports two evaluation modes:

### Default — Cross-Session (~77% accuracy)
```bash
python ml/nilm_pipeline.py
# or explicitly:
python ml/nilm_pipeline.py --mode cross_session
```
Trains on Session A recordings, tests on Session B recordings. This is the primary result used in the report and presentations.

### LOSO — 2-Fold Leave-One-Session-Out (~65% accuracy)
```bash
python ml/nilm_pipeline.py --mode loso
```
Runs two folds (A→B and B→A), averaging the results. This is the most rigorous evaluation given two sessions of data.

> **For demos and presentations, use the default `cross_session` mode.**

---

## What the Pipeline Does

```
[1]  Load & clean raw CSVs
       • Dead-current rows removed (|current| < 2 mA)
       • Class 2 trimmed to largest contiguous segment
         (removes a known timestamp-reset artefact)

[2]  Chronological 80/20 raw-row split per class
       • 128-sample purge gap prevents window-boundary leakage

[3]  Windowed feature extraction
       • Window: 128 samples (1.28s), step: 64 (50% overlap)
       • Current & power statistics (mean, std, min, max, RMS, IQR)
       • Per-axis accel statistics (std, range) + FFT spectral energy
       • accel_rms statistics, temperature means

[4]  Purged blocked 5-fold cross-validation on development set
       • Feature ablation: 5 variants (all 46 → current+power only 12)
       • Models: Random Forest (200 trees) and KNN (k=7)
       • Best selected by macro-F1, accuracy, then fewest features

[5]  Within-session held-out test
[6]  Detailed classification report + text confusion matrix
[7]  Top-15 Random Forest feature importances

[A–D] Cross-session evaluation sections (the honest numbers)
       • Per-class window accuracy on unseen session data
       • Matplotlib confusion matrix saved to results/
       • Signal statistics comparison between sessions
```

---

## Key Results

| Evaluation | Accuracy | Macro-F1 |
|---|---|---|
| Within-session (same recording) | 100.0% | 100.0% |
| Cross-session (Train A → Test B) | **76.8%** | **71.0%** |
| LOSO average (2-fold) | 64.8% | 56.8% |

> Within-session 100% is expected and not meaningful — the model memorises the session's noise floor. **Cross-session figures are the real benchmark.**

The model correctly identifies `unloaded` and `stall` with 100% cross-session accuracy. `heavy_load` is confused with `light_load` due to vibration baseline drift between sessions — a known hardware/physics issue, not a code bug.

---

## Key Configuration Constants

Located near the top of `ml/nilm_pipeline.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `WINDOW_SIZE` | 128 | Samples per feature window (1.28s @ 100 Hz) |
| `STEP_SIZE` | 64 | Window step size (50% overlap) |
| `SAMPLE_RATE` | 100 | Sensor sample rate (Hz) |
| `DEAD_CURRENT_THRESHOLD` | 2.0 mA | Motor-off threshold |
| `DEVELOPMENT_FRACTION` | 0.80 | Train/dev split ratio |
| `N_CV_SPLITS` | 5 | Cross-validation folds |
| `PURGE_GAP` | 128 | Leakage-prevention gap (raw rows) |
| `SEGMENT_GAP_MS` | 500 | Timestamp gap marking a sub-recording break |

---

## Requirements

See `requirements.txt`. Core dependencies:
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
