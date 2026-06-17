# Repository Guidelines

## Project Structure & Module Organization

This repository contains a NILM motor-load classifier and related ESP32 firmware.

- `ml/nilm_pipeline.py`: main Python ML pipeline for loading CSV data, extracting window features, training, evaluating, and writing `ml/model.pkl`.
- `single.py`: standalone Python workflow used alongside the main pipeline.
- `test_azure_connection.py`: Azure IoT Hub connection diagnostic.
- `firmware/iot_testbed/`: active Arduino sketch for 100 Hz sensor capture and CSV streaming.
- `firmware/legacy/`: modular ESP32/FreeRTOS firmware with sensor, motor, NILM, calibration, and MQTT components.
- `docs/` and `NILM_Cross_Session_Report.md`: design notes and evaluation report.
- `data/`, `results/`, `*.csv`, `*.png`, `*.pkl`, and `.env` are generated or local-only and must not be committed.

## Build, Test, and Development Commands

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the primary evaluation:

```bash
python ml/nilm_pipeline.py
python ml/nilm_pipeline.py --mode loso
```

Check Azure IoT Hub credentials and connectivity:

```bash
python test_azure_connection.py
```

Compile/upload firmware from the Arduino IDE or `arduino-cli`, selecting the ESP32 board and installing the libraries used by the sketch, such as `Adafruit_ADXL345_U`, `SparkFun_TMP117`, and `Adafruit_INA219`.

## Coding Style & Naming Conventions

Use Python 3.10+ with 4-space indentation, `snake_case` functions and variables, and uppercase constants for pipeline configuration such as `WINDOW_SIZE` and `DATA_DIR`. Keep data-loading, feature-extraction, and evaluation logic separated. Firmware uses C++/Arduino style with `.h`/`.cpp` modules, uppercase `#define` constants, and clear task names such as `motorTask` and `sensorTask`.

## Testing Guidelines

There is no formal pytest suite yet. Validate ML changes by running `python ml/nilm_pipeline.py` with the required CSVs in `data/`, then compare accuracy, macro-F1, reports, and generated files in `results/`. For cloud changes, run `python test_azure_connection.py` with a local `.env` based on `.env.example`. Do not commit credentials or generated artifacts.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, for example `readme updated` and `Refactored ml pipeline and added the firmware with updated readme`. Keep commits focused and mention the affected area when useful, such as `ml: adjust cross-session feature selection`. Pull requests should describe the behavior change, list commands run, note dataset assumptions, and include screenshots or plots when evaluation output changes.
