"""
single.py — Azure IoT Hub Bridge
=================================
Reads live 100 Hz sensor data from the ESP32 over USB Serial,
accumulates 128-sample windows, runs ML inference using the
pre-trained Random Forest (ml/model.pkl), and forwards predictions
to Azure IoT Hub every ~0.64 seconds (one step of 64 samples).

Setup
-----
1. Run the ML pipeline once to generate the model:
       python ml/nilm_pipeline.py
2. Install extra dependency:
       pip install pyserial azure-iot-device
3. Set COM_PORT below to match your ESP32 (e.g. "COM5" on Windows,
   "/dev/ttyUSB0" on Linux/Mac). Run this to find it:
       python -m serial.tools.list_ports
4. Run the bridge:
       python single.py

Azure IoT Hub
-------------
Messages arrive at the Hub every ~0.64 s.
The frontend teammate should listen to the Hub and consume the JSON
payload described in the PAYLOAD section below.
"""

import json
import os
import pickle
import sys
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import serial
from azure.iot.device import IoTHubDeviceClient, Message
from dotenv import load_dotenv

# Import feature extractor from the ML pipeline.
# sys.path is set here once so nilm_pipeline can be found from IotEl/single.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml"))
from nilm_pipeline import extract_window_features  # noqa: E402

# Load credentials from .env (must be in the repo root alongside this file)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if not os.path.exists(_env_path):
    print("[ERROR] .env file not found.")
    print(f"  Expected: {_env_path}")
    print("  Copy .env.example to .env and fill in your credentials.")
    sys.exit(1)
load_dotenv(_env_path)

# ─── USER CONFIG ──────────────────────────────────────────────────────────────
# Set this to the COM port the ESP32 is connected to.
# Windows example: "COM5"   Linux/Mac example: "/dev/ttyUSB0"
COM_PORT = "COM5"
BAUD_RATE = 115200

# ─── AZURE IOT HUB ────────────────────────────────────────────────────────────
_required_vars = ["IOT_HUB_HOSTNAME", "IOT_HUB_DEVICE_ID", "IOT_HUB_SHARED_ACCESS_KEY"]
_missing = [v for v in _required_vars if not os.environ.get(v)]
if _missing:
    print(f"[ERROR] Missing environment variables: {', '.join(_missing)}")
    print("  Check your .env file. See .env.example for the required keys.")
    sys.exit(1)

DEVICE_ID   = os.environ["IOT_HUB_DEVICE_ID"]
CONNECTION_STRING = (
    f"HostName={os.environ['IOT_HUB_HOSTNAME']};"
    f"DeviceId={os.environ['IOT_HUB_DEVICE_ID']};"
    f"SharedAccessKey={os.environ['IOT_HUB_SHARED_ACCESS_KEY']}"
)

# ─── MODEL ────────────────────────────────────────────────────────────────────
# Path is relative to this file's location (IotEl/single.py → IotEl/ml/model.pkl)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml", "model.pkl")

# ─── CSV COLUMN SCHEMA ────────────────────────────────────────────────────────
# Must match the firmware's Serial.print order exactly.
COLS = [
    "timestamp_ms", "accel_x", "accel_y", "accel_z", "accel_rms",
    "tmp117_c", "ntc_c", "ina_a_ma", "ina_a_mw",
    "ina_b_ma", "ina_b_mw", "motor_rpm", "load_class",
]

# ─── STATUS MAPPING ───────────────────────────────────────────────────────────
# Maps ML class → (status string, base status_message) using the exact
# values the frontend already understands.
STATUS_MAP = {
    1: ("Active",   "Motor running unloaded"),
    2: ("Active",   "Motor under light load"),
    3: ("Warning",  "Motor under heavy load"),
    4: ("Critical", "Stall detected \u2014 motor shaft may be jammed"),
}


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
def load_model(path):
    if not os.path.exists(path):
        print(f"[ERROR] model.pkl not found at: {path}")
        print("  Run the ML pipeline first:  python ml/nilm_pipeline.py")
        sys.exit(1)
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"[✓] Model loaded: {bundle['model_name']} / {bundle['variant']}")
    print(f"    Features : {len(bundle['feature_cols'])}")
    print(f"    Classes  : {bundle['class_names']}")
    return bundle


# ─── PARSE A SINGLE SERIAL ROW ────────────────────────────────────────────────
def parse_row(line: str):
    """
    Parse a raw CSV line from the firmware into a dict.
    Returns None if the line is a comment, header, or malformed.
    """
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("timestamp"):
        return None                        # skip firmware comments and header
    parts = line.split(",")
    if len(parts) < 13:
        return None                        # malformed row
    try:
        return {col: float(parts[i]) for i, col in enumerate(COLS)}
    except ValueError:
        return None


# ─── BUILD INFERENCE PAYLOAD ──────────────────────────────────────────────────
def build_payload(window_rows: list, bundle: dict) -> dict:
    """
    Given 128 parsed sensor rows, run ML inference and build the Azure message
    using the FIXED frontend schema:

    {
        "device_id":      "stepper-motor-1-tvl5ry",
        "timestamp":      "2026-06-10T14:00:00Z",
        "rpm":            150.0,          # firmware constant (CONSTANT_RPM)
        "temperature":    28.5,           # NTC thermistor (°C)
        "vibration":      2.13,           # accel_rms_std of the window
        "current":        87.0,           # cur_mean_ma (unit: mA — NOT Amps)
        "status":         "Critical",     # Active / Warning / Critical
        "status_message": "Stall detected ... [ML: stall, conf: 94.1%]"
    }

    NOTE FOR FRONTEND TEAMMATE:
      - 'current' is in milliamps (mA). The INA219 on our board measures mA.
        Typical values: unloaded ~67 mA, light_load ~73 mA, stall ~67 mA.
      - 'vibration' is accel_rms_std (standard deviation of accelerometer RMS
        over the 1.28-second window), NOT mm/s. Typical: stall ~2.2, others <1.2.
      - 'rpm' is always 150.0 — the stepper motor is locked to a constant speed.
      - The ML class and confidence are embedded in status_message.
    """
    df = pd.DataFrame(window_rows)
    features = extract_window_features(df)
    feature_cols = bundle["feature_cols"]
    X = pd.DataFrame([features]).reindex(columns=feature_cols, fill_value=0)

    model       = bundle["model"]
    class_names = bundle["class_names"]   # {1: "unloaded", ..., 4: "stall"}
    pred_label  = int(model.predict(X)[0])
    class_name  = class_names[pred_label]

    # Confidence (Random Forest supports predict_proba; fall back gracefully)
    try:
        proba      = model.predict_proba(X)[0]
        confidence = round(float(max(proba)) * 100, 1)
    except AttributeError:
        confidence = 100.0

    # Map ML result to frontend status values
    status, base_message = STATUS_MAP.get(pred_label, ("Active", "Unknown state"))
    status_message = f"{base_message} [ML: {class_name}, conf: {confidence}%]"

    # Map sensor window stats to the fixed schema fields
    cur             = df["ina_a_ma"].values
    cur_mean_ma     = round(float(np.mean(cur)), 2)
    temperature_c   = round(float(df["ntc_c"].iloc[-1]), 2)
    vibration_index = round(float(np.std(df["accel_rms"].values)), 3)
    rpm             = round(float(df["motor_rpm"].iloc[-1]), 1)   # always 150.0

    return {
        "device_id":      DEVICE_ID,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "rpm":            rpm,
        "temperature":    temperature_c,
        "vibration":      vibration_index,
        "current":        cur_mean_ma,
        "status":         status,
        "status_message": status_message,
    }


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def _run_loop(row_source, bundle, client, send: bool = True):
    """
    Core inference + send loop — shared by both live (serial) and test (CSV) modes.
    row_source: an iterable that yields raw CSV line strings one at a time.
    """
    window_size = bundle["window_size"]   # 128
    step_size   = bundle["step_size"]     # 64

    buffer               = deque(maxlen=window_size)
    rows_since_last_send = 0
    total_windows_sent   = 0

    for raw_line in row_source:
        row = parse_row(raw_line)
        if row is None:
            if raw_line.strip().startswith("#"):
                print(raw_line.strip())   # show firmware init messages
            continue

        buffer.append(row)
        rows_since_last_send += 1

        if len(buffer) == window_size and rows_since_last_send >= step_size:
            rows_since_last_send = 0
            payload = build_payload(list(buffer), bundle)

            if send:
                msg = Message(json.dumps(payload))
                msg.content_encoding = "utf-8"
                msg.content_type = "application/json"
                client.send_message(msg)

            total_windows_sent += 1
            ts = datetime.now().strftime("%H:%M:%S")
            sent_flag = "" if send else "  [NOT SENT — --no-send mode]"
            print(
                f"[{ts}] Window #{total_windows_sent:04d}  "
                f"→ {payload['status']:8s} | {payload['status_message']}{sent_flag}"
            )

    print(f"\n  Done. {total_windows_sent} windows processed.")


def _csv_row_source(csv_path: str, speed: float = 1.0):
    """
    Yields lines from a CSV file, optionally throttled to simulate real-time.
    speed=1.0  → real 100 Hz (10 ms per row)
    speed=10.0 → 10× faster (1 ms per row)
    speed=0    → as fast as possible (no sleep)
    """
    delay = (1.0 / 100.0) / speed if speed > 0 else 0   # seconds per row

    print(f"  Replaying: {csv_path}")
    print(f"  Speed    : {speed}× real-time  ({delay * 1000:.1f} ms / row)\n")

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line
            if delay:
                time.sleep(delay)


def _serial_row_source(com_port: str, baud_rate: int):
    """Yields lines from a live serial port indefinitely."""
    try:
        ser = serial.Serial(com_port, baud_rate, timeout=2)
    except serial.SerialException as e:
        print(f"[ERROR] Could not open {com_port}: {e}")
        print("  Check the port name and make sure the ESP32 is connected.")
        sys.exit(1)

    print(f"[✓] Serial port open ({com_port}). Waiting for sensor data...\n")
    try:
        while True:
            raw = ser.readline()
            if raw:
                yield raw.decode("utf-8", errors="replace")
    finally:
        ser.close()
        print("Serial port closed.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="NILM bridge — send ML predictions to Azure IoT Hub"
    )
    parser.add_argument(
        "--test",
        metavar="CSV_FILE",
        default=None,
        help=(
            "Run in test mode: replay a CSV from the data/ folder instead of "
            "reading live from the serial port. "
            "Example: --test 4_stall_test.csv"
        ),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=5.0,
        help=(
            "Replay speed multiplier (test mode only). "
            "1.0 = real-time 100 Hz, 10.0 = 10× faster, 0 = no delay. "
            "Default: 5.0"
        ),
    )
    parser.add_argument(
        "--no-send",
        action="store_true",
        help="Run inference but do NOT send anything to Azure (dry-run).",
    )
    args = parser.parse_args()

    bundle = load_model(MODEL_PATH)

    # Connect to Azure (skipped in --no-send mode)
    client = None
    if not args.no_send:
        print("\nConnecting to Azure IoT Hub...")
        client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
        client.connect()
        print("[✓] Connected to Azure IoT Hub")
    else:
        print("\n[--no-send] Dry-run mode — inference only, nothing sent to Azure.")

    print()

    try:
        if args.test:
            # ── Test / replay mode ────────────────────────────────────────────
            csv_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data", args.test
            )
            if not os.path.exists(csv_path):
                print(f"[ERROR] Test file not found: {csv_path}")
                print(f"  Available files in data/:")
                for f in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")):
                    if f.endswith(".csv"):
                        print(f"    {f}")
                sys.exit(1)
            row_source = _csv_row_source(csv_path, speed=args.speed)
        else:
            # ── Live serial mode ──────────────────────────────────────────────
            if COM_PORT == "COM_PORT_PLACEHOLDER":
                print("[ERROR] COM_PORT is not set.")
                print("  Edit single.py and set COM_PORT to your ESP32 port.")
                print("  Run:  python -m serial.tools.list_ports")
                sys.exit(1)
            print(f"Opening serial port {COM_PORT} at {BAUD_RATE} baud...")
            row_source = _serial_row_source(COM_PORT, BAUD_RATE)

        _run_loop(row_source, bundle, client, send=not args.no_send)

    except KeyboardInterrupt:
        print("\n\nStopping bridge...")

    finally:
        if client:
            client.disconnect()
            print("Disconnected from Azure IoT Hub.")


if __name__ == "__main__":
    main()
