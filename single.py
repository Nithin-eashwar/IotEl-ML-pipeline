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
def main():
    bundle = load_model(MODEL_PATH)
    window_size = bundle["window_size"]   # 128
    step_size   = bundle["step_size"]     # 64

    # Connect to Azure IoT Hub
    print("\nConnecting to Azure IoT Hub...")
    client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
    client.connect()
    print("[✓] Connected to Azure IoT Hub\n")

    # Open the serial port
    print(f"Opening serial port {COM_PORT} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
    except serial.SerialException as e:
        print(f"[ERROR] Could not open {COM_PORT}: {e}")
        print("  Check the port name and make sure the ESP32 is connected.")
        client.disconnect()
        sys.exit(1)
    print(f"[✓] Serial port open. Waiting for sensor data...\n")

    # Rolling buffer — holds up to window_size rows
    buffer = deque(maxlen=window_size)
    rows_since_last_send = 0
    total_windows_sent = 0

    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue

            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                continue

            row = parse_row(line)
            if row is None:
                # Print firmware init messages so we can see sensor status
                if raw.decode("utf-8", errors="replace").startswith("#"):
                    print(raw.decode("utf-8", errors="replace").strip())
                continue

            buffer.append(row)
            rows_since_last_send += 1

            # Once we have a full window AND have accumulated step_size new rows
            if len(buffer) == window_size and rows_since_last_send >= step_size:
                rows_since_last_send = 0
                window_rows = list(buffer)

                # Run inference and build payload
                payload = build_payload(window_rows, bundle)

                # Send to Azure IoT Hub
                msg = Message(json.dumps(payload))
                msg.content_encoding = "utf-8"
                msg.content_type = "application/json"
                client.send_message(msg)

                total_windows_sent += 1
                ts = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{ts}] Window #{total_windows_sent:04d}  "
                    f"→ {payload['class_name']:12s}  "
                    f"conf={payload['confidence']:5.1f}%  "
                    f"status={payload['status']}"
                )

    except KeyboardInterrupt:
        print("\n\nStopping bridge...")

    finally:
        ser.close()
        client.disconnect()
        print("Serial port closed. Disconnected from Azure IoT Hub.")


if __name__ == "__main__":
    main()
