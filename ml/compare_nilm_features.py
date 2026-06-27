#!/usr/bin/env python3
"""
NILM Feature Comparison Tool
=============================
Reads the JSON dump produced by firmware/nilm_validate_features.c and
recomputes all 20 features in Python using the exact same formulas.
Reports per-feature differences.
"""

import sys, json, numpy as np
import re

WINDOW_SIZE = 128
SAMPLE_RATE = 100.0
NILM_DEAD_CURRENT_MA = 2.0

FEATURE_NAMES = [
    "cur_mean", "cur_std", "cur_min", "cur_max", "cur_range",
    "cur_rms", "cur_p25", "cur_p75", "cur_iqr",
    "pwr_mean", "pwr_std", "pwr_range",
    "accel_x_std", "accel_x_range", "accel_y_std", "accel_y_range",
    "accel_z_std", "accel_z_range",
    "accel_rms_mean", "accel_rms_std",
]


def extract_features_python(samples):
    """Exact Python replica of esp32_telemetry.c feature extraction.

    samples: list of dicts with keys:
        accel_x, accel_y, accel_z, accel_rms, tmp117_c, ntc_c, ina_a_ma, ina_a_mw
    """
    feats = {}

    # ---- Current (from ina_a_ma, mA) ----
    cur = np.array([s["ina_a_ma"] for s in samples], dtype=np.float64)
    feats["cur_mean"]  = float(np.mean(cur))
    feats["cur_std"]   = float(np.std(cur))
    feats["cur_min"]   = float(np.min(cur))
    feats["cur_max"]   = float(np.max(cur))
    feats["cur_range"] = feats["cur_max"] - feats["cur_min"]
    feats["cur_rms"]   = float(np.sqrt(np.mean(cur ** 2)))
    feats["cur_p25"]   = float(np.percentile(cur, 25))
    feats["cur_p75"]   = float(np.percentile(cur, 75))
    feats["cur_iqr"]   = feats["cur_p75"] - feats["cur_p25"]

    # ---- Power (from ina_a_mw, mW) ----
    pwr = np.array([s["ina_a_mw"] for s in samples], dtype=np.float64)
    feats["pwr_mean"]  = float(np.mean(pwr))
    feats["pwr_std"]   = float(np.std(pwr))
    feats["pwr_range"] = float(np.max(pwr) - np.min(pwr))

    # ---- Accelerometer axes ----
    for axis in ["accel_x", "accel_y", "accel_z"]:
        sig = np.array([s[axis] for s in samples], dtype=np.float64)
        feats[f"{axis}_std"]   = float(np.std(sig))
        feats[f"{axis}_range"] = float(np.max(sig) - np.min(sig))

    # ---- Accel RMS ----
    rms = np.array([s["accel_rms"] for s in samples], dtype=np.float64)
    feats["accel_rms_mean"] = float(np.mean(rms))
    feats["accel_rms_std"]  = float(np.std(rms))

    return feats


def parse_dump(lines):
    """Extract the JSON block between START/END markers."""
    inside = False
    buf = []
    for line in lines:
        if "=== NILM_WINDOW_DUMP_START ===" in line:
            inside = True
            buf = []
            continue
        if "=== NILM_WINDOW_DUMP_END ===" in line:
            return "\n".join(buf)
        if inside:
            buf.append(line)
    raise ValueError("No NILM_WINDOW_DUMP markers found")


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            content = f.read()
    else:
        print("Reading from stdin (paste the ESP32 JSON dump, then Ctrl+D)...")
        content = sys.stdin.read()

    json_str = parse_dump(content.splitlines())
    dump = json.loads(json_str)

    # Parse samples: each row is [ax, ay, az, a_rms, tmp, ntc, ina_a_ma, ina_a_mw]
    keys = ["accel_x", "accel_y", "accel_z", "accel_rms",
            "tmp117_c", "ntc_c", "ina_a_ma", "ina_a_mw"]
    samples = [dict(zip(keys, row)) for row in dump["samples"]]
    esp_features = dump["features"]
    py_features = extract_features_python(samples)

    print("\n{:20s}  {:>14s}  {:>14s}  {:>10s}  {:>8s}".format(
        "Feature", "ESP32 (C)", "Python", "Δ", "Δ%"))
    print("-" * 80)

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    all_match = True

    for name in FEATURE_NAMES:
        c_val = float(esp_features[name])
        py_val = float(py_features[name])
        diff = c_val - py_val

        if abs(py_val) > 1e-9:
            rel_diff = abs(diff / py_val) * 100
        else:
            rel_diff = abs(diff) * 100 if abs(diff) > 1e-9 else 0.0

        max_abs_diff = max(max_abs_diff, abs(diff))
        max_rel_diff = max(max_rel_diff, rel_diff)

        flag = ""
        if rel_diff > 1.0:
            flag = " ⚠"
            all_match = False
        elif rel_diff > 0.1:
            flag = " ~"

        print("{:20s}  {:14.6g}  {:14.6g}  {:10.3g}  {:7.2f}%{}".format(
            name, c_val, py_val, diff, rel_diff, flag))

    print("-" * 80)
    print(f"\nMax absolute difference: {max_abs_diff:.6g}")
    print(f"Max relative difference: {max_rel_diff:.2f}%")

    if all_match:
        print("\n✓ ALL FEATURES MATCH within 1% tolerance.")
    elif max_rel_diff < 5.0:
        print("\n⚠ Minor differences detected — likely float32 vs float64 precision.")
    else:
        print("\n✗ SIGNIFICANT MISMATCHES — check sensor drivers, units, or formulas.")

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
