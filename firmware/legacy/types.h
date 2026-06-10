#pragma once
#include <stdint.h>

// ═══════════════════════════════════════════════════════════════════
//  types.h — Shared structs used across all firmware modules
// ═══════════════════════════════════════════════════════════════════

// Raw accelerometer reading (m/s² or g — depends on ADXL345 range config)
struct AccelData {
    float x   = 0.0f;
    float y   = 0.0f;
    float z   = 0.0f;
    float rms = 0.0f;   // sqrt(x²+y²+z²)
    bool  valid = false;
};

// NILM feature vector — one per sensor loop tick
struct FeatureVector {
    float delta_I   = 0.0f;    // Current change (mA) — INA219 (null until arrived)
    float delta_P   = 0.0f;    // Power change (mW)   — derived from INA219
    float delta_Q   = 0.0f;    // Reactive power       — derived
    float accel_rms = 0.0f;    // ADXL345 RMS
    float t_winding = 0.0f;    // NTC corrected temp
    float delta_T   = 0.0f;    // (T_ntc - T_tmp117) - baseline_offset

    // Availability flags
    bool  ina219_valid = false; // false until INA219 modules arrive
};

// Telemetry payload — serialised to JSON and published over MQTT
struct TelemetryPayload {
    uint64_t    timestamp_ms = 0;
    AccelData   accel;
    float       t_ambient   = 0.0f;
    float       t_winding   = 0.0f;
    float       delta_t     = 0.0f;
    float       i_phaseA    = 0.0f;   // NaN if INA219 not available
    float       i_phaseB    = 0.0f;   // NaN if INA219 not available
    uint32_t    motor_rpm   = 0;
    const char* load_class  = "unknown";
    bool        fault       = false;
    bool        ina219_valid = false;
};

// Result of the 10-minute baseline calibration
struct BaselineResult {
    float offset_celsius = 0.0f;  // delta at idle: (T_ntc - T_tmp117) at rest
    bool  valid          = false;
};