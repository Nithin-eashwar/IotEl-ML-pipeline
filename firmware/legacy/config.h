#pragma once

// ═══════════════════════════════════════════════════════════════════
//  config.h — Ground-truth constants for all firmware modules
//  Matches context.md GPIO map exactly. Do NOT hardcode pins elsewhere.
// ═══════════════════════════════════════════════════════════════════

// ─── GPIO Map ───────────────────────────────────────────────────────
#define PIN_SDA     21          // I2C SDA — TMP117, ADXL345, INA219s
#define PIN_SCL     22          // I2C SCL — TMP117, ADXL345, INA219s
#define PIN_NTC     34          // ADC input-only — NEVER set as OUTPUT
#define PIN_STEP    25          // A4988 STEP — Core 0 only
#define PIN_DIR     26          // A4988 DIR  — Core 0 only
#define PIN_EN      27          // A4988 EN   — Active LOW (pull LOW to enable)

// ─── I2C Addresses ──────────────────────────────────────────────────
#define ADDR_TMP117   0x48      // ADD0 → GND | Live
#define ADDR_ADXL345  0x53      // SDO  → GND | Live
#define ADDR_INA219A  0x40      // A0,A1 → GND  | PENDING — not yet wired
#define ADDR_INA219B  0x41      // A0 → 3V3     | PENDING — not yet wired

// ─── NTC Thermistor ─────────────────────────────────────────────────
#define NTC_NOMINAL_R   10000.0f    // R at 25 °C (Ω)
#define NTC_NOMINAL_T   25.0f       // Reference temp (°C)
#define NTC_BETA        3950.0f     // Beta coefficient
#define NTC_SERIES_R    10000.0f    // Top-arm divider resistor (Ω)
#define ADC_RESOLUTION  4095.0f     // 12-bit ESP32 ADC

// ─── ADXL345 ────────────────────────────────────────────────────────
// ODR set to 800 Hz in hardware; software reads every 2 ms → ~500 Hz effective
#define ACCEL_SAMPLE_MS     2       // Sensor loop tick period (ms)

// ─── Motor Control ──────────────────────────────────────────────────
#define MOTOR_DEFAULT_RPM   60
#define MOTOR_STEPS_PER_REV 200     // NEMA17 @ 1.8°/step full-step
#define MOTOR_MICROSTEP     1       // Full-step (adjust if MS1/MS2/MS3 are set)
#define MOTOR_STEP_PULSE_US 5       // A4988 min STEP high pulse = 1 µs; use 5 µs

// ─── Baseline Calibration ───────────────────────────────────────────
// Set BASELINE_SKIP=1 in build_flags to bypass during development
#ifndef BASELINE_SKIP
  #define BASELINE_SKIP 0
#endif
#define BASELINE_STABILISE_MS  600000UL  // 10 min motor-off stabilisation
#define BASELINE_SAMPLE_MS     30000UL   // 30-second averaging window

// ─── Thermal Thresholds (delta_T_corrected) ─────────────────────────
#define THERMAL_WARN_C    10.0f     // > 10 °C → heavy load warning
#define THERMAL_FAULT_C   15.0f     // > 15 °C → overload/stall → fault

// ─── Wi-Fi ──────────────────────────────────────────────────────────
#define WIFI_SSID       "YOUR_WIFI_SSID"
#define WIFI_PASSWORD   "YOUR_WIFI_PASSWORD"
#define WIFI_TIMEOUT_MS 15000

// ─── MQTT / ThingSpeak (prototype) ──────────────────────────────────
#define MQTT_BROKER     "mqtt3.thingspeak.com"
#define MQTT_PORT       1883
#define MQTT_CLIENT_ID  "ESP32_SmartMeter_001"
#define MQTT_USERNAME   "YOUR_THINGSPEAK_USERNAME"
#define MQTT_PASSWORD   "YOUR_MQTT_API_KEY"         // ThingSpeak MQTT API Key
#define MQTT_CHANNEL_ID "YOUR_CHANNEL_ID"
#define MQTT_TOPIC      "channels/" MQTT_CHANNEL_ID "/publish"
#define MQTT_PUB_MS     2000        // Publish interval (ms)
#define MQTT_KEEPALIVE  60          // Keepalive interval (s)