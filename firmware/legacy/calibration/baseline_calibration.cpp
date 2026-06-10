#include "baseline_calibration.h"
#include <Arduino.h>
#include "../config.h"

// ═══════════════════════════════════════════════════════════════════
//  baseline_calibration.cpp
//
//  Protocol (from context.md):
//   1. Motor OFF, both sensors free-air for BASELINE_STABILISE_MS (10 min)
//   2. Sample (T_ntc - T_tmp117) every 2 s for BASELINE_SAMPLE_MS (30 s)
//   3. Average = baseline_offset
//   4. Motor re-enabled by caller
//
//  If BASELINE_SKIP=1 in build flags, returns offset=0.0 immediately.
// ═══════════════════════════════════════════════════════════════════

BaselineResult baseline_calibrate(
    TMP117Driver& tmp117,
    NTCDriver&    ntc,
    uint32_t      stabilise_ms,
    uint32_t      sample_ms)
{
    BaselineResult result;

#if BASELINE_SKIP
    Serial.println("[CALIB] BASELINE_SKIP=1 — offset forced to 0.0 °C (dev mode)");
    result.offset_celsius = 0.0f;
    result.valid = true;
    return result;
#endif

    // ── Phase 1: Stabilisation wait ────────────────────────────────
    Serial.printf("[CALIB] Stabilising for %lu seconds — keep motor OFF\n",
                  stabilise_ms / 1000);

    uint32_t start = millis();
    while (millis() - start < stabilise_ms) {
        uint32_t remaining = (stabilise_ms - (millis() - start)) / 1000;
        if (remaining % 60 == 0) {
            Serial.printf("[CALIB] %lu min remaining...\n", remaining / 60);
        }
        vTaskDelay(pdMS_TO_TICKS(1000));
    }

    // ── Phase 2: 30-second sampling window ─────────────────────────
    Serial.println("[CALIB] Sampling baseline delta-T for 30 seconds...");

    float    sum        = 0.0f;
    uint32_t n_samples  = 0;
    uint32_t sample_interval_ms = 2000; // One sample every 2 s

    start = millis();
    while (millis() - start < sample_ms) {
        float t_amb  = tmp117.read();
        float t_ntc  = ntc.read();
        float delta  = t_ntc - t_amb;
        sum += delta;
        n_samples++;
        Serial.printf("[CALIB]  T_ntc=%.2f  T_amb=%.2f  delta=%.3f\n",
                      t_ntc, t_amb, delta);
        vTaskDelay(pdMS_TO_TICKS(sample_interval_ms));
    }

    result.offset_celsius = (n_samples > 0) ? (sum / (float)n_samples) : 0.0f;
    result.valid          = (n_samples > 0);

    Serial.printf("[CALIB] Done — baseline_offset = %.3f °C (n=%lu)\n",
                  result.offset_celsius, n_samples);
    return result;
}