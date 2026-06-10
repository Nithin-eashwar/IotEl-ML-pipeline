#include "nilm_classifier.h"
#include <math.h>

// ═══════════════════════════════════════════════════════════════════
//  nilm_classifier.cpp — 5-class NILM load state detection
//
//  ┌─────────────────────────────────────────────────────────────┐
//  │  PHASE 1: Rule-based heuristic (operational now)            │
//  │  PHASE 2: Edge Impulse TinyML .tflite model (TODO)          │
//  └─────────────────────────────────────────────────────────────┘
//
//  Phase 2 integration steps:
//  1. Collect labelled dataset on edge-impulse.com
//     (record accel + temp data per class, label manually)
//  2. Train 5-class classifier in Edge Impulse Studio
//  3. Export → "Arduino Library" → place in lib/ei-model/
//  4. Uncomment the TinyML block below and remove rule-based code
//  5. Uncomment lib_extra_dirs in platformio.ini
//
//  5 Classes (from context.md):
//   holding    — energised, not moving, low stable current
//   unloaded   — spinning freely, regular step pulses
//   light_load — moderate current rise, vibration amplitude increases
//   heavy_load — high current, thermal rise, vibration harmonics distorted
//   stalled    — current spike, back-EMF collapses, step-FFT peak disappears
// ═══════════════════════════════════════════════════════════════════

// ── Phase 1: Rule-Based Heuristic ───────────────────────────────────────────
// Decision uses accel_rms and delta_T_corrected (both LIVE sensors).
// When INA219 arrives, refine with delta_I thresholds below.
//
//  Thresholds (calibrate against actual motor runs):
//   accel_rms (m/s²):  < 1.0  → stationary | 1–5 → light | > 5 → heavy
//   delta_T (°C):      < 2    → idle        | 2–8 → load  | > 10 → overload

static const char* rule_based_classify(const FeatureVector& fv) {
    float rms     = fv.accel_rms;
    float delta_t = fv.delta_T;

    // ── Stall detection ──────────────────────────────────────────────
    // Stall: motor commanded but vibration collapses.
    // With INA219: also check for current spike (delta_I > threshold).
    if (rms < 1.0f && delta_t > 8.0f) {
        return "stalled";
    }

    // ── Holding (energised, stationary) ─────────────────────────────
    if (rms < 1.0f && delta_t <= 2.0f) {
        return "holding";
    }

    // ── Heavy load ───────────────────────────────────────────────────
    if (rms > 5.0f || delta_t > 10.0f) {
        return "heavy_load";
    }

    // ── Light load ───────────────────────────────────────────────────
    if (rms >= 1.0f && rms <= 5.0f && delta_t >= 2.0f) {
        return "light_load";
    }

    // ── Default: unloaded (spinning freely) ──────────────────────────
    return "unloaded";
}

// ── Phase 2 (TODO): Edge Impulse TinyML Inference ───────────────────────────
// Uncomment and complete once Edge Impulse Arduino library is exported.
/*
#include <your_ei_model_inferencing.h>

static const char* tinyml_classify(const FeatureVector& fv) {
    float input[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = {
        fv.delta_I,
        fv.delta_P,
        fv.delta_Q,
        fv.accel_rms,
        fv.t_winding,
        fv.delta_T
    };

    signal_t signal;
    numpy::signal_from_buffer(input, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);

    ei_impulse_result_t result = {0};
    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);

    if (err != EI_IMPULSE_OK) return "unknown";

    // Find highest confidence class
    size_t best = 0;
    for (size_t i = 1; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > result.classification[best].value)
            best = i;
    }
    return result.classification[best].label;
}
*/

// ── Public API ───────────────────────────────────────────────────────────────
const char* nilm_classify(const FeatureVector& fv) {
    // Switch to tinyml_classify() once Edge Impulse model is deployed
    return rule_based_classify(fv);
}