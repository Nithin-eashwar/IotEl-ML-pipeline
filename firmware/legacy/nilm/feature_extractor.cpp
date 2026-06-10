#include "feature_extractor.h"
#include <math.h>

// ═══════════════════════════════════════════════════════════════════
//  feature_extractor.cpp — NILM feature vector construction
//
//  Feature vector (from context.md):
//  [delta_I, delta_P, delta_Q, accel_RMS, T_winding, delta_T_corrected]
//
//  delta_I and delta_P are NaN while INA219 is pending.
//  delta_T_corrected = (T_ntc - T_ambient) - baseline_offset
// ═══════════════════════════════════════════════════════════════════

// Track previous current sample for delta computation
static float _prev_i_total = 0.0f;
static float _prev_p_total = 0.0f;

FeatureVector build_feature_vector(
    const AccelData& accel,
    float            t_ambient,
    float            t_winding,
    float            i_phaseA_mA,
    float            i_phaseB_mA,
    uint32_t         motor_rpm,
    float            baseline_offset)
{
    FeatureVector fv;

    // ── accel_rms ────────────────────────────────────────────────────
    fv.accel_rms = accel.rms;

    // ── T_winding ─────────────────────────────────────────────────────
    fv.t_winding = t_winding;

    // ── delta_T_corrected ─────────────────────────────────────────────
    // delta_T ≈ 0   → idle baseline
    // delta_T 5-10  → normal load
    // delta_T > 15  → overload / stall (fault)
    fv.delta_T = (t_winding - t_ambient) - baseline_offset;

    // ── INA219-derived features (PENDING) ─────────────────────────────
    bool a_valid = !isnan(i_phaseA_mA);
    bool b_valid = !isnan(i_phaseB_mA);
    fv.ina219_valid = (a_valid && b_valid);

    if (fv.ina219_valid) {
        float i_total = i_phaseA_mA + i_phaseB_mA;
        float v_bus   = 12.0f;  // Nominal motor supply voltage (update per setup)
        float p_total = i_total * v_bus;

        fv.delta_I  = i_total - _prev_i_total;
        fv.delta_P  = p_total - _prev_p_total;
        // delta_Q approximated as 10% of delta_P for inductive load (placeholder)
        fv.delta_Q  = fv.delta_P * 0.10f;

        _prev_i_total = i_total;
        _prev_p_total = p_total;
    } else {
        // INA219 not available — set to NaN to signal downstream consumers
        fv.delta_I = nanf("");
        fv.delta_P = nanf("");
        fv.delta_Q = nanf("");
    }

    return fv;
}