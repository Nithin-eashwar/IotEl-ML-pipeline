#pragma once
#include "../types.h"

// ═══════════════════════════════════════════════════════════════════
//  feature_extractor.h — Builds the NILM feature vector per sensor tick
// ═══════════════════════════════════════════════════════════════════

FeatureVector build_feature_vector(
    const AccelData& accel,
    float            t_ambient,
    float            t_winding,
    float            i_phaseA_mA,    // NaN if INA219 not available
    float            i_phaseB_mA,    // NaN if INA219 not available
    uint32_t         motor_rpm,
    float            baseline_offset
);