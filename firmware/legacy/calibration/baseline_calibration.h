#pragma once
#include "../types.h"
#include "../sensors/tmp117_driver.h"
#include "../sensors/ntc_driver.h"

// ═══════════════════════════════════════════════════════════════════
//  baseline_calibration.h
//  Measures (T_ntc - T_tmp117) offset at idle startup.
//  Motor MUST be off and both sensors stabilised for BASELINE_STABILISE_MS.
//  Result stored in BaselineResult.offset_celsius.
// ═══════════════════════════════════════════════════════════════════

BaselineResult baseline_calibrate(
    TMP117Driver& tmp117,
    NTCDriver&    ntc,
    uint32_t      stabilise_ms,
    uint32_t      sample_ms
);