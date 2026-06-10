#pragma once
#include "../types.h"

// ═══════════════════════════════════════════════════════════════════
//  nilm_classifier.h — 5-class NILM load state classifier
//
//  Phases:
//   Phase 1 (current): Rule-based heuristic using accel_rms + delta_T
//   Phase 2 (TODO):    Replace with Edge Impulse TinyML .tflite model
//
//  5 Classes: holding | unloaded | light_load | heavy_load | stalled
// ═══════════════════════════════════════════════════════════════════

// Returns one of: "holding","unloaded","light_load","heavy_load","stalled"
const char* nilm_classify(const FeatureVector& fv);