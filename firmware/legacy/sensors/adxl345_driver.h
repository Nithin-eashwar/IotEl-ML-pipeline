#pragma once
#include <Arduino.h>
#include "../types.h"
#include "../config.h"

// ═══════════════════════════════════════════════════════════════════
//  adxl345_driver.h — ADXL345 vibration accelerometer
//  I2C address: 0x53 | ODR: 800 Hz HW → ~500 Hz SW effective
//  X axis aligned with NEMA17 shaft (do not remount without documenting)
//  Core 1 only
// ═══════════════════════════════════════════════════════════════════

class ADXL345Driver {
public:
    bool is_online = false;

    bool      begin();
    AccelData read();   // Returns x/y/z in m/s² and RMS magnitude
};