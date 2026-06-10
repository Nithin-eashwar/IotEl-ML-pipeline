#pragma once
#include <Wire.h>
#include "../types.h"

// ═══════════════════════════════════════════════════════════════════
//  tmp117_driver.h — TMP117 precision ambient temperature sensor
//  I2C address: 0x48 | Interface: I2C | Core 1 only
// ═══════════════════════════════════════════════════════════════════

class TMP117Driver {
public:
    float last_temp_c = 0.0f;
    bool  is_online   = false;

    bool  begin();
    float read();       // Returns °C; updates last_temp_c
};