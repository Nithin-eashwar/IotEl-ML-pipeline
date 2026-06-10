#include "tmp117_driver.h"
#include <SparkFun_TMP117.h>

// ═══════════════════════════════════════════════════════════════════
//  tmp117_driver.cpp — TMP117 precision temperature sensor (±0.1 °C)
//  I2C address: 0x48 (ADD0 tied to GND)
//  Core 1 only — do not call from motorTask
// ═══════════════════════════════════════════════════════════════════

static TMP117 _sensor;

bool TMP117Driver::begin() {
    is_online = _sensor.begin();
    if (is_online) {
        Serial.printf("[TMP117] Online at I2C 0x%02X\n", ADDR_TMP117);
    } else {
        Serial.println("[TMP117] ERROR — not found. Check SDA(21)/SCL(22) and 4.7k pull-ups.");
    }
    return is_online;
}

float TMP117Driver::read() {
    if (!is_online) return last_temp_c;

    if (_sensor.dataReady()) {
        last_temp_c = _sensor.readTempC();
    }
    return last_temp_c;
}