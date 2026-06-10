#include "adxl345_driver.h"
#include <Adafruit_ADXL345_U.h>
#include <math.h>

// ═══════════════════════════════════════════════════════════════════
//  adxl345_driver.cpp — ADXL345 vibration accelerometer
//  I2C address: 0x53 (SDO tied to GND)
//  ODR: 800 Hz hardware, software-sampled at 500 Hz via 2 ms tick
//  ADXL345 range: ±16g  (maximise sensitivity for vibration)
//  X axis = shaft axis — do NOT remount without updating this comment
//  Core 1 only
// ═══════════════════════════════════════════════════════════════════

static Adafruit_ADXL345_Unified _accel(12345); // unique sensor ID

bool ADXL345Driver::begin() {
    is_online = _accel.begin();

    if (!is_online) {
        Serial.println("[ADXL345] ERROR — not found. Check SDA(21)/SCL(22) and 4.7k pull-ups.");
        return false;
    }

    // Set ±16g range — best for capturing motor vibration harmonics
    _accel.setRange(ADXL345_RANGE_16_G);

    // Set ODR to 800 Hz (register BW_RATE → 0x0D)
    // The Adafruit library uses setDataRate(); ADXL345_DATARATE_800_HZ = 0x0D
    _accel.setDataRate(ADXL345_DATARATE_800_HZ);

    Serial.printf("[ADXL345] Online at I2C 0x%02X — range=±16g, ODR=800 Hz\n", ADDR_ADXL345);
    return true;
}

AccelData ADXL345Driver::read() {
    AccelData data;

    if (!is_online) return data;

    sensors_event_t event;
    _accel.getEvent(&event);

    data.x     = event.acceleration.x;
    data.y     = event.acceleration.y;
    data.z     = event.acceleration.z;
    data.rms   = sqrtf(data.x * data.x + data.y * data.y + data.z * data.z);
    data.valid = true;

    return data;
}