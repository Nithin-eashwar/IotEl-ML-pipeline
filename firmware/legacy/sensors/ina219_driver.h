#pragma once
#include <Arduino.h>
#include "../config.h"

// ═══════════════════════════════════════════════════════════════════
//  ina219_driver.h — INA219 current/power sensor (STUB)
//
//  STATUS: HARDWARE NOT YET ARRIVED
//  This driver returns NaN for all readings until INA219 modules
//  are wired and verified. NILM features delta_I, delta_P are null.
//
//  When hardware arrives:
//   1. Wire INA219 #1 → I2C addr 0x40 (A0,A1 → GND)
//   2. Wire INA219 #2 → I2C addr 0x41 (A0 → 3V3, A1 → GND)
//   3. Add 100nF MLCC decoupling caps per IC VCC
//   4. Add 100µF electrolytic across A4988 VMOT
//   5. Set is_available = true and uncomment full driver code
// ═══════════════════════════════════════════════════════════════════

class INA219Driver {
public:
    bool is_available = false;  // Set true when hardware arrives

    explicit INA219Driver(uint8_t i2c_addr);
    void  begin();
    float read_current_mA();    // Returns NaN if not available
    float read_voltage_V();     // Returns NaN if not available
    float read_power_mW();      // Returns NaN if not available

private:
    uint8_t _addr;
};