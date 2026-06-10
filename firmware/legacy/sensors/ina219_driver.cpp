#include "ina219_driver.h"
#include <math.h>

// ═══════════════════════════════════════════════════════════════════
//  ina219_driver.cpp — INA219 current/power sensor STUB
//
//  HARDWARE NOT YET ARRIVED — all reads return NaN.
//  is_available is forced false.
//
//  TODO when hardware arrives:
//  1. Uncomment Adafruit_INA219 code below
//  2. Set is_available = true in begin() if init succeeds
//  3. Return actual readings in read_* functions
// ═══════════════════════════════════════════════════════════════════

// #include <Adafruit_INA219.h>
// static Adafruit_INA219 _ina_a(ADDR_INA219A);
// static Adafruit_INA219 _ina_b(ADDR_INA219B);

INA219Driver::INA219Driver(uint8_t i2c_addr) : _addr(i2c_addr) {}

void INA219Driver::begin() {
    // Stub — do nothing until hardware arrives
    is_available = false;
    Serial.printf("[INA219] 0x%02X — PENDING (hardware not yet arrived)\n", _addr);
}

float INA219Driver::read_current_mA() {
    if (!is_available) return nanf("");
    // TODO: return _ina.getCurrent_mA();
    return nanf("");
}

float INA219Driver::read_voltage_V() {
    if (!is_available) return nanf("");
    // TODO: return _ina.getBusVoltage_V();
    return nanf("");
}

float INA219Driver::read_power_mW() {
    if (!is_available) return nanf("");
    // TODO: return _ina.getPower_mW();
    return nanf("");
}