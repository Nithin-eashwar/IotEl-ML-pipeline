#pragma once
#include <Arduino.h>
#include "../config.h"

// ═══════════════════════════════════════════════════════════════════
//  ntc_driver.h — NTC 10k thermistor winding temperature reader
//  ADC pin: GPIO34 (input-only) | Beta equation | Core 1 only
//
//  CAUTION: Use delta_T_corrected for all logic.
//  Raw NTC absolute value has ±2°C accuracy + Beta tolerance ±1-3%.
//  ESP32 ADC is linear only ~100 mV – 3.1 V.
// ═══════════════════════════════════════════════════════════════════

class NTCDriver {
public:
    float last_temp_c = 0.0f;

    explicit NTCDriver(uint8_t adc_pin);
    void  begin();
    float read();   // Returns corrected °C via Beta equation

private:
    uint8_t _pin;
    float   adc_to_celsius(int raw);
};