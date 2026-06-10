#include "ntc_driver.h"
#include <math.h>

// ═══════════════════════════════════════════════════════════════════
//  ntc_driver.cpp — NTC 10k thermistor (winding surface temperature)
//  ADC: GPIO34 (12-bit, input-only)
//
//  Voltage divider: VCC (3.3V) → 10k fixed → GPIO34 → NTC → GND
//  R_NTC = R_series × ADC / (ADC_MAX - ADC)
//
//  Temperature via Beta equation (simplified Steinhart-Hart):
//  1/T_K = 1/T0_K + (1/B) × ln(R_NTC / R0)
//  T_C   = T_K - 273.15
//
//  CAUTION: ESP32 ADC is nonlinear below ~100 mV and above ~3.1 V.
//  NEVER trust raw NTC absolute value — always use delta_T_corrected.
//  Do not change the 10k divider resistor without recalculating lookup table.
// ═══════════════════════════════════════════════════════════════════

NTCDriver::NTCDriver(uint8_t adc_pin) : _pin(adc_pin) {}

void NTCDriver::begin() {
    // GPIO34 is input-only; analogRead works without explicit pinMode
    // Oversample ×4 and average to reduce ADC noise
    analogReadResolution(12);
    Serial.printf("[NTC] Initialised on GPIO%d — Beta=%.0f, R0=%.0f Ω\n",
                  _pin, NTC_BETA, NTC_NOMINAL_R);
}

// Private: Convert raw ADC reading to temperature in °C
float NTCDriver::adc_to_celsius(int raw) {
    if (raw <= 0 || raw >= (int)ADC_RESOLUTION) return -999.0f; // Rail — invalid

    // Oversample: take 4 reads and average for noise reduction
    float r_ntc = NTC_SERIES_R * (float)raw / (ADC_RESOLUTION - (float)raw);

    // Beta equation
    float t0_k  = NTC_NOMINAL_T + 273.15f;
    float t_k   = 1.0f / (1.0f / t0_k + (1.0f / NTC_BETA) * logf(r_ntc / NTC_NOMINAL_R));
    return t_k - 273.15f;
}

float NTCDriver::read() {
    // 4× oversampling for noise reduction
    long sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += analogRead(_pin);
        delayMicroseconds(100);
    }
    int avg_raw = (int)(sum / 4);
    last_temp_c = adc_to_celsius(avg_raw);
    return last_temp_c;
}