#include "motor_control.h"
#include <Arduino.h>

// ═══════════════════════════════════════════════════════════════════
//  motor_control.cpp — A4988 stepper driver implementation
//  ALL functions run on Core 0 (APP_CPU) ONLY.
//  Never call these from Core 1 / sensorTask.
// ═══════════════════════════════════════════════════════════════════

void motor_init() {
    pinMode(PIN_STEP, OUTPUT);
    pinMode(PIN_DIR,  OUTPUT);
    pinMode(PIN_EN,   OUTPUT);

    digitalWrite(PIN_STEP, LOW);
    digitalWrite(PIN_DIR,  HIGH);   // Clockwise by default
    motor_enable();

    Serial.println("[MOTOR] A4988 initialised — GPIO25(STEP), GPIO26(DIR), GPIO27(EN)");
}

void motor_enable() {
    digitalWrite(PIN_EN, LOW);      // A4988 EN is active LOW
}

void motor_disable() {
    digitalWrite(PIN_EN, HIGH);     // Pull HIGH to disable (de-energise coils)
}

void motor_set_direction(bool clockwise) {
    digitalWrite(PIN_DIR, clockwise ? HIGH : LOW);
}

// Returns the total step period in µs for a given RPM.
// step_period = 60,000,000 / (rpm × steps_per_rev × microstep_divisor)
uint32_t motor_rpm_to_step_period_us(uint32_t rpm) {
    if (rpm == 0) return UINT32_MAX;
    return 60000000UL / ((uint32_t)rpm * MOTOR_STEPS_PER_REV * MOTOR_MICROSTEP);
}

// Generate one STEP pulse. Call this in a tight loop inside motorTask.
// Uses delayMicroseconds() — only accurate because this is the sole task on Core 0.
void motor_step(uint32_t rpm) {
    uint32_t period_us = motor_rpm_to_step_period_us(rpm);

    digitalWrite(PIN_STEP, HIGH);
    delayMicroseconds(MOTOR_STEP_PULSE_US);
    digitalWrite(PIN_STEP, LOW);

    // Remaining delay to complete the step period
    if (period_us > MOTOR_STEP_PULSE_US) {
        delayMicroseconds(period_us - MOTOR_STEP_PULSE_US);
    }
}