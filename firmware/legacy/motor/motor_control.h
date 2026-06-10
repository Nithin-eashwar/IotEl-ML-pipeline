#pragma once
#include <stdint.h>
#include "config.h"

// ═══════════════════════════════════════════════════════════════════
//  motor_control.h — A4988 stepper driver interface (Core 0 only)
// ═══════════════════════════════════════════════════════════════════

void     motor_init();
void     motor_enable();
void     motor_disable();
void     motor_set_direction(bool clockwise);
void     motor_step(uint32_t rpm);          // One step pulse; call in tight loop
uint32_t motor_rpm_to_step_period_us(uint32_t rpm);