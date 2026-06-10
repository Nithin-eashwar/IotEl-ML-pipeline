/**
 * @file main.cpp
 * @brief IoT Smart Metering on Stepper Motor Testbed — Main Entry Point
 *
 * ESP32-WROOM-30 | FreeRTOS | Dual-Core | NILM via TinyML
 *
 * ┌──────────────────────────────────────────────────────────────────────┐
 * │  Core 0 (APP_CPU) → motorTask   : STEP/DIR/EN pulse generation       │
 * │  Core 1 (PRO_CPU) → sensorTask  : I2C sensors + NILM + MQTT publish  │
 * └──────────────────────────────────────────────────────────────────────┘
 *
 * RULE: All I2C calls live exclusively on Core 1 (sensorTask).
 *       If a future Core 0 task needs I2C, wrap every Wire call
 *       with the i2c_mutex semaphore.
 *
 * Firmware milestones (context.md sequence):
 *  ✅ 1  I2C scanner (test/i2c_scanner/)
 *  ✅ 2  TMP117 driver
 *  ✅ 3  ADXL345 driver
 *  ✅ 4  NTC ADC reader
 *  ✅ 5  A4988 motor control
 *  ✅ 6  Dual-core integration
 *  ✅ 7  Baseline calibration
 *  ✅ 8  MQTT publisher (ThingSpeak)
 *  ⬜ 9  INA219 integration (hardware pending)
 *  ⬜ 10 Edge Impulse dataset collection + training
 *  ⬜ 11 NILM TinyML inference loop
 *  ⬜ 12 AWS IoT Core migration
 */

#include <Arduino.h>
#include <Wire.h>
#include "config.h"
#include "types.h"

#include "motor/motor_control.h"
#include "sensors/tmp117_driver.h"
#include "sensors/adxl345_driver.h"
#include "sensors/ntc_driver.h"
#include "sensors/ina219_driver.h"
#include "calibration/baseline_calibration.h"
#include "nilm/feature_extractor.h"
#include "nilm/nilm_classifier.h"
#include "comms/mqtt_publisher.h"

// ─── Task Handles ────────────────────────────────────────────────────────────
static TaskHandle_t motorTaskHandle  = NULL;
static TaskHandle_t sensorTaskHandle = NULL;

// ─── Shared State ────────────────────────────────────────────────────────────
// motor_running written by Core 1 (fault/calib), read by Core 0.
// motor_rpm written by Core 1 (future speed control), read by Core 0.
// Use volatile — single-writer, single-reader, no struct alignment issue.
volatile uint32_t g_motor_rpm     = MOTOR_DEFAULT_RPM;
volatile bool     g_motor_running = false; // false until calibration complete

// ─── FreeRTOS mutex (reserved for future multi-task I2C use) ─────────────────
SemaphoreHandle_t i2c_mutex = NULL;

// ═══════════════════════════════════════════════════════════════════
//  Core 0 Task — Motor Control (motorTask)
//  Sole task on Core 0. Uses delayMicroseconds() for step timing.
//  Never touch I2C here.
// ═══════════════════════════════════════════════════════════════════
static void motorTask(void* pvParameters) {
    motor_init();

    for (;;) {
        if (g_motor_running) {
            motor_enable();
            motor_step(g_motor_rpm); // Blocking µs-level pulse
        } else {
            motor_disable();
            vTaskDelay(pdMS_TO_TICKS(10)); // Yield while stopped
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Core 1 Task — Sensors + NILM + MQTT (sensorTask)
//  All I2C calls here. Uses vTaskDelayUntil() for drift-free 2 ms tick.
// ═══════════════════════════════════════════════════════════════════
static void sensorTask(void* pvParameters) {

    // ── I2C Bus ────────────────────────────────────────────────────────────
    Wire.begin(PIN_SDA, PIN_SCL);
    Wire.setClock(400000); // 400 kHz Fast Mode

    // ── Sensor Init ───────────────────────────────────────────────────────
    TMP117Driver  tmp117;
    ADXL345Driver adxl345;
    NTCDriver     ntc(PIN_NTC);
    INA219Driver  ina219a(ADDR_INA219A);
    INA219Driver  ina219b(ADDR_INA219B);

    tmp117.begin();
    adxl345.begin();
    ntc.begin();
    ina219a.begin(); // Silently marks unavailable — PENDING hardware
    ina219b.begin();

    // ── Baseline Calibration ──────────────────────────────────────────────
    Serial.println("[MAIN] Starting baseline calibration — motor disabled.");
    g_motor_running = false;

    BaselineResult baseline = baseline_calibrate(
        tmp117, ntc,
        BASELINE_STABILISE_MS,
        BASELINE_SAMPLE_MS
    );

    g_motor_running = true;
    Serial.println("[MAIN] Motor ENABLED — entering sensor loop.");

    // ── MQTT Init ─────────────────────────────────────────────────────────
    mqtt_init();

    // ── Main Sensor Loop (drift-free via vTaskDelayUntil) ─────────────────
    TickType_t xLastWake    = xTaskGetTickCount();
    uint32_t   slowCounter  = 0;   // Increments every 2 ms tick
    uint32_t   publishTimer = 0;   // Accumulated ms since last publish

    float t_ambient = 25.0f; // Initialised to room temp; updated every 500 ms
    float t_winding = 25.0f;

    for (;;) {
        // ── 1. ADXL345: read every tick (2 ms → ~500 Hz effective) ─────────
        AccelData accel = adxl345.read();

        // ── 2. TMP117 + NTC: read every 500 ms (250 ticks × 2 ms) ──────────
        if (slowCounter % 250 == 0) {
            t_ambient = tmp117.read();
            t_winding = ntc.read();
        }

        // ── 3. INA219: read every tick (null if not available) ───────────────
        float i_phaseA = ina219a.read_current_mA();
        float i_phaseB = ina219b.read_current_mA();

        // ── 4. Build NILM feature vector ─────────────────────────────────────
        FeatureVector fv = build_feature_vector(
            accel, t_ambient, t_winding,
            i_phaseA, i_phaseB,
            g_motor_rpm,
            baseline.offset_celsius
        );

        // ── 5. Thermal protection ─────────────────────────────────────────────
        bool fault = false;
        if (fv.delta_T > THERMAL_FAULT_C) {
            fault           = true;
            g_motor_running = false;
            Serial.printf("[FAULT] Thermal overload! delta_T=%.1f°C > %.0f°C — Motor DISABLED\n",
                          fv.delta_T, THERMAL_FAULT_C);
        } else if (!g_motor_running) {
            // Re-enable motor if fault has cleared (thermal recovery)
            if (fv.delta_T < THERMAL_WARN_C) {
                g_motor_running = true;
                Serial.println("[RECOVERY] Thermal OK — Motor re-enabled");
            }
        }

        // ── 6. NILM Inference ─────────────────────────────────────────────────
        const char* load_class = nilm_classify(fv);

        // ── 7. MQTT publish every MQTT_PUB_MS ────────────────────────────────
        publishTimer += ACCEL_SAMPLE_MS;
        if (publishTimer >= MQTT_PUB_MS) {
            publishTimer = 0;

            TelemetryPayload payload;
            payload.timestamp_ms  = (uint64_t)(esp_timer_get_time() / 1000ULL);
            payload.accel         = accel;
            payload.t_ambient     = t_ambient;
            payload.t_winding     = t_winding;
            payload.delta_t       = fv.delta_T;
            payload.i_phaseA      = i_phaseA;
            payload.i_phaseB      = i_phaseB;
            payload.motor_rpm     = g_motor_rpm;
            payload.load_class    = load_class;
            payload.fault         = fault;
            payload.ina219_valid  = fv.ina219_valid;

            mqtt_publish(payload);
            mqtt_loop();
        }

        slowCounter++;

        // ── 8. Drift-free 2 ms wait ───────────────────────────────────────────
        vTaskDelayUntil(&xLastWake, pdMS_TO_TICKS(ACCEL_SAMPLE_MS));
    }
}

// ═══════════════════════════════════════════════════════════════════
//  setup() — Entry point. Launches both tasks. loop() is empty.
//  Stack sizes: motorTask=2048 (GPIO only), sensorTask=8192 (JSON+MQTT)
// ═══════════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n╔══════════════════════════════════════════════╗");
    Serial.println("║  IoT Smart Metering — Stepper Motor Testbed  ║");
    Serial.println("║  ESP32-WROOM  |  FreeRTOS  |  NILM v1.0      ║");
    Serial.println("╚══════════════════════════════════════════════╝\n");

    i2c_mutex = xSemaphoreCreateMutex();
    configASSERT(i2c_mutex != NULL);

    xTaskCreatePinnedToCore(motorTask,  "Motor",   2048, NULL, 2, &motorTaskHandle,  0);
    xTaskCreatePinnedToCore(sensorTask, "Sensors", 8192, NULL, 1, &sensorTaskHandle, 1);

    vTaskDelete(NULL); // Kill Arduino loop task — we use FreeRTOS tasks only
}

void loop() {} // Intentionally empty — do not add code here