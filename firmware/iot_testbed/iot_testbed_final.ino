/**
 * IoT Smart Metering Testbed - Data Collection Firmware
 * Architecture: Dual-Core FreeRTOS
 * - Core 0: High-priority constant speed motor control (150 RPM).
 * - Core 1: Strict 100Hz I2C/ADC sensor polling and CSV stream.
 *
 * NOTE: INA219 B has been removed. INA219 A now measures total 12V supply current.
 * CSV format remains unchanged (13 columns) to preserve ML pipeline compatibility.
 *
 * Capture CSV on Windows (PowerShell) — run AFTER closing Serial Monitor:
 *   $port = New-Object System.IO.Ports.SerialPort COM5,115200
 *   $port.Open()
 *   $port.ReadExisting() | Out-Null
 *   $file = "class1_unloaded.csv"
 *   $count = 0
 *   while ($true) {
 *     try {
 *       $line = $port.ReadLine().Trim()
 *       if ($line -notmatch "^#" -and $line.Length -gt 0) {
 *         Add-Content -Path $file -Value $line
 *         $count++
 *         if ($count % 100 -eq 0) { Write-Host "Rows: $count" }
 *       }
 *     } catch { Write-Host "Stopped at $count rows."; break }
 *   }
 *   $port.Close()
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include <SparkFun_TMP117.h>
#include <Adafruit_INA219.h>
#include "esp_task_wdt.h"

// ==========================================
// 1. ML CONFIGURATION (TEAMMATE EDITS THIS)
// ==========================================
// 0: idle_holding, 1: unloaded, 2: light_load, 3: heavy_load, 4: stalled
#define LOAD_CLASS 1

// ==========================================
// 2. HARDWARE & MOTOR PROFILES
// ==========================================
#define CONSTANT_RPM 150.0  // Locked to 150 RPM for clean ML feature extraction

#define PIN_STEP 25
#define PIN_DIR  26
#define PIN_EN   27
#define PIN_NTC  34

// ==========================================
// 3. GLOBAL OBJECTS & STATE
// ==========================================
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);
TMP117          tmp117;
Adafruit_INA219 ina219_A(0x40);  // Measures TOTAL 12V supply current
// ina219_B physically removed from circuit

TaskHandle_t motorTaskHandle;
TaskHandle_t sensorTaskHandle;

// Shared state for slow sensors (2Hz update rate)
volatile float cached_tmp117_c = 0.0;
volatile float cached_ntc_c    = 0.0;

// ==========================================
// 4. UTILITY FUNCTIONS
// ==========================================
/**
 * Reads NTC thermistor using the Beta equation.
 * Includes divide-by-zero protection for hardware shorts.
 */
float readNTC() {
  int raw = analogRead(PIN_NTC);
  if (raw <= 0 || raw >= 4095) return 0.0;

  float ratio       = (float)raw / 4095.0;
  float r_ntc       = 10000.0 * ratio / (1.0 - ratio);
  float temperature = r_ntc / 10000.0;
  temperature       = log(temperature);
  temperature      /= 3950.0;
  temperature      += 1.0 / (25.0 + 273.15);
  temperature       = 1.0 / temperature;
  temperature      -= 273.15;
  return temperature;
}

// ==========================================
// 5. FREERTOS TASKS
// ==========================================

/**
 * Task: Motor Control (Pinned to Core 0)
 * Drives the stepper driver at a constant 150 RPM.
 * Intentionally never yields — WDT disabled.
 */
void motorTask(void *pvParameters) {
  disableCore0WDT();

  // Explicit float cast for precise microsecond delay calculation
  // 150 RPM -> delay_us = 150000 / 150 = 1000 us per half-pulse
  unsigned long delay_us = (unsigned long)(150000.0 / CONSTANT_RPM);

  for (;;) {
    digitalWrite(PIN_STEP, HIGH);
    delayMicroseconds(delay_us);
    digitalWrite(PIN_STEP, LOW);
    delayMicroseconds(delay_us);
  }
}

/**
 * Task: Data Pipeline (Pinned to Core 1)
 * Polls sensors and writes CSV data strictly at 100Hz (10ms ticks).
 */
void sensorTask(void *pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xFrequency = pdMS_TO_TICKS(10);  // 100Hz loop
  uint8_t slow_sensor_counter = 0;

  for (;;) {
    vTaskDelayUntil(&xLastWakeTime, xFrequency);

    unsigned long ts = millis();

    // Fast Sensor Reads (100Hz)
    sensors_event_t event;
    accel.getEvent(&event);
    float ax    = event.acceleration.x;
    float ay    = event.acceleration.y;
    float az    = event.acceleration.z;
    float a_rms = sqrt((ax * ax) + (ay * ay) + (az * az));

    // INA219 A — total system current and power
    float ina_a_ma = ina219_A.getCurrent_mA();
    float ina_a_mw = ina219_A.getPower_mW();

    // INA219 B removed — hardcoded to 0.00 to preserve 13-column schema
    float ina_b_ma = 0.00;
    float ina_b_mw = 0.00;

    // Slow Sensor Reads (2Hz — every 50 ticks)
    if (slow_sensor_counter >= 50) {
      if (tmp117.dataReady()) {
        cached_tmp117_c = tmp117.readTempC();
      }
      cached_ntc_c = readNTC();
      slow_sensor_counter = 0;
    } else {
      slow_sensor_counter++;
    }

    // CSV Serial Output
    Serial.print(ts);                 Serial.print(",");
    Serial.print(ax, 3);              Serial.print(",");
    Serial.print(ay, 3);              Serial.print(",");
    Serial.print(az, 3);              Serial.print(",");
    Serial.print(a_rms, 3);           Serial.print(",");
    Serial.print(cached_tmp117_c, 2); Serial.print(",");
    Serial.print(cached_ntc_c, 2);    Serial.print(",");
    Serial.print(ina_a_ma, 2);        Serial.print(",");
    Serial.print(ina_a_mw, 2);        Serial.print(",");
    Serial.print(ina_b_ma, 2);        Serial.print(",");  // 0.00 — keeps 13 columns
    Serial.print(ina_b_mw, 2);        Serial.print(",");  // 0.00 — keeps 13 columns
    Serial.print(CONSTANT_RPM, 1);    Serial.print(",");
    Serial.println(LOAD_CLASS);
  }
}

// ==========================================
// 6. MAIN SETUP
// ==========================================
void setup() {
  Serial.begin(115200);

  // 2-second timeout for headless (no USB) operation
  unsigned long start_wait = millis();
  while (!Serial && (millis() - start_wait < 2000)) {
    delay(10);
  }

  pinMode(PIN_STEP, OUTPUT);
  pinMode(PIN_DIR,  OUTPUT);
  pinMode(PIN_EN,   OUTPUT);
  digitalWrite(PIN_DIR, HIGH);  // CW rotation
  digitalWrite(PIN_EN,  LOW);   // Enable A4988 driver

  analogReadResolution(12);

  Wire.begin();
  Wire.setClock(400000);

  Serial.println("# === Sensor Init ===");
  bool hardware_ok = true;

  if (!accel.begin()) {
    Serial.println("# ADXL345  : FAIL");
    hardware_ok = false;
  } else {
    accel.setRange(ADXL345_RANGE_4_G);
    accel.setDataRate(ADXL345_DATARATE_100_HZ);
    Serial.println("# ADXL345  : OK  range=+-4g  rate=100Hz  addr=0x53");
  }

  if (!tmp117.begin()) {
    Serial.println("# TMP117   : FAIL");
    hardware_ok = false;
  } else {
    Serial.println("# TMP117   : OK  addr=0x48");
  }

  if (!ina219_A.begin()) {
    Serial.println("# INA219 A : FAIL");
    hardware_ok = false;
  } else {
    ina219_A.setCalibration_32V_2A();
    Serial.println("# INA219 A : OK  addr=0x40  cal=32V/2A  Total Load");
  }

  // Hardware guard — halt if any sensor failed init
  // Prevents tasks launching with dead sensors and silent bad data
  if (!hardware_ok) {
    Serial.println("# CRITICAL ERROR: One or more sensors failed to initialize.");
    Serial.println("# System Halted. Check wiring and pull-ups.");
    while (true) {
      delay(1000);
    }
  }

  // Pre-flight temperature reads — prevents zeroed-out first 50 rows
  cached_ntc_c = readNTC();
  delay(20);  // Guarantee TMP117 first conversion completes
  cached_tmp117_c = tmp117.readTempC();

  Serial.print("# Recording class: ");
  Serial.println(LOAD_CLASS);

  Serial.println("timestamp_ms,accel_x,accel_y,accel_z,accel_rms,tmp117_c,ntc_c,ina_a_ma,ina_a_mw,ina_b_ma,ina_b_mw,motor_rpm,load_class");

  xTaskCreatePinnedToCore(
    motorTask, "MotorTask", 8192, NULL, 2, &motorTaskHandle, 0
  );

  xTaskCreatePinnedToCore(
    sensorTask, "SensorTask", 16384, NULL, 1, &sensorTaskHandle, 1
  );
}

void loop() {
  // Empty — FreeRTOS tasks handle all execution
  vTaskDelete(NULL);
}
