/*
 * ESP32 FreeRTOS Firmware — NEMA 17 Motor + Sensor Telemetry + Azure IoT Hub
 * ===========================================================================
 * Architecture:  3 FreeRTOS tasks + 1 queue + RMT stepper peripheral
 *
 *   vMotorTask   — RMT-driven A4988 step pulses (µs resolution), tracks RPM
 *   vSensorTask  — reads TMP117/INA219/ADXL345, rolling vibration RMS, ml_infer()
 *   vAzureTask   — MQTT+TLS to Azure IoT Hub, SAS token auth, 5‑sec publish
 *   sas_refresh  — background HMAC‑SHA256 SAS token generator (55‑min cycle)
 *
 * Hardware:
 *   NEMA 17 stepper + A4988 driver (STEP/DIR/ENABLE + RMT channel)
 *   TMP117  (I2C 0x48) — ±0.1 °C temperature
 *   INA219A (I2C 0x40) + INA219B (I2C 0x41) — phase current
 *   ADXL345 (I2C 0x53) — 3‑axis accelerometer (vibration RMS)
 *
 * Credentials — paste from dashboard after device registration:
 *   IOT_HUB_HOST  = "your-hub.azure-devices.net"
 *   DEVICE_ID     = "nema17-bay3-abc123"
 *   PRIMARY_KEY  = "base64‑encoded‑device‑key"
 *
 * Build with ESP‑IDF >= 6.0:
 *   idf.py create-project motor_telemetry
 *   cp esp32_telemetry.c nilm_model.* motor_telemetry/main/
 *   idf.py set-target esp32
 *   idf.py build && idf.py flash monitor
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <stdatomic.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_sntp.h"
#include "nvs_flash.h"

#include "esp_task_wdt.h"

#include "driver/i2c_master.h"
#include "driver/gpio.h"
#include "driver/rmt_tx.h"
#include "esp_adc/adc_oneshot.h"

#include "mqtt_client.h"

#include "esp_crt_bundle.h"

#include "mbedtls/base64.h"
#include "psa/crypto.h"

#ifndef __has_include
#define __has_include(x) 0
#endif

#if __has_include("nilm_model.h")
#include "nilm_model.h"
#define NILM_MODEL_AVAILABLE 1
#else
#define NILM_MODEL_AVAILABLE 0
#endif

/* ================================================================== */
/*  CONFIG  —  injected via CMake from host environment              */
/* ================================================================== */

/* ---------- Motor -------------------------------------------------- */

#define MOTOR_TARGET_RPM    150.0f
#define MOTOR_STEPS_PER_REV 200             /* NEMA 17  1.8 °/step */
#define MICROSTEP_MODE      1               /* MS1/MS2/MS3 tied to GND */
#define STEP_PIN            GPIO_NUM_25     /* RMT TX channel */
#define DIR_PIN             GPIO_NUM_26
#define ENABLE_PIN          GPIO_NUM_27     /* A4988 ENABLE  active-low */

/* Minimum step period that guarantees A4988 1 µs pulse within 1 MHz RMT
 * resolution:  2 µs high + 1 µs low  =  3 µs  (~333 k steps/s  max) */
#define MOTOR_MIN_STEP_US   3

/* ---------- Sensor I2C --------------------------------------------- */

#define I2C_MASTER_SCL      GPIO_NUM_22
#define I2C_MASTER_SDA      GPIO_NUM_21
#define I2C_MASTER_FREQ     400000
#define I2C_MASTER_PORT     I2C_NUM_0

#define ADXL345_ADDR        0x53
#define TMP117_ADDR         0x48
#define INA219_A_ADDR       0x40
#define INA219_B_ADDR       0x41

/* ---------- NTC ADC ------------------------------------------------ */

#define NTC_ADC_CHANNEL     ADC_CHANNEL_6   /* GPIO34 */
#define NTC_NOMINAL_R       10000.0f
#define NTC_NOMINAL_T_C     25.0f
#define NTC_BETA            3950.0f
#define NTC_SERIES_R        10000.0f
#define ADC_MAX_COUNTS      4095.0f

/* ---------- Timing ------------------------------------------------- */

#define SENSOR_PERIOD_MS    10              /* 100 Hz, matches ML training data */
#define AZURE_PUBLISH_MS    5000
#define SAS_TOKEN_TTL_SEC   3600
#define SAS_REFRESH_SEC     3300            /* 55 min */
#define SAS_TOKEN_BUF_LEN   512

/* ---------- Reconnection & Watchdog --------------------------------- */

#define MQTT_MAX_FAILS      5
#define STUCK_LIMIT         10
#define BACKOFF_MAX_MS      60000

/* NILM model windowing must match ml/nilm_pipeline.py. */
#define NILM_WINDOW_SIZE    128
#define NILM_STEP_SIZE      64
#define NILM_SAMPLE_RATE_HZ 100.0f
#define NILM_DEAD_CURRENT_MA 2.0f

/* ================================================================== */
/*  DATA STRUCTURES                                                    */
/* ================================================================== */

typedef struct {
    float rpm;
    float temperature;         /* Motor/contact temperature from NTC */
    float ambient_temperature; /* Ambient temperature from TMP117 */
    float vibration_rms;
    float current;
    float phase_a_current;
    float phase_b_current;
} sensor_features_t;

typedef struct {
    const char *status;         /* "Active" | "warning" | "critical" | "idle" */
    const char *status_message;
    float       confidence;     /* 0.0–1.0  (placeholder for ML model output) */
} ml_result_t;

typedef struct {
    sensor_features_t features;
    ml_result_t       ml;
    char              timestamp[32];
} telemetry_packet_t;

typedef struct {
    float accel_x;
    float accel_y;
    float accel_z;
    float accel_rms;
    float tmp117_c;
    float ntc_c;
    float ina_a_ma;
    float ina_a_mw;
} nilm_sample_t;

typedef struct {
    nilm_sample_t samples[NILM_WINDOW_SIZE];
    size_t        next;
    size_t        count;
    size_t        since_infer;
    ml_result_t   last_result;
} nilm_window_t;

/* ================================================================== */
/*  GLOBALS                                                            */
/* ================================================================== */

static const char *TAG = "MOTOR";

/* ---- SAS token (single buffer shared between tasks, mutex‑protected) */
static char             g_sas_token[SAS_TOKEN_BUF_LEN];
static bool             g_sas_valid = false;
static SemaphoreHandle_t g_sas_mutex = NULL;

/* ---- Effective RPM (protected by a spinlock — atomic_float is not C11) */
static float             g_effective_rpm     = 0.0f;
static portMUX_TYPE      g_rpm_mux           = portMUX_INITIALIZER_UNLOCKED;

/* ---- Sensor state (mutex‑protected) */
static sensor_features_t g_sensors   = {0};
static ml_result_t       g_ml_result = {"Active", "Normal operation", 0.0f};
static SemaphoreHandle_t g_sensor_mutex = NULL;
static nilm_window_t     g_nilm_window = {
    .last_result = {"Active", "Collecting NILM window", 0.0f},
};

/* ---- RMT stepper channel */
static rmt_channel_handle_t    g_step_chan   = NULL;
static rmt_encoder_handle_t    g_step_enc    = NULL;
static rmt_symbol_word_t       g_step_symbol = {0};
static uint32_t                g_steps_per_rev = 0;

/* ---- MQTT */
static esp_mqtt_client_handle_t g_mqtt_client = NULL;

/* ---- Telemetry queue */
static QueueHandle_t g_telemetry_queue = NULL;

/* ---- NTP sync flag */
static volatile bool g_ntp_synced = false;

/* ---- Epoch minimum for SAS generation */
#define MIN_VALID_EPOCH 1704067200ULL  /* Jan 1 2024 */

/* ---- I2C master bus handle (IDF 6 new driver) */
static i2c_master_bus_handle_t g_i2c_bus = NULL;

/* ---- ADC oneshot handle (IDF 6 new driver) */
static adc_oneshot_unit_handle_t g_adc_handle = NULL;

/* ---- Reconnection & Watchdog state */
static uint32_t          s_wifi_retry    = 0;
static uint32_t          s_mqtt_retry    = 0;
static uint8_t           s_pub_fails     = 0;
static volatile uint8_t  s_stuck_counter = 0;
static volatile bool     g_connected     = false;

/* ================================================================== */
/*  ML INFERENCE HOOK                                                  */
/* ================================================================== */

/*
 * Default — threshold‑based anomaly detection.
 * Override this weak function with your own model.
 *
 * Expected statuses (matching dashboard telemetry_live schema):
 *   "Active"   — all nominal
 *   "warning"  — parameter approaching threshold
 *   "critical" — threshold exceeded
 *   "idle"     — motor stopped
 *
 *  When replacing with TFLite Micro:
 *   • Increase vSensorTask stack from 16384 to 32768+
 *   • Allocate the tensor arena as a static buffer, not on the stack
 */
__attribute__((weak)) ml_result_t ml_infer(sensor_features_t f)
{
    float vib  = f.vibration_rms;
    float temp = f.temperature;
    float cur  = f.current;
    float rpm  = f.rpm;

    if (rpm < 1.0f) {
        return (ml_result_t){"idle", "Motor stopped", 1.0f};
    }
    if (temp > 80.0f || vib > 4.0f) {
        return (ml_result_t){"critical", "Critical threshold exceeded", 0.95f};
    }
    if (temp > 60.0f || vib > 3.0f || cur > 2.5f) {
        return (ml_result_t){"warning", "Parameter approaching threshold", 0.85f};
    }
    return (ml_result_t){"Active", "Normal operation", 0.98f};
}

static int float_compare(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static float percentile(float values[NILM_WINDOW_SIZE], float q)
{
    qsort(values, NILM_WINDOW_SIZE, sizeof(float), float_compare);
    float pos = (NILM_WINDOW_SIZE - 1) * q;
    int lower = (int)floorf(pos);
    int upper = (int)ceilf(pos);
    float frac = pos - (float)lower;
    return values[lower] * (1.0f - frac) + values[upper] * frac;
}

static void nilm_window_add(nilm_window_t *window, nilm_sample_t sample)
{
    window->samples[window->next] = sample;
    window->next = (window->next + 1) % NILM_WINDOW_SIZE;
    if (window->count < NILM_WINDOW_SIZE) window->count++;
    if (window->since_infer < NILM_STEP_SIZE) window->since_infer++;
}

static const nilm_sample_t *nilm_window_sample(const nilm_window_t *window,
                                               size_t index)
{
    size_t start = window->count == NILM_WINDOW_SIZE ? window->next : 0;
    return &window->samples[(start + index) % NILM_WINDOW_SIZE];
}

static float sample_value(const nilm_sample_t *sample, const char *series)
{
    if (strcmp(series, "cur") == 0) return sample->ina_a_ma;
    if (strcmp(series, "pwr") == 0) return sample->ina_a_mw;
    if (strcmp(series, "accel_x") == 0) return sample->accel_x;
    if (strcmp(series, "accel_y") == 0) return sample->accel_y;
    if (strcmp(series, "accel_z") == 0) return sample->accel_z;
    if (strcmp(series, "accel_rms") == 0) return sample->accel_rms;
    if (strcmp(series, "ntc") == 0) return sample->ntc_c;
    if (strcmp(series, "tmp117") == 0) return sample->tmp117_c;
    return 0.0f;
}

static void collect_series(const nilm_window_t *window, const char *series,
                           float values[NILM_WINDOW_SIZE])
{
    for (size_t i = 0; i < NILM_WINDOW_SIZE; ++i) {
        values[i] = sample_value(nilm_window_sample(window, i), series);
    }
}

static float mean_of(const float values[NILM_WINDOW_SIZE])
{
    float sum = 0.0f;
    for (size_t i = 0; i < NILM_WINDOW_SIZE; ++i) sum += values[i];
    return sum / (float)NILM_WINDOW_SIZE;
}

static float std_of(const float values[NILM_WINDOW_SIZE], float mean)
{
    float sum = 0.0f;
    for (size_t i = 0; i < NILM_WINDOW_SIZE; ++i) {
        float d = values[i] - mean;
        sum += d * d;
    }
    return sqrtf(sum / (float)NILM_WINDOW_SIZE);
}

static float min_of(const float values[NILM_WINDOW_SIZE])
{
    float value = values[0];
    for (size_t i = 1; i < NILM_WINDOW_SIZE; ++i) {
        if (values[i] < value) value = values[i];
    }
    return value;
}

static float max_of(const float values[NILM_WINDOW_SIZE])
{
    float value = values[0];
    for (size_t i = 1; i < NILM_WINDOW_SIZE; ++i) {
        if (values[i] > value) value = values[i];
    }
    return value;
}

static float rms_of(const float values[NILM_WINDOW_SIZE])
{
    float sum = 0.0f;
    for (size_t i = 0; i < NILM_WINDOW_SIZE; ++i) sum += values[i] * values[i];
    return sqrtf(sum / (float)NILM_WINDOW_SIZE);
}

static float fft_bin_energy(const float values[NILM_WINDOW_SIZE], int bin)
{
    const float pi = 3.14159265358979323846f;
    float mean = mean_of(values);
    float lower = (NILM_SAMPLE_RATE_HZ / 2.0f) * ((float)bin / 8.0f);
    float upper = (NILM_SAMPLE_RATE_HZ / 2.0f) * ((float)(bin + 1) / 8.0f);
    float energy = 0.0f;

    for (int k = 0; k <= NILM_WINDOW_SIZE / 2; ++k) {
        float freq = ((float)k * NILM_SAMPLE_RATE_HZ) / (float)NILM_WINDOW_SIZE;
        if (freq < lower || freq >= upper) continue;

        float real = 0.0f;
        float imag = 0.0f;
        for (int n = 0; n < NILM_WINDOW_SIZE; ++n) {
            float angle = 2.0f * pi * (float)k * (float)n
                / (float)NILM_WINDOW_SIZE;
            float centered = values[n] - mean;
            real += centered * cosf(angle);
            imag -= centered * sinf(angle);
        }
        energy += real * real + imag * imag;
    }
    return energy;
}

static float nilm_feature_value(const nilm_window_t *window, const char *name)
{
    float values[NILM_WINDOW_SIZE];

    if (strncmp(name, "accel_x_fft_", 12) == 0) {
        collect_series(window, "accel_x", values);
        return fft_bin_energy(values, atoi(name + 12));
    }
    if (strncmp(name, "accel_y_fft_", 12) == 0) {
        collect_series(window, "accel_y", values);
        return fft_bin_energy(values, atoi(name + 12));
    }
    if (strncmp(name, "accel_z_fft_", 12) == 0) {
        collect_series(window, "accel_z", values);
        return fft_bin_energy(values, atoi(name + 12));
    }

    const char *series = NULL;
    const char *stat = NULL;
    if (strncmp(name, "cur_", 4) == 0) {
        series = "cur";
        stat = name + 4;
    } else if (strncmp(name, "pwr_", 4) == 0) {
        series = "pwr";
        stat = name + 4;
    } else if (strncmp(name, "accel_x_", 8) == 0) {
        series = "accel_x";
        stat = name + 8;
    } else if (strncmp(name, "accel_y_", 8) == 0) {
        series = "accel_y";
        stat = name + 8;
    } else if (strncmp(name, "accel_z_", 8) == 0) {
        series = "accel_z";
        stat = name + 8;
    } else if (strncmp(name, "accel_rms_", 10) == 0) {
        series = "accel_rms";
        stat = name + 10;
    } else if (strcmp(name, "ntc_mean") == 0) {
        series = "ntc";
        stat = "mean";
    } else if (strcmp(name, "tmp117_mean") == 0) {
        series = "tmp117";
        stat = "mean";
    }

    if (series == NULL || stat == NULL) return 0.0f;

    collect_series(window, series, values);
    float mean = mean_of(values);
    if (strcmp(stat, "mean") == 0) return mean;
    if (strcmp(stat, "std") == 0) return std_of(values, mean);
    if (strcmp(stat, "min") == 0) return min_of(values);
    if (strcmp(stat, "max") == 0) return max_of(values);
    if (strcmp(stat, "range") == 0) return max_of(values) - min_of(values);
    if (strcmp(stat, "rms") == 0) return rms_of(values);
    if (strcmp(stat, "p25") == 0) return percentile(values, 0.25f);
    if (strcmp(stat, "p75") == 0) return percentile(values, 0.75f);
    if (strcmp(stat, "iqr") == 0) {
        float p25_values[NILM_WINDOW_SIZE];
        memcpy(p25_values, values, sizeof(p25_values));
        float p25 = percentile(p25_values, 0.25f);
        return percentile(values, 0.75f) - p25;
    }
    return 0.0f;
}

static ml_result_t nilm_result_from_class(int class_id, float confidence)
{
    switch (class_id) {
    case 1:
        return (ml_result_t){"Active", "ML: unloaded", confidence};
    case 2:
        return (ml_result_t){"Active", "ML: light_load", confidence};
    case 3:
        return (ml_result_t){"warning", "ML: heavy_load", confidence};
    case 4:
        return (ml_result_t){"critical", "ML: stall", confidence};
    default:
        return (ml_result_t){"warning", "ML: unknown class", confidence};
    }
}

/* Forward declarations for INA219 functions used by ml_infer_window */
static float ina219_read_power_mw(uint8_t addr);

static ml_result_t ml_infer_window(sensor_features_t f, float ax, float ay,
                                   float az)
{
    float mw = ina219_read_power_mw(INA219_A_ADDR);
    nilm_sample_t sample = {
        .accel_x = isnan(ax) ? 0.0f : ax,
        .accel_y = isnan(ay) ? 0.0f : ay,
        .accel_z = isnan(az) ? 0.0f : az,
        .accel_rms = f.vibration_rms,
        .tmp117_c = f.ambient_temperature,
        .ntc_c = f.temperature,
        .ina_a_ma = f.phase_a_current * 1000.0f,
        .ina_a_mw = isnan(mw) ? 0.0f : mw,
    };
    if (fabsf(sample.ina_a_ma) >= NILM_DEAD_CURRENT_MA) {
        nilm_window_add(&g_nilm_window, sample);
    }

    if (g_nilm_window.count < NILM_WINDOW_SIZE) {
        g_nilm_window.last_result = ml_infer(f);
        return g_nilm_window.last_result;
    }

    if (g_nilm_window.since_infer < NILM_STEP_SIZE) {
        return g_nilm_window.last_result;
    }
    g_nilm_window.since_infer = 0;

#if NILM_MODEL_AVAILABLE
    float features[NILM_MODEL_FEATURE_COUNT];
    for (size_t i = 0; i < NILM_MODEL_FEATURE_COUNT; ++i) {
        features[i] = nilm_feature_value(
            &g_nilm_window,
            nilm_model_feature_name(i)
        );
    }

    float confidence = 0.0f;
    int class_id = nilm_model_predict(features, &confidence);
    g_nilm_window.last_result = nilm_result_from_class(class_id, confidence);
#else
    g_nilm_window.last_result = ml_infer(f);
#endif

    return g_nilm_window.last_result;
}

/* ================================================================== */
/*  SENSOR DRIVERS                                                     */
/* ================================================================== */

static void i2c_init(void)
{
    i2c_master_bus_config_t cfg = {
        .clk_source                     = I2C_CLK_SRC_DEFAULT,
        .i2c_port                       = I2C_MASTER_PORT,
        .scl_io_num                     = I2C_MASTER_SCL,
        .sda_io_num                     = I2C_MASTER_SDA,
        .glitch_ignore_cnt              = 7,
        .flags.enable_internal_pullup   = true,
    };
    ESP_ERROR_CHECK(i2c_new_master_bus(&cfg, &g_i2c_bus));
}

static esp_err_t i2c_write_byte(uint8_t addr, uint8_t reg, uint8_t val)
{
    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address  = addr,
        .scl_speed_hz    = I2C_MASTER_FREQ,
    };
    i2c_master_dev_handle_t dev;
    ESP_ERROR_CHECK(i2c_master_bus_add_device(g_i2c_bus, &dev_cfg, &dev));
    uint8_t buf[2] = {reg, val};
    esp_err_t ret = i2c_master_transmit(dev, buf, 2, 100);
    i2c_master_bus_rm_device(dev);
    return ret;
}

static esp_err_t i2c_read_bytes(uint8_t addr, uint8_t reg, uint8_t *dst,
                                size_t len)
{
    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address  = addr,
        .scl_speed_hz    = I2C_MASTER_FREQ,
    };
    i2c_master_dev_handle_t dev;
    ESP_ERROR_CHECK(i2c_master_bus_add_device(g_i2c_bus, &dev_cfg, &dev));
    esp_err_t ret = i2c_master_transmit_receive(dev, &reg, 1, dst, len, 100);
    i2c_master_bus_rm_device(dev);
    return ret;
}

/* ---- TMP117 (I2C 0x48)  ------------------------------------------ */

static float tmp117_read(void)
{
    uint8_t raw[2];
    if (i2c_read_bytes(TMP117_ADDR, 0x00, raw, 2) != ESP_OK) return NAN;
    int16_t v = (int16_t)((raw[0] << 8) | raw[1]);
    return v * 0.0078125f;          /* 7.8125 m°C/LSB → °C */
}

/* ---- NTC 10k thermistor (GPIO34 / ADC1_CH6)  --------------------- */

static void ntc_init(void)
{
    adc_oneshot_unit_init_cfg_t unit_cfg = {
        .unit_id = ADC_UNIT_1,
    };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&unit_cfg, &g_adc_handle));

    adc_oneshot_chan_cfg_t chan_cfg = {
        .bitwidth = ADC_BITWIDTH_12,
        .atten    = ADC_ATTEN_DB_12,
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(g_adc_handle, NTC_ADC_CHANNEL, &chan_cfg));
}

static float ntc_read(void)
{
    int raw = 0;
    if (adc_oneshot_read(g_adc_handle, NTC_ADC_CHANNEL, &raw) != ESP_OK) return NAN;
    if (raw <= 0 || raw >= (int)ADC_MAX_COUNTS) return NAN;

    float ratio = (float)raw / ADC_MAX_COUNTS;
    float r_ntc = NTC_SERIES_R * ratio / (1.0f - ratio);
    float t0_k  = NTC_NOMINAL_T_C + 273.15f;
    float t_k   = 1.0f / (1.0f / t0_k + (1.0f / NTC_BETA)
                  * logf(r_ntc / NTC_NOMINAL_R));
    return t_k - 273.15f;
}

/* ---- INA219 (I2C 0x40 / 0x41)  ----------------------------------- */

static void ina219_init_one(uint8_t addr)
{
    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address  = addr,
        .scl_speed_hz    = I2C_MASTER_FREQ,
    };
    i2c_master_dev_handle_t dev;
    ESP_ERROR_CHECK(i2c_master_bus_add_device(g_i2c_bus, &dev_cfg, &dev));
    /* Config reg:  16 V bus, ±3.2 A range, 12‑bit, 128‑sample avg */
    uint8_t cfg[] = {0x00, 0x39, 0x9F};
    i2c_master_transmit(dev, cfg, 3, 100);
    /* Cal reg:  0.1 Ω shunt, 3.2 A max  → LSB = 100 µA, Power_LSB = 2 mW */
    uint8_t cal[] = {0x05, 0x10, 0x00};
    i2c_master_transmit(dev, cal, 3, 100);
    i2c_master_bus_rm_device(dev);
}

static void ina219_init(void)
{
    ina219_init_one(INA219_A_ADDR);
    ina219_init_one(INA219_B_ADDR);
}

static float ina219_read_current(uint8_t addr)
{
    uint8_t raw[2];
    if (i2c_read_bytes(addr, 0x04, raw, 2) != ESP_OK) return NAN;
    int16_t v = (int16_t)((raw[0] << 8) | raw[1]);
    return v * 0.0001f;             /* 100 µA/LSB → A */
}

static float ina219_read_power_mw(uint8_t addr)
{
    uint8_t raw[2];
    if (i2c_read_bytes(addr, 0x03, raw, 2) != ESP_OK) return NAN;
    int16_t v = (int16_t)((raw[0] << 8) | raw[1]);
    /* Power_LSB = 20 × Current_LSB = 2 mW/LSB when Cal=4096. */
    return v * 2.0f;
}

/* ---- ADXL345 (I2C 0x53)  ----------------------------------------- */

static void adxl345_init(void)
{
    i2c_write_byte(ADXL345_ADDR, 0x31, 0x0B);   /* DATA_FORMAT  ±16g */
    i2c_write_byte(ADXL345_ADDR, 0x2C, 0x0A);   /* BW_RATE   100 Hz */
    i2c_write_byte(ADXL345_ADDR, 0x2D, 0x08);   /* POWER_CTL  measure */
}

static void adxl345_read_raw(float *x, float *y, float *z)
{
    uint8_t raw[6];
    if (i2c_read_bytes(ADXL345_ADDR, 0x32, raw, 6) != ESP_OK) {
        *x = *y = *z = NAN;
        return;
    }
    *x = (int16_t)(raw[0] | (raw[1] << 8)) * 0.0039f * 9.81f;
    *y = (int16_t)(raw[2] | (raw[3] << 8)) * 0.0039f * 9.81f;
    *z = (int16_t)(raw[4] | (raw[5] << 8)) * 0.0039f * 9.81f;
}

/* ================================================================== */
/*  SAS TOKEN  —  HMAC‑SHA256  Shared Access Signature                  */
/* ================================================================== */

/*
 * Token format:
 *   SharedAccessSignature sr={host}%2Fdevices%2F{id}&sig={sig}&se={expiry}
 *
 *   sig  = URL‑encoded Base64 ( HMAC‑SHA256 ( URL‑host + "\n" + expiry , key ) )
 *
 * Base64‑encoded Azure device keys routinely contain  + / =  which MUST be
 * percent‑encoded in the final signature.
 */

static void url_encode_sas(const char *src, char *dst, size_t dst_len)
{
    static const char hex[] = "0123456789ABCDEF";
    size_t di = 0;
    for (size_t si = 0; src[si] && di + 3 < dst_len; si++) {
        uint8_t c = (uint8_t)src[si];
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
            || (c >= '0' && c <= '9')) {
            dst[di++] = (char)c;
        } else {
            dst[di++] = '%';
            dst[di++] = hex[c >> 4];
            dst[di++] = hex[c & 0x0F];
        }
    }
    dst[di] = '\0';
}

static bool generate_sas_token(char *token_buf, size_t buf_len)
{
    /* Epoch guard — refuse to generate before NTP syncs */
    if ((uint64_t)time(NULL) < MIN_VALID_EPOCH) {
        ESP_LOGE(TAG, "Clock not synced — refusing SAS generation");
        return false;
    }

    /* 1.  Base64‑decode the primary key  --------------------------- */
    size_t kr = 0;
    mbedtls_base64_decode(NULL, 0, &kr,
        (const unsigned char *)PRIMARY_KEY, strlen(PRIMARY_KEY));
    if (kr == 0 || kr > 256) return false;

    uint8_t raw_key[256];
    if (mbedtls_base64_decode(raw_key, sizeof(raw_key), &kr,
        (const unsigned char *)PRIMARY_KEY, strlen(PRIMARY_KEY)) != 0)
        return false;

    /* 2.  String‑to‑sign  =  URL(host/devices/deviceid) + "\n" + expiry  -- */
    time_t expiry = time(NULL) + SAS_TOKEN_TTL_SEC;
    char   resource_uri[256];
    snprintf(resource_uri, sizeof(resource_uri), "%s/devices/%s",
             IOT_HUB_HOST, DEVICE_ID);
    char url_resource[256];
    url_encode_sas(resource_uri, url_resource, sizeof(url_resource));

    char sts[512];
    int  sts_len = snprintf(sts, sizeof(sts), "%s\n%" PRIu64,
                            url_resource, (uint64_t)expiry);

    /* 3.  HMAC‑SHA256  --------------------------------------------- */
    uint8_t hmac[32];
    if (mbedtls_md_hmac(mbedtls_md_info_from_type(MBEDTLS_MD_SHA256),
            raw_key, kr, (const uint8_t *)sts, sts_len, hmac) != 0)
        return false;

    /* 4.  Base64‑encode the HMAC, then URL‑encode it  -------------- */
    size_t elen = 0;
    mbedtls_base64_encode(NULL, 0, &elen, hmac, 32);
    char b64[64];
    if (mbedtls_base64_encode((uint8_t *)b64, sizeof(b64), &elen, hmac, 32) != 0)
        return false;
    char sig_enc[96];
    url_encode_sas(b64, sig_enc, sizeof(sig_enc));

    /* 5.  Assemble full SAS token  --------------------------------- */
    snprintf(token_buf, buf_len,
        "SharedAccessSignature sig=%s&se=%" PRIu64 "&sr=%s",
        sig_enc, (uint64_t)expiry, url_resource);

    return true;
}

/* ---- Single buffer, mutex‑protected — reader/writer safe  ------- */

static char *sas_token_copy(char *dst, size_t len)
{
    xSemaphoreTake(g_sas_mutex, portMAX_DELAY);
    if (g_sas_valid) strncpy(dst, g_sas_token, len - 1);
    dst[len - 1] = '\0';
    bool ok = g_sas_valid;
    xSemaphoreGive(g_sas_mutex);
    return ok ? dst : NULL;
}

static void sas_refresh_task(void *pv)
{
    while (1) {
        char token[SAS_TOKEN_BUF_LEN];
        if (generate_sas_token(token, sizeof(token))) {
            xSemaphoreTake(g_sas_mutex, portMAX_DELAY);
            strncpy(g_sas_token, token, sizeof(g_sas_token) - 1);
            g_sas_token[sizeof(g_sas_token) - 1] = '\0';
            g_sas_valid = true;
            xSemaphoreGive(g_sas_mutex);
            ESP_LOGI(TAG, "SAS token refreshed (expires in %d s)",
                     SAS_TOKEN_TTL_SEC);
        } else {
            ESP_LOGE(TAG, "SAS token generation FAILED");
        }
        vTaskDelay(pdMS_TO_TICKS(SAS_REFRESH_SEC * 1000));
    }
}

/* ================================================================== */
/*  MOTOR  —  RMT‑driven A4988 step pulses                             */
/* ================================================================== */

/*
 * Uses ESP32 RMT peripheral (80 MHz clock, 1 MHz resolution) to generate
 * a continuous square wave on STEP_PIN.  1 RMT symbol  =  1 step pulse:
 *
 *   ┌──┐                  ┌──┐
 *   │  │                  │  │  ← level0 = H,  duration0 = PULSE_US
 * ──┘  └──────────────────┘  └──  ← level1 = L,  duration1 = period – PULSE_US
 *
 * The RMT copy encoder loops this single symbol infinitely.
 * To change RPM, stop the channel, update the symbol, and re‑transmit.
 */

#define RMT_PULSE_US    2               /* A4988 min pulse width ≈ 1 µs   */
#define RMT_RESOL_HZ    1000000         /* 1 µs per tick                   */

static void step_start(float rpm)
{
    uint32_t sprev = MOTOR_STEPS_PER_REV * MICROSTEP_MODE;
    float    sps   = (rpm * (float)sprev) / 60.0f;           /* steps/s  */
    uint32_t per   = (uint32_t)(1000000.0f / sps);           /* µs/step  */

    if (per < MOTOR_MIN_STEP_US) {
        ESP_LOGW(TAG, "RPM %.0f → %lu µs < min %d µs; clamping", rpm, per,
                 MOTOR_MIN_STEP_US);
        per = MOTOR_MIN_STEP_US;
    }

    /* Build one RMT symbol = one step pulse */
    rmt_symbol_word_t sym = {
        .duration0 = RMT_PULSE_US,          /* high 2 µs  */
        .level0    = 1,
        .duration1 = per - RMT_PULSE_US,    /* low  rest */
        .level1    = 0,
    };

    /* Stop any running transmission */
    ESP_ERROR_CHECK(rmt_disable(g_step_chan));

    /* Re‑transmit the (possibly updated) symbol in an infinite loop */
    rmt_transmit_config_t tx_cfg = { .loop_count = -1 };
    ESP_ERROR_CHECK(rmt_transmit(g_step_chan, g_step_enc,
                                 &sym, sizeof(sym), &tx_cfg));
    ESP_ERROR_CHECK(rmt_enable(g_step_chan));

    g_step_symbol = sym;
    g_steps_per_rev = sprev;
}

static void motor_init(void)
{
    /* ---- GPIO: DIR, ENABLE -------------------------------------- */
    gpio_config_t io = {
        .mode         = GPIO_MODE_OUTPUT,
        .intr_type    = GPIO_INTR_DISABLE,
        .pin_bit_mask = (1ULL << DIR_PIN)
                      | (1ULL << ENABLE_PIN),
    };
    gpio_config(&io);

    gpio_set_level(DIR_PIN,    1);          /* CW                      */
    gpio_set_level(ENABLE_PIN, 0);          /* A4988  active‑low       */

    /* ---- RMT TX channel for STEP_PIN ---------------------------- */
    rmt_tx_channel_config_t tx = {
        .gpio_num          = STEP_PIN,
        .clk_src           = RMT_CLK_SRC_DEFAULT,
        .resolution_hz     = RMT_RESOL_HZ,
        .mem_block_symbols = 64,
        .trans_queue_depth = 4,
        .intr_priority     = 0,
    };
    ESP_ERROR_CHECK(rmt_new_tx_channel(&tx, &g_step_chan));

    rmt_copy_encoder_config_t ecfg = {};
    ESP_ERROR_CHECK(rmt_new_copy_encoder(&ecfg, &g_step_enc));

    ESP_ERROR_CHECK(rmt_enable(g_step_chan));
}

static void vMotorTask(void *pv)
{
    motor_init();
    step_start(MOTOR_TARGET_RPM);

    uint32_t  step_count     = 0;
    int64_t   window_start_us = esp_timer_get_time();

    ESP_LOGI(TAG, "Motor running — %.0f RPM, %lu steps/rev, RMT on GPIO %d",
             MOTOR_TARGET_RPM, g_steps_per_rev, STEP_PIN);

    while (1) {
        /*
         * Count steps indirectly via the effective period every second.
         * step_start() already sets the exact RMT frequency; here we just
         * report the commanded RPM (identical to actual thanks to hardware
         * timing) and update the RPM atomically.
         */
        int64_t now = esp_timer_get_time();
        if (now - window_start_us >= 1000000LL) {
            float revs  = (float)step_count / (float)g_steps_per_rev;
            float rpm   = revs * 60.0f;
            taskENTER_CRITICAL(&g_rpm_mux);
            g_effective_rpm = rpm;
            taskEXIT_CRITICAL(&g_rpm_mux);
            step_count      = 0;
            window_start_us = now;
        }
        step_count++;

        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

/* ================================================================== */
/*  SENSOR TASK                                                        */
/* ================================================================== */

static void vSensorTask(void *pv)
{
    ntc_init();
    adxl345_init();
    ina219_init();

    TickType_t last_wake = xTaskGetTickCount();

    while (1) {
        sensor_features_t f = {0};

        /* ---- Temperature + current  (single sensor reads) -------- */
        f.ambient_temperature = tmp117_read();
        f.temperature         = ntc_read();
        f.phase_a_current     = ina219_read_current(INA219_A_ADDR);
        f.phase_b_current     = ina219_read_current(INA219_B_ADDR);

        if (isnan(f.ambient_temperature)) f.ambient_temperature = 0.0f;
        if (isnan(f.temperature))         f.temperature         = 0.0f;
        if (isnan(f.phase_a_current))     f.phase_a_current     = 0.0f;
        if (isnan(f.phase_b_current))     f.phase_b_current     = 0.0f;

        /* INA219‑A only — pinned to match training. */
        f.current = fabsf(f.phase_a_current);

        /* ---- Vibration  (instantaneous RMS, matching training) ---- */
        float ax, ay, az;
        adxl345_read_raw(&ax, &ay, &az);
        if (!isnan(ax)) {
            f.vibration_rms = sqrtf(ax * ax + ay * ay + az * az);
        } else {
            f.vibration_rms = 0.0f;
        }

        /* ---- RPM (atomic read — no mutex needed) ----------------- */
        taskENTER_CRITICAL(&g_rpm_mux);
        f.rpm = g_effective_rpm;
        taskEXIT_CRITICAL(&g_rpm_mux);

        /* ---- Guard against NaN ----------------------------------- */
        if (isnan(f.current))             f.current             = 0.0f;
        if (isnan(f.vibration_rms))       f.vibration_rms       = 0.0f;

        /* ---- NILM inference over a rolling 128-sample window ------ */
        ml_result_t ml = ml_infer_window(f, ax, ay, az);

        /* ---- Update shared state --------------------------------- */
        xSemaphoreTake(g_sensor_mutex, portMAX_DELAY);
        g_sensors   = f;
        g_ml_result = ml;
        xSemaphoreGive(g_sensor_mutex);

        /* ---- ISO‑8601 timestamp ---------------------------------- */
        time_t t = time(NULL);
        struct tm tm;
        gmtime_r(&t, &tm);
        char ts[32];
        strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tm);

        /* ---- Queue for Azure task -------------------------------- */
        telemetry_packet_t pkt;
        pkt.features = f;
        pkt.ml       = ml;
        strncpy(pkt.timestamp, ts, sizeof(pkt.timestamp));
        xQueueOverwrite(g_telemetry_queue, &pkt);

        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(SENSOR_PERIOD_MS));
    }
}

/* ================================================================== */
/*  AZURE MQTT TASK  —  TLS  +  SAS  +  JSON publish                    */
/* ================================================================== */

/*
 * Topic:   devices/{DEVICE_ID}/messages/events/
 * Payload matches the dashboard telemetry_live schema:
 *   {device_id, timestamp, rpm, temperature, vibration, current,
 *    status, status_message}
 */

static void mqtt_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    esp_mqtt_event_handle_t ev = (esp_mqtt_event_handle_t)data;
    switch (ev->event_id) {
    case MQTT_EVENT_CONNECTED:
        ESP_LOGI(TAG, "MQTT connected to Azure IoT Hub");        break;
    case MQTT_EVENT_DISCONNECTED:
        ESP_LOGW(TAG, "MQTT disconnected — auto‑reconnecting");  break;
    case MQTT_EVENT_ERROR:
        ESP_LOGE(TAG, "MQTT error");                             break;
    case MQTT_EVENT_PUBLISHED:
        ESP_LOGD(TAG, "Published  msg_id=%d", ev->msg_id);       break;
    default: break;
    }
}

static void vAzureTask(void *pv)
{
    char broker_uri[128];
    snprintf(broker_uri, sizeof(broker_uri), "mqtts://%s:8883", IOT_HUB_HOST);

    char mqtt_user[256];
    snprintf(mqtt_user, sizeof(mqtt_user),
             "%s/%s/?api-version=2021-04-12", IOT_HUB_HOST, DEVICE_ID);

    char pub_topic[128];
    snprintf(pub_topic, sizeof(pub_topic),
             "devices/%s/messages/events/", DEVICE_ID);

    esp_mqtt_client_config_t cfg = {
        .broker.address.uri                     = broker_uri,
        .broker.verification.crt_bundle_attach  = esp_crt_bundle_attach,
        .credentials.username                   = mqtt_user,
        .credentials.client_id                  = DEVICE_ID,
        .session.keepalive                      = 30,
        .network.disable_auto_reconnect         = true,
    };

    g_mqtt_client = esp_mqtt_client_init(&cfg);
    esp_mqtt_client_register_event(g_mqtt_client, ESP_EVENT_ANY_ID,
                                   mqtt_event_handler, NULL);

    ESP_LOGI(TAG, "Azure IoT Hub target:  %s", IOT_HUB_HOST);

    TickType_t last_pub = xTaskGetTickCount();

    while (1) {
        /* ---- Set SAS token as MQTT password -------------------- */
        char sas[SAS_TOKEN_BUF_LEN];
        if (sas_token_copy(sas, sizeof(sas)) != NULL) {
            esp_mqtt_set_config(g_mqtt_client,
                &(esp_mqtt_client_config_t){
                    .broker.address.uri                     = broker_uri,
                    .broker.verification.crt_bundle_attach  = esp_crt_bundle_attach,
                    .credentials.username                   = mqtt_user,
                    .credentials.authentication.password    = sas,
                    .credentials.client_id                  = DEVICE_ID,
                });

            if (!g_connected) {
                uint32_t mqtt_delay = (1000u << s_mqtt_retry);
                if (mqtt_delay > BACKOFF_MAX_MS) mqtt_delay = BACKOFF_MAX_MS;
                if (s_mqtt_retry < 12) s_mqtt_retry++;
                if (mqtt_delay > 0) vTaskDelay(pdMS_TO_TICKS(mqtt_delay));
                /* Destroy old handle before re-creating */
                if (g_mqtt_client != NULL) {
                    esp_mqtt_client_stop(g_mqtt_client);
                    esp_mqtt_client_destroy(g_mqtt_client);
                    g_mqtt_client = NULL;
                }
                esp_mqtt_client_config_t *re_cfg = &(esp_mqtt_client_config_t){
                    .broker.address.uri                     = broker_uri,
                    .broker.verification.crt_bundle_attach  = esp_crt_bundle_attach,
                    .credentials.username                   = mqtt_user,
                    .credentials.client_id                  = DEVICE_ID,
                    .session.keepalive                      = 30,
                    .network.disable_auto_reconnect         = true,
                };
                g_mqtt_client = esp_mqtt_client_init(re_cfg);
                esp_mqtt_client_register_event(g_mqtt_client, ESP_EVENT_ANY_ID,
                                               mqtt_event_handler, NULL);
                esp_err_t e = esp_mqtt_client_start(g_mqtt_client);
                g_connected = (e == ESP_OK);
                if (g_connected) {
                    s_mqtt_retry = 0;
                } else {
                    ESP_LOGE(TAG, "MQTT start: %d", e);
                }
            }
        }

        /* ---- Publish every 5 s --------------------------------- */
        if (g_connected
            && xTaskGetTickCount() - last_pub >= pdMS_TO_TICKS(AZURE_PUBLISH_MS)) {

            telemetry_packet_t pkt;
            if (xQueuePeek(g_telemetry_queue, &pkt, 0) == pdTRUE) {
                char payload[768];
                int  n = snprintf(payload, sizeof(payload),
                    "{"
                    "\"device_id\":\"%s\","
                    "\"timestamp\":\"%s\","
                    "\"rpm\":%.1f,"
                    "\"temperature\":%.1f,"
                    "\"ambient_temperature\":%.1f,"
                    "\"vibration\":%.2f,"
                    "\"current\":%.2f,"
                    "\"phase_a_current\":%.2f,"
                    "\"phase_b_current\":%.2f,"
                    "\"status\":\"%s\","
                    "\"status_message\":\"%s\""
                    "}",
                    DEVICE_ID, pkt.timestamp,
                    pkt.features.rpm,
                    pkt.features.temperature,
                    pkt.features.ambient_temperature,
                    pkt.features.vibration_rms,
                    pkt.features.current,
                    pkt.features.phase_a_current,
                    pkt.features.phase_b_current,
                    pkt.ml.status,
                    pkt.ml.status_message);

                if (n < 0 || n >= (int)sizeof(payload)) {
                    ESP_LOGE(TAG, "JSON overflow");
                } else {
                    int mid = esp_mqtt_client_publish(g_mqtt_client, pub_topic,
                                                      payload, 0, 1, 0);
                    if (mid < 0) {
                        s_pub_fails++;
                        ESP_LOGE(TAG, "Publish failed (%d/%d) — reconnecting",
                                 s_pub_fails, MQTT_MAX_FAILS);
                        if (s_pub_fails >= MQTT_MAX_FAILS) {
                            s_stuck_counter = STUCK_LIMIT;
                        }
                        esp_mqtt_client_stop(g_mqtt_client);
                        g_connected = false;
                    } else {
                        s_pub_fails = 0;
                        ESP_LOGI(TAG, "→ Azure  |  status=%-8s  ntc=%.1f °C  "
                                 "amb=%.1f °C  vib=%.2f g   cur=%.2f A   "
                                 "ia=%.2f A   ib=%.2f A   rpm=%.0f",
                                 pkt.ml.status, pkt.features.temperature,
                                 pkt.features.ambient_temperature,
                                 pkt.features.vibration_rms, pkt.features.current,
                                 pkt.features.phase_a_current,
                                 pkt.features.phase_b_current,
                                 pkt.features.rpm);
                    }
                }
            }
            last_pub = xTaskGetTickCount();
        }
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

/* ================================================================== */
/*  WATCHDOG                                                           */
/* ================================================================== */

static void vWatchdogTask(void *pv)
{
    esp_task_wdt_add(NULL);

    while (1) {
        if (g_sas_valid && g_connected) {
            s_stuck_counter = 0;
            esp_task_wdt_reset();
        } else {
            s_stuck_counter++;
            ESP_LOGW(TAG, "Watchdog: stuck_counter=%d  sas=%d  connected=%d",
                     s_stuck_counter, g_sas_valid, g_connected);
        }
        if (s_stuck_counter >= STUCK_LIMIT) {
            ESP_LOGE(TAG, "Watchdog: system stuck — restarting");
            esp_restart();
        }
        vTaskDelay(pdMS_TO_TICKS(30000));
    }
}

/* ================================================================== */
/*  WIFI                                                               */
/* ================================================================== */

static void wifi_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    if (id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_event_sta_disconnected_t *ev = (wifi_event_sta_disconnected_t *)data;
        switch (ev->reason) {
        case WIFI_REASON_AUTH_FAIL:
        case WIFI_REASON_NO_AP_FOUND:
        case WIFI_REASON_ASSOC_LEAVE:
            ESP_LOGE(TAG, "Wi‑Fi disconnected — permanent failure (reason %d)", ev->reason);
            return;
        default:
            break;
        }
        uint32_t delay = (1000u << s_wifi_retry);
        if (delay > BACKOFF_MAX_MS) delay = BACKOFF_MAX_MS;
        if (s_wifi_retry < 12) s_wifi_retry++;
        ESP_LOGW(TAG, "Wi‑Fi disconnected (reason %d) — reconnecting in %lu ms",
                 ev->reason, (unsigned long)delay);
        vTaskDelay(pdMS_TO_TICKS(delay));
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *ev = (ip_event_got_ip_t *)data;
        s_wifi_retry = 0;
        ESP_LOGI(TAG, "Wi‑Fi connected — IP: " IPSTR, IP2STR(&ev->ip_info.ip));
    }
}

static void wifi_init(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t any, ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &any));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, &ip));

    wifi_config_t wc = { .sta = { .ssid = WIFI_SSID, .password = WIFI_PASS } };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
}

/* ---- NTP sync callback  ----------------------------------------- */

static void sntp_sync_cb(struct timeval *tv)
{
    g_ntp_synced = true;
    ESP_LOGI(TAG, "NTP synced via callback — epoch %" PRIu64, (uint64_t)tv->tv_sec);
}

/* ================================================================== */
/*  MAIN                                                               */
/* ================================================================== */

void app_main(void)
{
    ESP_LOGI(TAG, "=== NEMA 17 Motor Telemetry Firmware ===");
    ESP_LOGI(TAG, "Device:  %s  |  RPM: %.0f  |  µstepping: 1/%d",
             DEVICE_ID, MOTOR_TARGET_RPM, MICROSTEP_MODE);

    wifi_init();
    i2c_init();

    /* ---- NTP time sync (required for SAS & payload timestamps) -- */
    vTaskDelay(pdMS_TO_TICKS(2000));  /* settle delay for network */
    esp_sntp_set_time_sync_notification_cb(sntp_sync_cb);
    esp_sntp_setoperatingmode(SNTP_OPMODE_POLL);
    esp_sntp_setservername(0, "time.google.com");
    esp_sntp_setservername(1, "pool.ntp.org");
    esp_sntp_init();
    for (int retry = 0; !g_ntp_synced && retry < 60; retry++)
        vTaskDelay(pdMS_TO_TICKS(1000));
    ESP_LOGI(TAG, "NTP %s (epoch %" PRIu64 ")",
             sntp_get_sync_status() == SNTP_SYNC_STATUS_COMPLETED
                 ? "synced" : "TIMEOUT",
             (uint64_t)time(NULL));

    /* ---- Primitives --------------------------------------------- */
    g_sas_mutex       = xSemaphoreCreateMutex();
    g_sensor_mutex    = xSemaphoreCreateMutex();
    g_telemetry_queue = xQueueCreate(1, sizeof(telemetry_packet_t));

    /* ---- Task watchdog (60 s timeout, trigger panic on expiry) -- */
    esp_task_wdt_config_t twdt_config = {
        .timeout_ms      = 60000,
        .idle_core_mask  = 0,
        .trigger_panic   = true,
    };
    ESP_ERROR_CHECK(esp_task_wdt_reconfigure(&twdt_config));

    /* ---- Tasks ---------------------------------------------------
     * watchdog    prio 1  —  health check, feeds TWDT, triggers restart
     * sas_refresh  prio 0  —  low‑priority background token refresh
     * motor        prio 2  —  RMT stepper (RMT ISR has higher hw prio)
     * sensors      prio 3  —  I2C + ML inference
     * azure        prio 1  —  MQTT publish
     *
     * Stack sizes are generous:  sensor task at 16 KB allows room for
     * moderate TFLite Micro models;  increase to 32 KB if your model
     * uses a large tensor arena (allocate arena statically, not on stack).
     * ------------------------------------------------------------- */
    xTaskCreate(vWatchdogTask,   "watchdog",   2048,  NULL, 1, NULL);
    xTaskCreate(sas_refresh_task, "sas_refresh", 4096,  NULL, 0, NULL);
    xTaskCreate(vMotorTask,       "motor",       2048,  NULL, 2, NULL);
    xTaskCreate(vSensorTask,      "sensors",     16384, NULL, 3, NULL);
    xTaskCreate(vAzureTask,       "azure",       8192,  NULL, 1, NULL);

    /* Idle the app_main task — everything runs in RTOS tasks */
    while (1) vTaskDelay(pdMS_TO_TICKS(10000));
}