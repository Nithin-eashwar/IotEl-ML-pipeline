/*
 * NILM Feature Validation Firmware
 * =================================
 * Captures one 128-sample window from live sensors, computes all 20 features
 * in C using the exact nilm_feature_value() code path, and dumps raw samples
 * plus computed features over Serial as JSON.
 *
 * Usage:
 *   idf.py flash monitor  →  copy the JSON block  →  feed to compare_nilm_features.py
 *
 * Sensors required:  ADXL345 (0x53), TMP117 (0x48), INA219A (0x40)
 * INA219B (0x41) is initialised but not used for features (matches training).
 *
 * This is a STANDALONE test tool.  Do NOT combine with the full telemetry
 * firmware — it replaces app_main() and only runs once.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/i2c.h"
#include "driver/gpio.h"
#include "driver/adc.h"

#include "nilm_model.h"   /* only for NILM_MODEL_FEATURE_COUNT, nilm_model_feature_name() */

/* ---------- Pin / address map (identical to esp32_telemetry.c) ------ */

#define I2C_MASTER_SCL      GPIO_NUM_22
#define I2C_MASTER_SDA      GPIO_NUM_21
#define I2C_MASTER_FREQ     400000
#define I2C_MASTER_PORT     I2C_NUM_0

#define ADXL345_ADDR        0x53
#define TMP117_ADDR         0x48
#define INA219_A_ADDR       0x40
#define INA219_B_ADDR       0x41

/* NTC ADC */
#define NTC_ADC_CHANNEL     ADC1_CHANNEL_6   /* GPIO34 */
#define NTC_NOMINAL_R       10000.0f
#define NTC_NOMINAL_T_C     25.0f
#define NTC_BETA            3950.0f
#define NTC_SERIES_R        10000.0f
#define ADC_MAX_COUNTS      4095.0f

/* Timing */
#define SENSOR_PERIOD_MS    10               /* 100 Hz */

/* ---------- NILM windowing (must match ml/nilm_pipeline.py) --------- */

#define NILM_WINDOW_SIZE    128
#define NILM_STEP_SIZE      64
#define NILM_SAMPLE_RATE_HZ 100.0f
#define NILM_DEAD_CURRENT_MA 2.0f

/* ---------- Data structures (identical to esp32_telemetry.c) -------- */

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
} nilm_window_t;

/* ---------- Globals ------------------------------------------------- */

static nilm_window_t g_window = {0};

/* ==================================================================== */
/*  Feature computation  (exact copy of esp32_telemetry.c)               */
/* ==================================================================== */

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
}

static const nilm_sample_t *nilm_window_sample(const nilm_window_t *window,
                                               size_t index)
{
    size_t start = window->count == NILM_WINDOW_SIZE ? window->next : 0;
    return &window->samples[(start + index) % NILM_WINDOW_SIZE];
}

static float sample_value(const nilm_sample_t *sample, const char *series)
{
    if (strcmp(series, "cur") == 0)       return sample->ina_a_ma;
    if (strcmp(series, "pwr") == 0)       return sample->ina_a_mw;
    if (strcmp(series, "accel_x") == 0)   return sample->accel_x;
    if (strcmp(series, "accel_y") == 0)   return sample->accel_y;
    if (strcmp(series, "accel_z") == 0)   return sample->accel_z;
    if (strcmp(series, "accel_rms") == 0) return sample->accel_rms;
    if (strcmp(series, "ntc") == 0)       return sample->ntc_c;
    if (strcmp(series, "tmp117") == 0)    return sample->tmp117_c;
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

/* ==================================================================== */
/*  Sensor drivers  (identical to esp32_telemetry.c)                     */
/* ==================================================================== */

static void i2c_init(void)
{
    i2c_config_t conf = {
        .mode             = I2C_MODE_MASTER,
        .sda_io_num       = I2C_MASTER_SDA,
        .scl_io_num       = I2C_MASTER_SCL,
        .sda_pullup_en    = GPIO_PULLUP_ENABLE,
        .scl_pullup_en    = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ,
    };
    i2c_param_config(I2C_MASTER_PORT, &conf);
    i2c_driver_install(I2C_MASTER_PORT, conf.mode, 0, 0, 0);
}

static esp_err_t i2c_write_byte(uint8_t addr, uint8_t reg, uint8_t val)
{
    uint8_t buf[2] = {reg, val};
    return i2c_master_write_to_device(I2C_MASTER_PORT, addr, buf, 2, 10);
}

static esp_err_t i2c_read_bytes(uint8_t addr, uint8_t reg, uint8_t *dst,
                                size_t len)
{
    return i2c_master_write_read_device(I2C_MASTER_PORT, addr, &reg, 1,
                                        dst, len, 10);
}

static float tmp117_read(void)
{
    uint8_t raw[2];
    if (i2c_read_bytes(TMP117_ADDR, 0x00, raw, 2) != ESP_OK) return NAN;
    int16_t v = (int16_t)((raw[0] << 8) | raw[1]);
    return v * 0.0078125f;
}

static void ntc_init(void)
{
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(NTC_ADC_CHANNEL, ADC_ATTEN_DB_11);
}

static float ntc_read(void)
{
    int raw = adc1_get_raw(NTC_ADC_CHANNEL);
    if (raw <= 0 || raw >= (int)ADC_MAX_COUNTS) return NAN;

    float ratio = (float)raw / ADC_MAX_COUNTS;
    float r_ntc = NTC_SERIES_R * ratio / (1.0f - ratio);
    float t0_k  = NTC_NOMINAL_T_C + 273.15f;
    float t_k   = 1.0f / (1.0f / t0_k + (1.0f / NTC_BETA)
                  * logf(r_ntc / NTC_NOMINAL_R));
    return t_k - 273.15f;
}

static void ina219_init_one(uint8_t addr)
{
    uint8_t cfg[] = {0x00, 0x39, 0x9F};
    i2c_master_write_to_device(I2C_MASTER_PORT, addr, cfg, 3, 10);
    /* Cal: 0.1 Ω shunt, 100 µA/LSB → 2 mW/LSB power */
    uint8_t cal[] = {0x05, 0x10, 0x00};
    i2c_master_write_to_device(I2C_MASTER_PORT, addr, cal, 3, 10);
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
    return v * 0.0001f;
}

static float ina219_read_power_mw(uint8_t addr)
{
    uint8_t raw[2];
    if (i2c_read_bytes(addr, 0x03, raw, 2) != ESP_OK) return NAN;
    int16_t v = (int16_t)((raw[0] << 8) | raw[1]);
    return v * 2.0f;
}

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

/* ==================================================================== */
/*  JSON dump helper                                                     */
/* ==================================================================== */

static void dump_window_json(void)
{
    printf("\n=== NILM_WINDOW_DUMP_START ===\n");
    printf("{\n");

    /* ---- Raw samples ---- */
    printf("  \"samples\": [\n");
    for (size_t i = 0; i < NILM_WINDOW_SIZE; ++i) {
        const nilm_sample_t *s = nilm_window_sample(&g_window, i);
        printf("    %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f%s\n",
               s->accel_x, s->accel_y, s->accel_z, s->accel_rms,
               s->tmp117_c, s->ntc_c, s->ina_a_ma, s->ina_a_mw,
               i == NILM_WINDOW_SIZE - 1 ? "" : ",");
    }
    printf("  ],\n");

    /* ---- Features ---- */
    printf("  \"features\": {\n");
    for (size_t i = 0; i < NILM_MODEL_FEATURE_COUNT; ++i) {
        const char *name = nilm_model_feature_name(i);
        float value = nilm_feature_value(&g_window, name);
        printf("    \"%s\": %.9g%s\n",
               name, (double)value,
               i == NILM_MODEL_FEATURE_COUNT - 1 ? "" : ",");
    }
    printf("  }\n");
    printf("}\n");
    printf("=== NILM_WINDOW_DUMP_END ===\n");

    fflush(stdout);
}

/* ==================================================================== */
/*  Main — capture one window                                            */
/* ==================================================================== */

void app_main(void)
{
    i2c_init();
    ntc_init();
    adxl345_init();
    ina219_init();

    printf("=== NILM Feature Validation Firmware ===\n");
    printf("Collecting %d samples at %d Hz (%.1f s)...\n",
           NILM_WINDOW_SIZE, 1000 / SENSOR_PERIOD_MS,
           (float)NILM_WINDOW_SIZE * SENSOR_PERIOD_MS / 1000.0f);

    TickType_t last_wake = xTaskGetTickCount();
    size_t collected = 0;

    while (g_window.count < NILM_WINDOW_SIZE) {
        float ax, ay, az;
        adxl345_read_raw(&ax, &ay, &az);
        float accel_rms = isnan(ax) ? 0.0f : sqrtf(ax * ax + ay * ay + az * az);

        float amb = tmp117_read();
        float ntc = ntc_read();
        float i_ma = ina219_read_current(INA219_A_ADDR) * 1000.0f;
        float mw = ina219_read_power_mw(INA219_A_ADDR);

        nilm_sample_t sample = {
            .accel_x   = isnan(ax)  ? 0.0f : ax,
            .accel_y   = isnan(ay)  ? 0.0f : ay,
            .accel_z   = isnan(az)  ? 0.0f : az,
            .accel_rms = isnan(accel_rms) ? 0.0f : accel_rms,
            .tmp117_c  = isnan(amb) ? 0.0f : amb,
            .ntc_c     = isnan(ntc) ? 0.0f : ntc,
            .ina_a_ma  = isnan(i_ma) ? 0.0f : i_ma,
            .ina_a_mw  = isnan(mw)   ? 0.0f : mw,
        };

        if (fabsf(sample.ina_a_ma) >= NILM_DEAD_CURRENT_MA) {
            nilm_window_add(&g_window, sample);
            collected++;
        }

        if (collected % 16 == 0) printf("  %zu/%d\r", g_window.count, NILM_WINDOW_SIZE);

        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(SENSOR_PERIOD_MS));
    }

    printf("\nCapture complete.  Dumping window and features...\n");
    dump_window_json();
    printf("\nDone.\n");

    while (1) vTaskDelay(1000);
}
