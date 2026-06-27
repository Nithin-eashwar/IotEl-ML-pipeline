/*
 * telemetry_test.c  —  Azure IoT Hub connectivity test (ESP-IDF v6.x)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_sntp.h"
#include "nvs_flash.h"
#include "esp_crt_bundle.h"
#include "mqtt_client.h"
#include "mbedtls/base64.h"
#include "mbedtls/md.h"

/* ================================================================== */
/*  CONFIG                                                             */
/* ================================================================== */

#define WIFI_SSID           "YOUR_WIFI_SSID"
#define WIFI_PASS           "YOUR_WIFI_PASSWORD"

#define IOT_HUB_HOST        "your-hub.azure-devices.net"
#define DEVICE_ID           "nema17-bay3-abc123"
#define PRIMARY_KEY         "your-base64-primary-key=="

/* ================================================================== */
/*  TIMING                                                             */
/* ================================================================== */

#define SEND_INTERVAL_MS    15000
#define SAS_TOKEN_TTL_SEC   3600
#define SAS_REFRESH_SEC     3300
#define SAS_TOKEN_BUF_LEN   512
#define BACKOFF_MAX_MS      60000

/* Minimum acceptable epoch — Jan 1 2024.
 * If time(NULL) is below this, NTP has not synced yet. */
#define MIN_VALID_EPOCH     1704067200ULL

/* ================================================================== */
/*  SIMULATED SENSOR BASE VALUES                                       */
/* ================================================================== */

#define BASE_RPM        3450.0f
#define BASE_TEMP       42.5f
#define BASE_VIBRATION  1.2f
#define BASE_CURRENT    12.1f
#define BASE_VOLTAGE    220.4f

/* ================================================================== */
/*  GLOBALS                                                            */
/* ================================================================== */

static const char *TAG = "TEST";

static char              g_sas_token[SAS_TOKEN_BUF_LEN];
static bool              g_sas_valid  = false;
static SemaphoreHandle_t g_sas_mutex  = NULL;

static volatile bool     g_wifi_ready   = false;
static volatile bool     g_ntp_synced   = false;
static volatile bool     g_mqtt_connected = false;
static uint32_t          s_wifi_retry   = 0;
static uint32_t          s_mqtt_retry   = 0;

static esp_mqtt_client_handle_t g_mqtt = NULL;
static int64_t g_start_us = 0;

/* ================================================================== */
/*  PRNG                                                               */
/* ================================================================== */

static uint32_t g_prng = 12345;

static float prng_uniform(float lo, float hi)
{
    g_prng = (uint32_t)(((uint64_t)g_prng * 279470273ULL) % 4294967291ULL);
    float t = (float)g_prng / 4294967291.0f;
    return lo + t * (hi - lo);
}

/* ================================================================== */
/*  URL ENCODE                                                         */
/* ================================================================== */

static void url_encode(const char *src, char *dst, size_t dst_len)
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

/* ================================================================== */
/*  SAS TOKEN                                                          */
/* ================================================================== */

static bool generate_sas_token(char *buf, size_t len)
{
    /* Guard: refuse to generate if clock is not valid */
    if ((uint64_t)time(NULL) < MIN_VALID_EPOCH) {
        ESP_LOGE(TAG, "Clock not synced — refusing SAS generation (epoch %" PRIu64 ")",
                 (uint64_t)time(NULL));
        return false;
    }

    size_t kr = 0;
    mbedtls_base64_decode(NULL, 0, &kr,
        (const uint8_t *)PRIMARY_KEY, strlen(PRIMARY_KEY));
    if (kr == 0 || kr > 256) return false;

    uint8_t raw_key[256];
    if (mbedtls_base64_decode(raw_key, sizeof(raw_key), &kr,
            (const uint8_t *)PRIMARY_KEY, strlen(PRIMARY_KEY)) != 0)
        return false;

    time_t expiry = time(NULL) + SAS_TOKEN_TTL_SEC;
    char resource_uri[256];
    snprintf(resource_uri, sizeof(resource_uri), "%s/devices/%s",
            IOT_HUB_HOST, DEVICE_ID);
    char url_resource[256];
    url_encode(resource_uri, url_resource, sizeof(url_resource));

    char sts[512];
    int sts_len = snprintf(sts, sizeof(sts), "%s\n%" PRIu64,
                        url_resource, (uint64_t)expiry);

    uint8_t hmac[32];
    if (mbedtls_md_hmac(mbedtls_md_info_from_type(MBEDTLS_MD_SHA256),
            raw_key, kr, (const uint8_t *)sts, sts_len, hmac) != 0)
        return false;

    size_t elen = 0;
    char b64[64];
    if (mbedtls_base64_encode((uint8_t *)b64, sizeof(b64), &elen, hmac, 32) != 0)
        return false;
    char sig[96];
    url_encode(b64, sig, sizeof(sig));

    snprintf(buf, len,
        "SharedAccessSignature sig=%s&se=%" PRIu64 "&sr=%s",
        sig, (uint64_t)expiry, url_resource);

    ESP_LOGI(TAG, "SAS expiry: %" PRIu64 "  now: %" PRIu64,
             (uint64_t)expiry, (uint64_t)time(NULL));
    return true;
}

static void sas_refresh_task(void *pv)
{
    while (1) {
        /* Wait until NTP has actually synced */
        while (!g_ntp_synced) {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }

        char token[SAS_TOKEN_BUF_LEN];
        if (generate_sas_token(token, sizeof(token))) {
            xSemaphoreTake(g_sas_mutex, portMAX_DELAY);
            strncpy(g_sas_token, token, SAS_TOKEN_BUF_LEN - 1);
            g_sas_token[SAS_TOKEN_BUF_LEN - 1] = '\0';
            g_sas_valid = true;
            xSemaphoreGive(g_sas_mutex);
            ESP_LOGI(TAG, "SAS token refreshed");
        } else {
            ESP_LOGE(TAG, "SAS token generation failed");
        }
        vTaskDelay(pdMS_TO_TICKS(SAS_REFRESH_SEC * 1000));
    }
}

/* ================================================================== */
/*  NTP callback                                                       */
/* ================================================================== */

static void sntp_sync_cb(struct timeval *tv)
{
    g_ntp_synced = true;
    ESP_LOGI(TAG, "NTP synced via callback — epoch %" PRIu64,
             (uint64_t)tv->tv_sec);
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
        wifi_event_sta_disconnected_t *ev = data;
        if (ev->reason == WIFI_REASON_AUTH_FAIL
                || ev->reason == WIFI_REASON_NO_AP_FOUND) {
            ESP_LOGE(TAG, "WiFi permanent failure (reason %d)", ev->reason);
            return;
        }
        uint32_t delay = (1000u << s_wifi_retry);
        if (delay > BACKOFF_MAX_MS) delay = BACKOFF_MAX_MS;
        if (s_wifi_retry < 12) s_wifi_retry++;
        ESP_LOGW(TAG, "WiFi disconnected — retry in %lu ms", (unsigned long)delay);
        vTaskDelay(pdMS_TO_TICKS(delay));
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *ev = data;
        s_wifi_retry = 0;
        g_wifi_ready = true;
        ESP_LOGI(TAG, "WiFi connected — IP: " IPSTR, IP2STR(&ev->ip_info.ip));
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

    wifi_config_t wc = {
        .sta = { .ssid = WIFI_SSID, .password = WIFI_PASS }
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
}

/* ================================================================== */
/*  MQTT                                                               */
/* ================================================================== */

static void mqtt_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    esp_mqtt_event_handle_t ev = data;
    switch (ev->event_id) {
    case MQTT_EVENT_CONNECTED:
        ESP_LOGI(TAG, "MQTT connected");
        g_mqtt_connected = true;
        s_mqtt_retry     = 0;
        break;
    case MQTT_EVENT_DISCONNECTED:
        ESP_LOGW(TAG, "MQTT disconnected");
        g_mqtt_connected = false;
        break;
    case MQTT_EVENT_ERROR:
        ESP_LOGE(TAG, "MQTT error");
        g_mqtt_connected = false;
        break;
    case MQTT_EVENT_PUBLISHED:
        ESP_LOGD(TAG, "Published msg_id=%d", ev->msg_id);
        break;
    default:
        break;
    }
}

static void mqtt_connect(void)
{
    char broker[128];
    snprintf(broker, sizeof(broker), "mqtts://%s:8883", IOT_HUB_HOST);

    char user[256];
    snprintf(user, sizeof(user),
             "%s/%s/?api-version=2021-04-12", IOT_HUB_HOST, DEVICE_ID);

    char sas[SAS_TOKEN_BUF_LEN] = {0};
    xSemaphoreTake(g_sas_mutex, portMAX_DELAY);
    if (g_sas_valid) strncpy(sas, g_sas_token, sizeof(sas) - 1);
    bool valid = g_sas_valid;
    xSemaphoreGive(g_sas_mutex);

    if (!valid) {
        ESP_LOGE(TAG, "No valid SAS token — skipping MQTT connect");
        return;
    }

    /* Always destroy and recreate the client to avoid ESP_ERR_INVALID_STATE */
    if (g_mqtt != NULL) {
        esp_mqtt_client_stop(g_mqtt);
        esp_mqtt_client_destroy(g_mqtt);
        g_mqtt = NULL;
    }

    esp_mqtt_client_config_t cfg = {
        .broker.address.uri                    = broker,
        .broker.verification.crt_bundle_attach = esp_crt_bundle_attach,
        .credentials.username                  = user,
        .credentials.client_id                 = DEVICE_ID,
        .credentials.authentication.password   = sas,
        .session.keepalive                     = 30,
        .network.disable_auto_reconnect        = true,
    };

    g_mqtt = esp_mqtt_client_init(&cfg);
    esp_mqtt_client_register_event(g_mqtt, ESP_EVENT_ANY_ID,
                                   mqtt_event_handler, NULL);

    uint32_t delay = (1000u << s_mqtt_retry);
    if (delay > BACKOFF_MAX_MS) delay = BACKOFF_MAX_MS;
    if (s_mqtt_retry < 8) s_mqtt_retry++;
    if (delay > 0) {
        ESP_LOGI(TAG, "MQTT backoff %lu ms", (unsigned long)delay);
        vTaskDelay(pdMS_TO_TICKS(delay));
    }

    esp_err_t e = esp_mqtt_client_start(g_mqtt);
    if (e != ESP_OK) {
        ESP_LOGE(TAG, "MQTT start failed: %d", e);
    }
}

/* ================================================================== */
/*  SIMULATED SENSOR DATA                                              */
/* ================================================================== */

typedef struct {
    float rpm;
    float temperature;
    float vibration;
    float current;
    float voltage;
    const char *status;
    const char *status_message;
} sim_telemetry_t;

static sim_telemetry_t simulate(void)
{
    float elapsed = (float)(esp_timer_get_time() - g_start_us) / 1e6f;

    sim_telemetry_t t;
    t.rpm         = BASE_RPM       + 120.0f * sinf(elapsed / 60.0f);
    t.temperature = BASE_TEMP      + 2.5f   * sinf(elapsed / 120.0f)
                                   + prng_uniform(-0.3f, 0.3f);
    t.vibration   = BASE_VIBRATION + 0.15f  * sinf(elapsed / 30.0f)
                                   + prng_uniform(-0.05f, 0.05f);
    t.current     = BASE_CURRENT   + (t.rpm - BASE_RPM) * 0.002f
                                   + prng_uniform(-0.2f, 0.2f);
    t.voltage     = BASE_VOLTAGE   + prng_uniform(-1.5f, 1.5f);

    t.status         = "ok";
    t.status_message = "All parameters within nominal range";
    if (t.temperature > 50.0f) {
        t.status         = "warning";
        t.status_message = "Motor temperature elevated";
    }
    if (t.vibration > 2.5f) {
        t.status         = "critical";
        t.status_message = "Excessive vibration detected";
    }
    return t;
}

/* ================================================================== */
/*  TELEMETRY TASK                                                     */
/* ================================================================== */

static void vTelemetryTask(void *pv)
{
    char topic[128];
    snprintf(topic, sizeof(topic), "devices/%s/messages/events/", DEVICE_ID);

    TickType_t last_send = xTaskGetTickCount();

    while (1) {
        /* Reconnect if not connected */
        if (!g_mqtt_connected && g_wifi_ready && g_sas_valid) {
            mqtt_connect();
            for (int i = 0; i < 20 && !g_mqtt_connected; i++) {
                vTaskDelay(pdMS_TO_TICKS(500));
            }
            if (!g_mqtt_connected) {
                ESP_LOGE(TAG, "MQTT connect failed after 10 s — will retry");
            }
        }

        /* Publish on interval */
        if (g_mqtt_connected
                && xTaskGetTickCount() - last_send >= pdMS_TO_TICKS(SEND_INTERVAL_MS)) {

            sim_telemetry_t s = simulate();

            time_t now = time(NULL);
            struct tm tm_info;
            gmtime_r(&now, &tm_info);
            char ts[32];
            strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tm_info);

            char payload[512];
            int n = snprintf(payload, sizeof(payload),
                "{"
                "\"device_id\":\"%s\","
                "\"timestamp\":\"%s\","
                "\"rpm\":%.2f,"
                "\"temperature_c\":%.2f,"
                "\"vibration_mms\":%.2f,"
                "\"current_a\":%.2f,"
                "\"voltage_v\":%.2f,"
                "\"status\":\"%s\","
                "\"status_message\":\"%s\""
                "}",
                DEVICE_ID, ts,
                s.rpm, s.temperature, s.vibration,
                s.current, s.voltage,
                s.status, s.status_message);

            if (n < 0 || n >= (int)sizeof(payload)) {
                ESP_LOGE(TAG, "Payload overflow");
            } else {
                int mid = esp_mqtt_client_publish(g_mqtt, topic,
                                                  payload, 0, 1, 0);
                if (mid >= 0) {
                    ESP_LOGI(TAG,
                        "→ Azure | status=%-8s  rpm=%.1f  temp=%.1f°C  "
                        "vib=%.2f  cur=%.2f A  volt=%.1f V",
                        s.status, s.rpm, s.temperature,
                        s.vibration, s.current, s.voltage);
                } else {
                    ESP_LOGE(TAG, "Publish failed — will reconnect");
                    g_mqtt_connected = false;
                }
            }
            last_send = xTaskGetTickCount();
        }

        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

/* ================================================================== */
/*  MAIN                                                               */
/* ================================================================== */

void app_main(void)
{
    ESP_LOGI(TAG, "=== Azure IoT Hub Telemetry Test ===");
    ESP_LOGI(TAG, "Device: %s  |  Interval: %d s", DEVICE_ID,
             SEND_INTERVAL_MS / 1000);

    wifi_init();

    /* Wait for WiFi IP */
    ESP_LOGI(TAG, "Waiting for WiFi...");
    for (int i = 0; i < 30 && !g_wifi_ready; i++) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
    if (!g_wifi_ready) {
        ESP_LOGE(TAG, "WiFi not ready after 30 s — halting");
        while (1) vTaskDelay(pdMS_TO_TICKS(5000));
    }

    /* Settle delay */
    vTaskDelay(pdMS_TO_TICKS(2000));

    /* NTP — register callback so we know exactly when sync completes */
    esp_sntp_set_time_sync_notification_cb(sntp_sync_cb);
    esp_sntp_setoperatingmode(SNTP_OPMODE_POLL);
    esp_sntp_setservername(0, "time.google.com");   /* more reliable than pool.ntp.org */
    esp_sntp_setservername(1, "pool.ntp.org");       /* fallback */
    esp_sntp_init();

    ESP_LOGI(TAG, "Waiting for NTP sync (up to 60 s)...");
    for (int retry = 0; !g_ntp_synced && retry < 60; retry++) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }

    if (g_ntp_synced) {
        ESP_LOGI(TAG, "NTP synced — epoch %" PRIu64, (uint64_t)time(NULL));
    } else {
        ESP_LOGE(TAG, "NTP FAILED — cannot proceed without valid time. Halting.");
        ESP_LOGE(TAG, "Check: hotspot blocks UDP 123? Try a different network.");
        while (1) vTaskDelay(pdMS_TO_TICKS(5000));
    }

    g_start_us  = esp_timer_get_time();
    g_sas_mutex = xSemaphoreCreateMutex();

    xTaskCreate(sas_refresh_task, "sas_refresh", 4096, NULL, 2, NULL);
    vTaskDelay(pdMS_TO_TICKS(500));  /* wait for first SAS token */

    xTaskCreate(vTelemetryTask, "telemetry", 8192, NULL, 3, NULL);

    while (1) vTaskDelay(pdMS_TO_TICKS(10000));
}