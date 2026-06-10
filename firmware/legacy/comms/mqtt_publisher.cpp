#include "mqtt_publisher.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <math.h>
#include "../config.h"

// ═══════════════════════════════════════════════════════════════════
//  mqtt_publisher.cpp — Wi-Fi + ThingSpeak MQTT publisher
//
//  Broker:  mqtt3.thingspeak.com:1883
//  Topic:   channels/<CHANNEL_ID>/publish
//  Payload: JSON per context.md MQTT Telemetry Payload Schema
//
//  Production migration (Step 12):
//   Swap MQTT_BROKER → AWS IoT Core endpoint
//   Add TLS via WiFiClientSecure + device certificate
// ═══════════════════════════════════════════════════════════════════

static WiFiClient   _wifi_client;
static PubSubClient _mqtt(_wifi_client);

// ─── Wi-Fi ──────────────────────────────────────────────────────────────────
static bool wifi_connect() {
    if (WiFi.status() == WL_CONNECTED) return true;

    Serial.printf("[MQTT] Connecting to Wi-Fi SSID: %s\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    uint32_t start = millis();
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - start > WIFI_TIMEOUT_MS) {
            Serial.println("[MQTT] Wi-Fi timeout — will retry next publish cycle");
            return false;
        }
        vTaskDelay(pdMS_TO_TICKS(500));
        Serial.print(".");
    }
    Serial.printf("\n[MQTT] Wi-Fi connected — IP: %s\n", WiFi.localIP().toString().c_str());
    return true;
}

// ─── MQTT Reconnect ─────────────────────────────────────────────────────────
static bool mqtt_reconnect() {
    if (_mqtt.connected()) return true;

    Serial.printf("[MQTT] Connecting to broker %s:%d\n", MQTT_BROKER, MQTT_PORT);

    if (_mqtt.connect(MQTT_CLIENT_ID, MQTT_USERNAME, MQTT_PASSWORD)) {
        Serial.println("[MQTT] Broker connected");
        return true;
    }

    Serial.printf("[MQTT] Connection failed, rc=%d — will retry\n", _mqtt.state());
    return false;
}

// ─── Public API ─────────────────────────────────────────────────────────────
void mqtt_init() {
    _mqtt.setServer(MQTT_BROKER, MQTT_PORT);
    _mqtt.setKeepAlive(MQTT_KEEPALIVE);
    _mqtt.setBufferSize(512);   // Ensure buffer fits our JSON payload

    if (wifi_connect()) {
        mqtt_reconnect();
    }
}

void mqtt_publish(const TelemetryPayload& payload) {
    if (!wifi_connect() || !mqtt_reconnect()) {
        Serial.println("[MQTT] Publish skipped — not connected");
        return;
    }

    // ── Serialise to JSON (ArduinoJson v7) ──────────────────────────────────
    JsonDocument doc;

    doc["ts"]         = payload.timestamp_ms;
    doc["accel_x"]    = serialized(String(payload.accel.x,   4));
    doc["accel_y"]    = serialized(String(payload.accel.y,   4));
    doc["accel_z"]    = serialized(String(payload.accel.z,   4));
    doc["accel_rms"]  = serialized(String(payload.accel.rms, 4));
    doc["t_ambient"]  = serialized(String(payload.t_ambient, 2));
    doc["t_winding"]  = serialized(String(payload.t_winding, 2));
    doc["delta_t"]    = serialized(String(payload.delta_t,   3));
    doc["motor_rpm"]  = payload.motor_rpm;
    doc["load_class"] = payload.load_class;
    doc["fault"]      = payload.fault;

    // INA219 fields: null if sensors not available
    if (payload.ina219_valid && !isnan(payload.i_phaseA)) {
        doc["i_phaseA"] = serialized(String(payload.i_phaseA, 2));
        doc["i_phaseB"] = serialized(String(payload.i_phaseB, 2));
    } else {
        doc["i_phaseA"] = nullptr;
        doc["i_phaseB"] = nullptr;
    }

    char json_buf[512];
    size_t n = serializeJson(doc, json_buf, sizeof(json_buf));

    bool ok = _mqtt.publish(MQTT_TOPIC, json_buf, n);
    if (ok) {
        Serial.printf("[MQTT] Published %d bytes → %s\n", (int)n, MQTT_TOPIC);
        Serial.printf("[MQTT] %s\n", json_buf);
    } else {
        Serial.println("[MQTT] Publish FAILED — buffer overflow or connection dropped");
    }
}

void mqtt_loop() {
    _mqtt.loop();   // Keeps connection alive, processes incoming messages
}

bool mqtt_is_connected() {
    return _mqtt.connected();
}