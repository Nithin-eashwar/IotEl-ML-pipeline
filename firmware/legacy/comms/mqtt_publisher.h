#pragma once
#include "../types.h"

// ═══════════════════════════════════════════════════════════════════
//  mqtt_publisher.h — Wi-Fi connection + ThingSpeak MQTT publish
//  Uses PubSubClient + ArduinoJson v7
// ═══════════════════════════════════════════════════════════════════

void mqtt_init();
void mqtt_publish(const TelemetryPayload& payload);
void mqtt_loop();   // Call periodically to keep connection alive
bool mqtt_is_connected();