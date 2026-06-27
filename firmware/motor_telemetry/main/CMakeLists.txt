idf_component_register(
    SRCS
        "motor_telemetry.c"
        "nilm_model.c"
    INCLUDE_DIRS
        "."
    REQUIRES
        esp_driver_i2c
        esp_driver_gpio
        esp_driver_rmt
        esp_adc
        esp_wifi
        esp_event
        espressif__mqtt
        nvs_flash
        mbedtls
        esp_timer
        freertos
        esp_netif
)

# --- Build-time secrets (read from host environment, never commit) ---
# Set these in your shell before building, or via a .env file sourced by your build script:
#   export WIFI_SSID="MySSID"
#   export WIFI_PASS="MyPassword"
#   export IOT_HUB_HOST="my-hub.azure-devices.net"
#   export DEVICE_ID="my-device-id"
#   export PRIMARY_KEY="base64-encoded-device-key"
if(DEFINED ENV{WIFI_SSID})
    add_definitions(-DWIFI_SSID=\"$ENV{WIFI_SSID}\")
else()
    add_definitions(-DWIFI_SSID=\"\")
endif()
if(DEFINED ENV{WIFI_PASS})
    add_definitions(-DWIFI_PASS=\"$ENV{WIFI_PASS}\")
else()
    add_definitions(-DWIFI_PASS=\"\")
endif()
if(DEFINED ENV{IOT_HUB_HOST})
    add_definitions(-DIOT_HUB_HOST=\"$ENV{IOT_HUB_HOST}\")
else()
    add_definitions(-DIOT_HUB_HOST=\"\")
endif()
if(DEFINED ENV{DEVICE_ID})
    add_definitions(-DDEVICE_ID=\"$ENV{DEVICE_ID}\")
else()
    add_definitions(-DDEVICE_ID=\"\")
endif()
if(DEFINED ENV{PRIMARY_KEY})
    add_definitions(-DPRIMARY_KEY=\"$ENV{PRIMARY_KEY}\")
else()
    add_definitions(-DPRIMARY_KEY=\"\")
endif()