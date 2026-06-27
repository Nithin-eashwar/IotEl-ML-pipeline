#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

source .env

docker run --rm \
    -v "$PWD:/workspace" \
    -w /workspace \
    -e WIFI_SSID \
    -e WIFI_PASS \
    -e IOT_HUB_HOST \
    -e DEVICE_ID \
    -e PRIMARY_KEY \
    espressif/idf:v6.0.2 \
    idf.py "$@"
