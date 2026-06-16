"""
Quick diagnostic script — tests ONLY the Azure IoT Hub connection.
Run: python test_azure_connection.py

This does not need the ESP32 or model.pkl.
"""
import os
import sys

from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path)

hostname  = os.environ.get("IOT_HUB_HOSTNAME", "NOT SET")
device_id = os.environ.get("IOT_HUB_DEVICE_ID", "NOT SET")
key       = os.environ.get("IOT_HUB_SHARED_ACCESS_KEY", "NOT SET")

print("=" * 60)
print("  Azure IoT Hub Connection Diagnostic")
print("=" * 60)
print(f"  Hostname  : {hostname}")
print(f"  Device ID : {device_id}")
print(f"  Key set   : {'YES (' + str(len(key)) + ' chars)' if key != 'NOT SET' else 'NO'}")
print()

if "NOT SET" in [hostname, device_id, key]:
    print("[FAIL] One or more env variables are missing. Check your .env file.")
    sys.exit(1)

if not hostname.endswith(".azure-devices.net"):
    print(f"[WARN] Hostname '{hostname}' does not end in .azure-devices.net")
    print("       It should look like:  my-hub.azure-devices.net")

conn_str = (
    f"HostName={hostname};"
    f"DeviceId={device_id};"
    f"SharedAccessKey={key}"
)

print("Attempting to connect...")
try:
    from azure.iot.device import IoTHubDeviceClient
    client = IoTHubDeviceClient.create_from_connection_string(conn_str)
    client.connect()
    print("[✓] Connected successfully!\n")

    # Send one test message
    from azure.iot.device import Message
    import json
    test_msg = Message(json.dumps({"test": True, "message": "Connection test from bridge"}))
    test_msg.content_encoding = "utf-8"
    test_msg.content_type = "application/json"
    client.send_message(test_msg)
    print("[✓] Test message sent to Azure IoT Hub!")
    print("    Your teammate should see this arrive now.")

    client.disconnect()
    print("[✓] Disconnected cleanly.")

except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")
    print()
    print("Possible causes:")
    print("  1. Wrong credentials — double-check IOT_HUB_SHARED_ACCESS_KEY in .env")
    print("  2. Wrong hostname  — make sure it ends in .azure-devices.net")
    print("  3. Device not registered on the Hub — ask teammate to confirm deviceId")
    print("  4. Firewall blocking port 8883 — try on a different network (hotspot)")
    sys.exit(1)
