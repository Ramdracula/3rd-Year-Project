import json
import cv2
import time
from picamera2 import Picamera2
from libcamera import controls

with open("calibration.json") as f:
    cal = json.load(f)

preview_w, preview_h = cal["preview_resolution"]

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (preview_w, preview_h)}
)
picam2.configure(config)
picam2.start()
time.sleep(1)

# Autofocus once, then save the found lens position
picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
picam2.set_controls({"AfTrigger": controls.AfTriggerEnum.Start})
time.sleep(2)

meta = picam2.capture_metadata()
print("Autofocus metadata:", meta)

if "LensPosition" in meta:
    cal["lens_position"] = meta["LensPosition"]
    print("Saved lens_position =", cal["lens_position"])
    with open("calibration.json", "w") as f:
        json.dump(cal, f, indent=2)

print("\nCalibration preview running.")
print("Press Q to quit.")

while True:
    frame = picam2.capture_array()
    h, w, _ = frame.shape

    crop_size = int(cal["crop_size_px"] * preview_w / cal["camera_resolution"][0])
    cx = int(cal["crop_center_x_px"] * preview_w / cal["camera_resolution"][0])
    cy = int(cal["crop_center_y_px"] * preview_h / cal["camera_resolution"][1])

    x1 = cx - crop_size // 2
    y1 = cy - crop_size // 2
    x2 = cx + crop_size // 2
    y2 = cy + crop_size // 2

    # Central trigger strip preview
    strip_h = cal["trigger_strip_height_px"]
    tx1 = w // 4
    tx2 = 3 * w // 4
    cv2.rectangle(frame, (tx1, 0), (tx2, strip_h), (255, 0, 0), 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Calibration Preview", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()
