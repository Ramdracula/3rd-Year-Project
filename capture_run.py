import cv2
import json
import time
import csv
import argparse
import os
import numpy as np
from datetime import datetime
from picamera2 import Picamera2
from libcamera import controls

parser = argparse.ArgumentParser()
parser.add_argument("--run-config", required=True)
args = parser.parse_args()

with open("calibration.json") as f:
    cal = json.load(f)

with open(args.run_config) as f:
    run = json.load(f)

os.makedirs("dataset", exist_ok=True)
csv_path = "dataset/index.csv"
csv_exists = os.path.isfile(csv_path)

picam2 = Picamera2()

main_size = tuple(cal["camera_resolution"])
preview_size = tuple(cal["preview_resolution"])

config = picam2.create_preview_configuration(
    main={"size": main_size, "format": "RGB888"},
    lores={"size": preview_size, "format": "YUV420"}
)

picam2.configure(config)
picam2.start()
time.sleep(1)

# Lock focus and exposure
picam2.set_controls({
    "AfMode": controls.AfModeEnum.Manual,
    "LensPosition": cal["lens_position"],
    "AeEnable": False,
    "ExposureTime": cal["shutter_us"],
    "AnalogueGain": cal["gain"]
})

baseline = cal["trigger_baseline"]
trigger_delta = cal["trigger_delta"]
rearm_delta = cal["rearm_delta"]
strip_height = cal["trigger_strip_height_px"]
delay = cal["trigger_delay_s"]
cooldown = cal["cooldown_s"]

crop_size = cal["crop_size_px"]
cx = cal["crop_center_x_px"]
cy = cal["crop_center_y_px"]

# --- Pattern ID ---
pattern_id = run.get("pattern_id", "UNKNOWN")

# --- Persistent indexing ---
existing_files = [
    f for f in os.listdir("dataset")
    if f.startswith(pattern_id) and f.endswith(".jpg")
]

if existing_files:
    indices = []
    for f in existing_files:
        try:
            parts = f.split("_")
            idx = int(parts[1])
            indices.append(idx)
        except:
            continue
    capture_count = max(indices) + 1 if indices else 0
else:
    capture_count = 0

# --- Other counters ---
trigger_count = 0
last_capture = 0
armed = True
last_saved_crop = None
last_saved_name = None
skip_first_capture = True

fieldnames = list(run.keys()) + [
    "image_id", "timestamp", "image_file", "trigger_brightness"
]

if not csv_exists:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

print("\nSystem ready")
print(f"Starting index: {capture_count}")
print("First triggered image will be discarded as calibration.")
print("Press Q in preview window to stop\n")

while True:
    lores = picam2.capture_array("lores")

    preview_h, preview_w = preview_size[1], preview_size[0]
    gray = lores[:preview_h, :preview_w]

    h, w = gray.shape

    tx1 = w // 4
    tx2 = 3 * w // 4
    trigger_zone = gray[0:strip_height, tx1:tx2]

    brightness = float(np.mean(trigger_zone))
    status = "WAIT"

    # Trigger on BRIGHTENING
    if armed and brightness > (baseline + trigger_delta) and (time.time() - last_capture) > cooldown:
        status = "TRIGGER"
        armed = False

        if delay > 0:
            time.sleep(delay)

        full = picam2.capture_array("main")

        crop = full[
            cy - crop_size // 2: cy + crop_size // 2,
            cx - crop_size // 2: cx + crop_size // 2
        ]

        trigger_count += 1
        last_capture = time.time()

        # First triggered image is calibration only: discard it
        if skip_first_capture:
            skip_first_capture = False
            print("Calibration capture taken and discarded.")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_id = f"{pattern_id}_{capture_count:03d}_{timestamp}"
            image_file = f"{image_id}.jpg"

            cv2.imwrite(
                f"dataset/{image_file}",
                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            )

            metadata = run.copy()
            metadata["image_id"] = image_id
            metadata["timestamp"] = timestamp
            metadata["image_file"] = image_file
            metadata["trigger_brightness"] = brightness

            with open(f"dataset/{image_id}.json", "w") as f:
                json.dump(metadata, f, indent=4)

            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(metadata)

            capture_count += 1
            last_saved_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            last_saved_name = image_file

            print("Captured:", image_file)

    elif not armed:
        status = "HOLD"
        if brightness < (baseline + rearm_delta):
            armed = True
            status = "REARMED"

    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(display, (tx1, 0), (tx2, strip_height), (255, 0, 0), 2)

    crop_size_preview = int(crop_size * w / cal["camera_resolution"][0])
    cx_preview = int(cx * w / cal["camera_resolution"][0])
    cy_preview = int(cy * h / cal["camera_resolution"][1])

    x1 = cx_preview - crop_size_preview // 2
    y1 = cy_preview - crop_size_preview // 2
    x2 = cx_preview + crop_size_preview // 2
    y2 = cy_preview + crop_size_preview // 2

    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(display, f"Pattern: {pattern_id}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display, f"Saved: {capture_count}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display, f"Triggers: {trigger_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display, f"Status: {status}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display, f"Brightness: {brightness:.1f}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display, f"Trigger if > {baseline + trigger_delta:.1f}", (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Preview", display)

    if last_saved_crop is not None:
        latest = cv2.resize(last_saved_crop, (400, 400))
        cv2.putText(latest, last_saved_name, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow("Last Capture", latest)

    if cv2.waitKey(1) == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
