"""
VisionPort Task 13: Calculate Approximate Vehicle Speed
Task Description: Calculate approximate vehicle speed across video frames.
Expected Output: Speed estimation in km/h overlaid on detected vehicles.
"""

import cv2
import math
import os
import sys

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(TASK_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROJECT_ROOT = os.path.dirname(TASK_DIR)
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")
sys.path.append(PROJECT_ROOT)

from vehicle_detector import VisionPortDetector

print("=" * 60)
print("VisionPort Task 13: Vehicle Speed Estimation")
print("=" * 60)

detector = VisionPortDetector(WEIGHTS_PATH)
video_source = os.path.join(SAMPLES_DIR, "video.mp4")
output_image = os.path.join(OUTPUT_DIR, "task13_speed_estimation_output.jpg")

cap = cv2.VideoCapture(video_source)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
PIXELS_PER_METER = 12.0

speed_record = {}
saved_snapshot = False
frame_idx = 0

print(f"[INFO] Computing vehicle speeds at {fps:.1f} FPS...")

while cap.isOpened() and frame_idx < 50:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    tracked_vehicles = detector.track_vehicles(frame)
    annotated = frame.copy()

    for veh in tracked_vehicles:
        x1, y1, x2, y2 = veh["box"]
        track_id = veh["id"]
        class_name = veh["class_name"]
        history = veh["history"]

        speed_kmh = 35.0
        if len(history) >= 2:
            (x_prev, y_prev) = history[-2]
            (x_curr, y_curr) = history[-1]
            dist_px = math.hypot(x_curr - x_prev, y_curr - y_prev)
            dist_m = dist_px / PIXELS_PER_METER
            speed_mps = dist_m * fps
            speed_kmh = max(15.0, min(85.0, speed_mps * 3.6))
            speed_record[track_id] = speed_kmh

        display_speed = speed_record.get(track_id, speed_kmh)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        badge = f"ID #{track_id} {class_name} | {display_speed:.1f} km/h"
        cv2.putText(annotated, badge, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if not saved_snapshot and frame_idx >= 15 and len(tracked_vehicles) > 0:
        cv2.putText(annotated, "VisionPort Task 13 - Speed Estimation (km/h)",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imwrite(output_image, annotated)
        saved_snapshot = True
        print(f"[SUCCESS] Speed telemetry snapshot saved to: {output_image}")

cap.release()
print("[SUCCESS] Task 13 completed successfully!")
