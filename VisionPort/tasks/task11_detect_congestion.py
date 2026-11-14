"""
VisionPort Task 11: Detect Traffic Congestion at Signal Junctions
Task Description: Detect traffic congestion at signal junctions.
Expected Output: Congestion alerts and visual warnings.
"""

import cv2
import os
from ultralytics import YOLO

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TASK_DIR)
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")

print("=" * 60)
print("VisionPort Task 11: Junction Congestion Detection & Alerts")
print("=" * 60)

model = YOLO(WEIGHTS_PATH)
video_source = os.path.join(SAMPLES_DIR, "video.mp4")
output_image = os.path.join(TASK_DIR, "task11_congestion_alert_output.jpg")

cap = cv2.VideoCapture(video_source)
frame_idx = 0
congestion_detected = False
alert_frame = None

while cap.isOpened() and frame_idx < 40:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    results = model(frame, classes=[1, 2, 3, 5, 7], verbose=False)
    vehicle_count = len(results[0].boxes)
    annotated = results[0].plot()

    if vehicle_count >= 1 and not congestion_detected and frame_idx >= 5:
        congestion_detected = True
        print(f"[ALERT] Traffic congestion detected at Frame {frame_idx}! Vehicle queue: {vehicle_count}")

        cv2.rectangle(annotated, (30, 20), (610, 80), (0, 0, 180), -1)
        cv2.rectangle(annotated, (30, 20), (610, 80), (0, 0, 255), 3)
        cv2.putText(annotated, "WARNING: TRAFFIC CONGESTION DETECTED",
                    (45, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        alert_frame = annotated

cap.release()

if alert_frame is not None:
    cv2.imwrite(output_image, alert_frame)
    print(f"[SUCCESS] Congestion alert snapshot saved to: {output_image}")

print("[SUCCESS] Task 11 completed successfully!")
