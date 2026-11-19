"""
VisionPort Task 9: Count Vehicles in a Video Frame
Task Description: Count the number of vehicles in a video frame.
Expected Output: Live vehicle count displayed on frame and summary log.
"""

import cv2
import os
from ultralytics import YOLO

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(TASK_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROJECT_ROOT = os.path.dirname(TASK_DIR)
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")

print("=" * 60)
print("VisionPort Task 9: Vehicle Count in Video Frame")
print("=" * 60)

model = YOLO(WEIGHTS_PATH)
source_image = os.path.join(SAMPLES_DIR, "intersection.jpg")
if not os.path.exists(source_image):
    source_image = os.path.join(SAMPLES_DIR, "sample_traffic.jpg")

output_image = os.path.join(OUTPUT_DIR, "task9_vehicle_count_output.jpg")
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]

frame = cv2.imread(source_image)
results = model(frame, classes=VEHICLE_CLASS_IDS, verbose=False)
boxes = results[0].boxes
vehicle_count = len(boxes)

annotated = frame.copy()

for box in boxes:
    cls_id = int(box.cls[0])
    name = model.names[cls_id].title()
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 120), 2)
    cv2.putText(annotated, f"{name}", (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)

overlay = annotated.copy()
cv2.rectangle(overlay, (20, 20), (360, 90), (10, 10, 10), -1)
cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

cv2.rectangle(annotated, (20, 20), (360, 90), (0, 255, 120), 2)
cv2.putText(annotated, f"VEHICLE COUNT: {vehicle_count}", (35, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 120), 3)

cv2.imwrite(output_image, annotated)

print(f"[INFO] Analyzed image: {source_image}")
print(f"[RESULT] Total Vehicles Detected: {vehicle_count}")
print(f"[OUTPUT] Saved count visualization to: {output_image}")
print("[SUCCESS] Task 9 completed successfully!")
