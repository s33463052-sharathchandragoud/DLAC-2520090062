"""
VisionPort Task 10: Calculate Traffic Density
Task Description: Calculate traffic density based on vehicle count.
Expected Output: Low, medium and high traffic level categorization with visual status indicator.
"""

import cv2
import os
from ultralytics import YOLO

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TASK_DIR)
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")

print("=" * 60)
print("VisionPort Task 10: Traffic Density Calculation")
print("=" * 60)

model = YOLO(WEIGHTS_PATH)
source_image = os.path.join(SAMPLES_DIR, "intersection.jpg")
if not os.path.exists(source_image):
    source_image = os.path.join(SAMPLES_DIR, "sample_traffic.jpg")

output_image = os.path.join(TASK_DIR, "task10_traffic_density_output.jpg")
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]

frame = cv2.imread(source_image)
results = model(frame, classes=VEHICLE_CLASS_IDS, verbose=False)
boxes = results[0].boxes
vehicle_count = len(boxes)

if vehicle_count <= 2:
    density_status = "LOW"
    status_color = (0, 220, 0)
elif vehicle_count <= 6:
    density_status = "MEDIUM"
    status_color = (0, 200, 255)
else:
    density_status = "HIGH"
    status_color = (0, 0, 255)

annotated = results[0].plot()

cv2.rectangle(annotated, (20, 20), (450, 120), (25, 25, 25), -1)
cv2.rectangle(annotated, (20, 20), (450, 120), status_color, 2)
cv2.putText(annotated, "VisionPort Traffic Monitor", (35, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
cv2.putText(annotated, f"Vehicles: {vehicle_count}", (35, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(annotated, f"Density: {density_status}", (210, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 3)

cv2.rectangle(annotated, (35, 95), (430, 105), (60, 60, 60), -1)
fill_width = int(min(1.0, vehicle_count / 10.0) * (430 - 35))
cv2.rectangle(annotated, (35, 95), (35 + fill_width, 105), status_color, -1)

cv2.imwrite(output_image, annotated)

print(f"[INFO] Analyzed image: {source_image}")
print(f"[RESULT] Detected Vehicles : {vehicle_count}")
print(f"[RESULT] Traffic Density   : {density_status}")
print(f"[OUTPUT] Saved density report snapshot: {output_image}")
print("[SUCCESS] Task 10 completed successfully!")
