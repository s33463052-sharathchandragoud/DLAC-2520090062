"""
VisionPort Task 7: Detect Multiple Vehicle Types
Task Description: Detect bikes, buses, cars, and trucks using YOLO.
Expected Output: Multiple vehicle detection with category-specific bounding boxes.
"""

import cv2
import os
from ultralytics import YOLO

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TASK_DIR)
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")

print("=" * 60)
print("VisionPort Task 7: Detect Multiple Vehicles (Bikes, Buses, Trucks, Cars)")
print("=" * 60)

model = YOLO(WEIGHTS_PATH)
image_path = os.path.join(SAMPLES_DIR, "intersection.jpg")
if not os.path.exists(image_path):
    image_path = os.path.join(SAMPLES_DIR, "sample_traffic.jpg")

output_image_path = os.path.join(TASK_DIR, "task7_multiple_vehicles_output.jpg")

VEHICLE_CLASSES = {
    1: ("Bicycle", (255, 100, 0)),
    2: ("Car", (0, 255, 0)),
    3: ("Motorcycle", (255, 255, 0)),
    5: ("Bus", (0, 165, 255)),
    7: ("Truck", (0, 0, 255))
}

img = cv2.imread(image_path)
results = model(img, classes=list(VEHICLE_CLASSES.keys()), verbose=False)

annotated = img.copy()
counts = {name: 0 for name, _ in VEHICLE_CLASSES.values()}

for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    if cls_id in VEHICLE_CLASSES:
        name, color = VEHICLE_CLASSES[cls_id]
        counts[name] += 1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

y_pos = 30
cv2.rectangle(annotated, (10, 10), (380, 40 + len(counts) * 25), (20, 20, 20), -1)
cv2.putText(annotated, "VisionPort: Vehicle Breakdown", (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

for name, color in VEHICLE_CLASSES.values():
    y_pos += 25
    cv2.putText(annotated, f"- {name}: {counts[name]}", (30, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

cv2.imwrite(output_image_path, annotated)

print("\nDetection Summary:")
print("-" * 40)
for name, _ in VEHICLE_CLASSES.values():
    print(f"  {name:<15}: {counts[name]}")
print("-" * 40)
print(f"[SUCCESS] Multi-vehicle output saved to: {output_image_path}")
