"""
VisionPort Task 6: Detect Cars in Traffic Using YOLO
Task Description: Detect cars in a traffic video/image using YOLO.
Expected Output: Car detection with bounding boxes and output snapshot.
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
print("VisionPort Task 6: Detect Cars Using YOLOv8")
print("=" * 60)

model = YOLO(WEIGHTS_PATH)
image_path = os.path.join(SAMPLES_DIR, "intersection.jpg")
if not os.path.exists(image_path):
    image_path = os.path.join(SAMPLES_DIR, "sample_traffic.jpg")

output_image_path = os.path.join(OUTPUT_DIR, "task6_car_detection_output.jpg")

# Class 2 in COCO is 'car'
CAR_CLASS_ID = 2

img = cv2.imread(image_path)
results = model(img, classes=[CAR_CLASS_ID], verbose=False)

annotated = img.copy()
car_count = 0

for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    car_count += 1
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"Car {conf:.2f}"
    cv2.putText(annotated, label, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.putText(annotated, f"VisionPort Task 6 - Cars Detected: {car_count}",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imwrite(output_image_path, annotated)
print(f"[SUCCESS] Cars detected: {car_count}")
print(f"[OUTPUT] Saved detection output to: {output_image_path}")
