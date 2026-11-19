"""
VisionPort Task 8: Display Labels and Confidence Scores
Task Description: Display vehicle names and confidence scores.
Expected Output: Labelled detection output with high-precision confidence badges.
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
print("VisionPort Task 8: Display Labels and Confidence Scores")
print("=" * 60)

model = YOLO(WEIGHTS_PATH)
image_path = os.path.join(SAMPLES_DIR, "intersection.jpg")
if not os.path.exists(image_path):
    image_path = os.path.join(SAMPLES_DIR, "sample_traffic.jpg")

output_path = os.path.join(OUTPUT_DIR, "task08_labelled_confidence_output.jpg")

img = cv2.imread(image_path)
results = model(img, verbose=False)
boxes = results[0].boxes
names = model.names

annotated = img.copy()

print(f"\n{'Index':<6} | {'Class Name':<16} | {'Confidence':<12} | {'Bounding Box (xyxy)'}")
print("-" * 65)

for idx, box in enumerate(boxes):
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    class_name = names.get(cls_id, f"Class {cls_id}")
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    print(f"{idx + 1:<6} | {class_name:<16} | {conf * 100:>6.2f}%     | [{x1}, {y1}, {x2}, {y2}]")
    
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
    badge_text = f"{class_name}: {conf * 100:.1f}%"
    (text_w, text_h), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    badge_y1 = max(0, y1 - text_h - 10)
    cv2.rectangle(annotated, (x1, badge_y1), (x1 + text_w + 10, y1), (0, 220, 255), -1)
    cv2.putText(annotated, badge_text, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

print("-" * 65)
cv2.imwrite(output_path, annotated)
print(f"[SUCCESS] Labelled detections with confidence scores saved to: {output_path}")
