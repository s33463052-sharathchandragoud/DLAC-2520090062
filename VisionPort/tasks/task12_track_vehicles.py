"""
VisionPort Task 12: Track Individual Vehicles Across Video Frames
Task Description: Track individual vehicles across video frames.
Expected Output: Vehicle tracking IDs assigned and persistent trajectories displayed.
"""

import cv2
import os
import sys

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TASK_DIR)
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")
sys.path.append(PROJECT_ROOT)

from vehicle_detector import VisionPortDetector

print("=" * 60)
print("VisionPort Task 12: Multi-Vehicle Tracking Across Frames")
print("=" * 60)

detector = VisionPortDetector(WEIGHTS_PATH)
video_source = os.path.join(SAMPLES_DIR, "video.mp4")
output_image = os.path.join(TASK_DIR, "task12_tracking_output.jpg")

cap = cv2.VideoCapture(video_source)
saved_snapshot = False
frame_idx = 0

print(f"[INFO] Tracking vehicles across frames in '{video_source}'...")

while cap.isOpened() and frame_idx < 60:
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

        color = (255, 180, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"ID #{track_id} : {class_name}"
        cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        points = veh["history"]
        for i in range(1, len(points)):
            cv2.line(annotated, points[i - 1], points[i], (0, 255, 255), 2)

    if not saved_snapshot and frame_idx >= 15 and len(tracked_vehicles) > 0:
        cv2.putText(annotated, f"VisionPort Task 12 - Active Track IDs: {[v['id'] for v in tracked_vehicles]}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imwrite(output_image, annotated)
        saved_snapshot = True
        print(f"[SUCCESS] Tracking frame with assigned IDs saved to: {output_image}")

cap.release()
print("[SUCCESS] Task 12 completed successfully!")
