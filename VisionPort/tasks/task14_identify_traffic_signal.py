"""
VisionPort Task 14: Identify Traffic Signal Status (Red, Yellow, Green)
Task Description: Identify traffic signal status using computer vision / color thresholding.
Expected Output: Real-time detection of signal color state (RED, YELLOW, GREEN).
"""

import cv2
import numpy as np
import os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(TASK_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROJECT_ROOT = os.path.dirname(TASK_DIR)
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")

print("=" * 60)
print("VisionPort Task 14: Traffic Signal Status Detection")
print("=" * 60)

video_source = os.path.join(SAMPLES_DIR, "video.mp4")
output_image = os.path.join(OUTPUT_DIR, "task14_traffic_signal_output.jpg")

cap = cv2.VideoCapture(video_source)

def detect_signal_state(frame):
    roi = frame[20:150, 540:620]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red1, upper_red1 = np.array([0, 120, 150]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 150]), np.array([180, 255, 255])
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    lower_yellow, upper_yellow = np.array([15, 120, 150]), np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_green, upper_green = np.array([40, 120, 150]), np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if red_pixels > yellow_pixels and red_pixels > green_pixels and red_pixels > 50:
        return "RED", (0, 0, 255)
    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels and yellow_pixels > 50:
        return "YELLOW", (0, 220, 255)
    elif green_pixels > red_pixels and green_pixels > yellow_pixels and green_pixels > 50:
        return "GREEN", (0, 255, 0)
    else:
        return "UNKNOWN", (180, 180, 180)

saved_snapshot = False
frame_idx = 0
signal_timeline = []

while cap.isOpened() and frame_idx < 100:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    state, color = detect_signal_state(frame)
    signal_timeline.append(state)

    if frame_idx in (25, 65, 95) and not saved_snapshot:
        annotated = frame.copy()
        cv2.rectangle(annotated, (535, 15), (625, 155), color, 3)
        cv2.rectangle(annotated, (20, 20), (380, 85), (20, 20, 20), -1)
        cv2.rectangle(annotated, (20, 20), (380, 85), color, 2)
        cv2.putText(annotated, f"TRAFFIC SIGNAL: {state}", (35, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 3)
        cv2.imwrite(output_image, annotated)
        saved_snapshot = True
        print(f"[SUCCESS] Frame {frame_idx}: Detected Signal State = {state}")

cap.release()
print(f"[INFO] Analyzed {len(signal_timeline)} frames. Transitions detected: {sorted(list(set(signal_timeline)))}")
print("[SUCCESS] Task 14 completed successfully!")
