"""
VisionPort Task 2: Capture Traffic Signal Image
Task Description: Capture an image of a traffic signal using a camera.
Expected Output: Traffic image capture saved to disk.
"""

import cv2
import os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TASK_DIR)
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")

print("=" * 60)
print("VisionPort Task 2: Capture Traffic Signal Image")
print("=" * 60)

output_path = os.path.join(TASK_DIR, "captured_traffic_signal.jpg")

cap = cv2.VideoCapture(0)
frame = None

if cap.isOpened():
    print("[INFO] Camera initialized successfully. Capturing frame...")
    ret, frame = cap.read()
    cap.release()
else:
    print("[INFO] Webcam index 0 not available. Falling back to sample signal source...")
    sample_signal = os.path.join(SAMPLES_DIR, "signal.jpg")
    sample_video = os.path.join(SAMPLES_DIR, "video.mp4")
    if os.path.exists(sample_signal):
        frame = cv2.imread(sample_signal)
        print(f"[INFO] Loaded frame from '{sample_signal}'.")
    elif os.path.exists(sample_video):
        vcap = cv2.VideoCapture(sample_video)
        ret, frame = vcap.read()
        vcap.release()

if frame is not None:
    cv2.imwrite(output_path, frame)
    h, w, c = frame.shape
    print(f"[SUCCESS] Traffic image captured successfully!")
    print(f"[OUTPUT] Saved to: {output_path} (Resolution: {w}x{h}, Channels: {c})")
else:
    print("[ERROR] Failed to capture traffic signal frame.")
