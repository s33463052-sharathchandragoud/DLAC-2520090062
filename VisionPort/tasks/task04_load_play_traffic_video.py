"""
VisionPort Task 4: Load and Play Traffic Video
Task Description: Load and play a traffic video using OpenCV.
Expected Output: Video playback with FPS and frame counter.
"""

import cv2
import os
import time

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TASK_DIR)
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")

print("=" * 60)
print("VisionPort Task 4: Load and Play Traffic Video")
print("=" * 60)

video_path = os.path.join(SAMPLES_DIR, "video.mp4")
if not os.path.exists(video_path):
    video_path = os.path.join(SAMPLES_DIR, "traffic_sample.mp4")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Could not open video file '{video_path}'.")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_sec = total_frames / fps if fps > 0 else 0

print(f"[INFO] Video loaded successfully: {video_path}")
print(f"       Resolution   : {width} x {height}")
print(f"       Total Frames : {total_frames}")
print(f"       Playback FPS : {fps:.1f}")
print(f"       Duration     : {duration_sec:.2f} seconds")
print("-" * 60)

frame_idx = 0
start_time = time.time()

while cap.isOpened() and frame_idx < 100:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    cv2.putText(
        frame,
        f"Frame: {frame_idx}/{total_frames} | FPS: {fps:.1f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    try:
        cv2.imshow("Task 4 - Traffic Video Playback", frame)
        delay = max(1, int(1000 / fps))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    except cv2.error:
        pass

cap.release()
cv2.destroyAllWindows()
elapsed = time.time() - start_time
print(f"[SUCCESS] Played {frame_idx} frames in {elapsed:.2f} seconds.")
