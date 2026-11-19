"""
VisionPort Task 5: Extract Frames From Traffic Video
Task Description: Extract individual frames from traffic videos.
Expected Output: Video frames saved as JPEG images inside 'extracted_frames/'.
"""

import cv2
import os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TASK_DIR)
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
output_dir = os.path.join(PROJECT_ROOT, "extracted_frames")
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("VisionPort Task 5: Extract Video Frames")
print("=" * 60)

video_path = os.path.join(SAMPLES_DIR, "video.mp4")
if not os.path.exists(video_path):
    video_path = os.path.join(SAMPLES_DIR, "traffic_sample.mp4")

cap = cv2.VideoCapture(video_path)
frame_interval = 5
saved_count = 0
frame_idx = 0

print(f"[INFO] Extracting frames from '{video_path}'...")

while cap.isOpened() and saved_count < 25:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        filename = f"frame_{saved_count + 1:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        saved_count += 1

    frame_idx += 1

cap.release()
print(f"[SUCCESS] Extracted and saved {saved_count} frames into '{output_dir}'.")
