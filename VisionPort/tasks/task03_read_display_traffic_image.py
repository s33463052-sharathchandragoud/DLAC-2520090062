"""
VisionPort Task 3: Read and Display Traffic Images
Task Description: Read and display traffic images using OpenCV.
Expected Output: Image display window with traffic image and metadata inspection.
"""

import cv2
import os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(TASK_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROJECT_ROOT = os.path.dirname(TASK_DIR)
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")

print("=" * 60)
print("VisionPort Task 3: Read and Display Traffic Images")
print("=" * 60)

image_path = os.path.join(SAMPLES_DIR, "intersection.jpg")
if not os.path.exists(image_path):
    image_path = os.path.join(SAMPLES_DIR, "sample_traffic.jpg")

image = cv2.imread(image_path)
if image is None:
    print(f"[ERROR] Failed to load image from '{image_path}'.")
    exit(1)

height, width, channels = image.shape
print(f"[INFO] Image loaded successfully: {image_path}")
print(f"       Resolution : {width} x {height}")
print(f"       Channels   : {channels} (BGR format)")
print(f"       Data Type  : {image.dtype}")

display_img = image.copy()
cv2.rectangle(display_img, (10, 10), (width - 10, 50), (20, 20, 20), -1)
cv2.putText(
    display_img,
    f"VisionPort: Traffic Scene ({width}x{height})",
    (20, 38),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 200),
    2
)

output_path = os.path.join(OUTPUT_DIR, "task03_displayed_traffic_image.jpg")
cv2.imwrite(output_path, display_img)
print(f"[OUTPUT] Verification image saved to: {output_path}")

try:
    cv2.imshow("Task 3 - Traffic Image Viewer", display_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
except cv2.error:
    pass

print("[SUCCESS] Task 3 completed successfully!")
