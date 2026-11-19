"""
VisionPort Task 1: Environment & Library Verification
Expected Output: Working Python Environment Verification
"""

import sys
import importlib
import os
import cv2
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(TASK_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROJECT_ROOT = os.path.dirname(TASK_DIR)

print("=" * 60)
print("VisionPort Task 1: Environment & Library Verification")
print("=" * 60)

print(f"Python Version: {sys.version}")
if sys.version_info < (3, 8):
    print("WARNING: Python 3.8 or higher is recommended.")
else:
    print("[OK] Python version is compatible.\n")

required_libraries = [
    ("cv2", "OpenCV (opencv-python)"),
    ("numpy", "NumPy"),
    ("PIL", "Pillow"),
    ("ultralytics", "Ultralytics YOLOv8"),
    ("torch", "PyTorch"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn")
]

all_passed = True
print("Checking installed packages:")
print("-" * 60)
for module_name, display_name in required_libraries:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "Installed")
        print(f"[PASSED] {display_name:<28} : v{version}")
    except ImportError:
        print(f"[MISSING] {display_name:<28} : NOT INSTALLED")
        all_passed = False

print("-" * 60)
if all_passed:
    print("\n[SUCCESS] All core libraries are installed and ready!")
    test_img = np.zeros((300, 500, 3), dtype=np.uint8)
    test_img[:] = (35, 45, 35)
    cv2.putText(test_img, "VisionPort Task 1", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2)
    cv2.putText(test_img, "OpenCV & Python Environment Verified!", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.circle(test_img, (250, 230), 30, (0, 200, 0), -1)
    cv2.putText(test_img, "OK", (238, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    output_filename = os.path.join(OUTPUT_DIR, "task01_opencv_test.jpg")
    cv2.imwrite(output_filename, test_img)
    print(f"[SAVED] Generated verification image: {output_filename}")
else:
    print("\n[FAILED] Please install missing dependencies with: pip install -r requirements.txt")
