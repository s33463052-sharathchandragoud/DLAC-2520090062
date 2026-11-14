"""
VisionPort Automated Batch Runner for All 15 Student Tasks
Executes Task 1 through Task 15 sequentially, reporting individual execution
status, runtimes, and a final batch evaluation summary.
"""

import os
import sys
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(BASE_DIR, "tasks")

TASK_SCRIPTS = [
    ("Task 1", "Environment Verification", "task1_setup_and_verify.py"),
    ("Task 2", "Capture Traffic Signal", "task2_capture_traffic_signal.py"),
    ("Task 3", "Display Traffic Image", "task3_read_display_traffic_image.py"),
    ("Task 4", "Play Traffic Video", "task4_load_play_traffic_video.py"),
    ("Task 5", "Extract Video Frames", "task5_extract_video_frames.py"),
    ("Task 6", "Detect Cars (YOLO)", "task6_detect_cars.py"),
    ("Task 7", "Detect Multiple Vehicles", "task7_detect_multiple_vehicles.py"),
    ("Task 8", "Labels & Confidence Scores", "task8_display_labels_confidence.py"),
    ("Task 9", "Count Vehicles in Frame", "task9_count_vehicles.py"),
    ("Task 10", "Traffic Density Metric", "task10_traffic_density.py"),
    ("Task 11", "Congestion Alerts", "task11_detect_congestion.py"),
    ("Task 12", "Vehicle Tracking IDs", "task12_track_vehicles.py"),
    ("Task 13", "Speed Estimation", "task13_calculate_vehicle_speed.py"),
    ("Task 14", "Signal State Detection", "task14_identify_traffic_signal.py"),
    ("Task 15", "Waiting Time Report", "task15_estimate_waiting_time.py")
]

def main():
    print("=" * 70)
    print(" VISIONPORT: AUTOMATED BATCH EVALUATION RUNNER (TASKS 1 - 15) ")
    print("=" * 70)
    print(f"Project Directory: {BASE_DIR}\n")

    results = []
    total_start = time.time()

    for tag, name, script_name in TASK_SCRIPTS:
        script_path = os.path.join(TASKS_DIR, script_name)
        if not os.path.exists(script_path):
            print(f"[-] {tag:<8} | {name:<30} | MISSING FILE: {script_name}")
            results.append((tag, name, "MISSING", 0.0))
            continue

        print(f"[*] Running {tag}: {name} ({script_name})...", end="", flush=True)
        t0 = time.time()
        
        proc = subprocess.run(
            [sys.executable, script_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - t0

        if proc.returncode == 0:
            print(f" -> [PASSED] ({elapsed:.2f}s)")
            results.append((tag, name, "PASSED", elapsed))
        else:
            print(f" -> [FAILED] ({elapsed:.2f}s)")
            print(f"\n--- Error Trace ({script_name}) ---")
            print(proc.stderr[:400] if proc.stderr else proc.stdout[:400])
            print("----------------------------------\n")
            results.append((tag, name, "FAILED", elapsed))

    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print(" BATCH RUN SUMMARY TABLE ")
    print("=" * 70)
    print(f"{'Task':<9} | {'Description':<32} | {'Status':<10} | {'Time'}")
    print("-" * 70)
    passed_count = 0
    for tag, name, status, el in results:
        if status == "PASSED":
            passed_count += 1
        print(f"{tag:<9} | {name:<32} | [{status:<6}] | {el:.2f}s")
    print("-" * 70)
    print(f"Total Completed: {passed_count}/{len(TASK_SCRIPTS)} passed in {total_time:.2f}s.")
    print("=" * 70)

    return 0 if passed_count == len(TASK_SCRIPTS) else 1

if __name__ == "__main__":
    sys.exit(main())
