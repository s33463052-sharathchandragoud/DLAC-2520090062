# VisionPort – Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning

VisionPort is an end-to-end intelligent traffic monitoring and computer vision platform powered by **YOLOv8** and **OpenCV**. It provides automated vehicle detection, multi-class classification, centroid tracking, speed estimation, traffic light status identification, density categorization, and congestion analysis.

---

## 📁 Project Structure

```
Deep Learning /
│
├── .vscode/
│   ├── launch.json                   # VS Code / Antigravity debug launch configurations
│   └── settings.json                 # Python environment configurations
│
├── extracted_frames/                 # Extracted video frames directory (Task 5)
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── visionportsrccode.txt             # Consolidated source code reference
│
├── sample_traffic.jpg                # Sample traffic scene image
├── traffic_sample.mp4                # Sample traffic video stream
├── yolov8n.pt                        # Pre-trained YOLOv8 Nano model weights
│
├── vehicle_detector.py               # Modular detector & centroid tracking engine
├── visionport.py                     # FastAPI web backend & streaming server
├── visionport.html                   # Interactive live monitoring dashboard
│
├── task1_setup_and_verify.py         # Task 1: Environment & library verification
├── task2_capture_traffic_signal.py   # Task 2: Capture traffic signal image
├── task3_read_display_traffic_image.py # Task 3: Read and display traffic image
├── task4_load_play_traffic_video.py  # Task 4: Load and play traffic video
├── task5_extract_video_frames.py     # Task 5: Extract individual video frames
├── task6_detect_cars.py              # Task 6: Detect cars using YOLO
├── task7_detect_multiple_vehicles.py # Task 7: Detect bikes, buses, trucks, cars
├── task8_display_labels_confidence.py# Task 8: Display vehicle names & confidence scores
├── task9_count_vehicles.py           # Task 9: Count vehicles in frame
├── task10_traffic_density.py         # Task 10: Calculate traffic density (Low/Med/High)
├── task11_traffic_congestion.py      # Task 11: Detect junction congestion & alerts
├── task12_track_vehicles.py          # Task 12: Track individual vehicles with IDs
├── task13_estimate_vehicle_speed.py  # Task 13: Approximate vehicle speed calculation
├── task14_traffic_signal_status.py   # Task 14: Identify traffic signal (Red/Yellow/Green)
└── task15_traffic_waiting_time.py    # Task 15: Estimate queue waiting time & report
```

---

## 🚀 Quick Setup & Installation

1. **Activate Python Environment**:
   Ensure Python 3.8+ is installed.

2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Environment**:
   ```bash
   python task1_setup_and_verify.py
   ```

---

## 🚦 Student Tasks & Execution Guide

| Task No. | Task Name | Command | Expected Output |
| :---: | :--- | :--- | :--- |
| **1** | Environment Verification | `python task1_setup_and_verify.py` | Validates OpenCV, PyTorch, YOLOv8, FastAPI |
| **2** | Capture Traffic Signal | `python task2_capture_traffic_signal.py` | Captures and saves `captured_traffic_signal.jpg` |
| **3** | Display Traffic Image | `python task3_read_display_traffic_image.py` | OpenCV display window & metadata analysis |
| **4** | Video Playback | `python task4_load_play_traffic_video.py` | Plays video with frame counter & FPS display |
| **5** | Extract Video Frames | `python task5_extract_video_frames.py` | Saves individual frames into `extracted_frames/` |
| **6** | Detect Cars | `python task6_detect_cars.py` | Detects cars with green bounding boxes |
| **7** | Multi-Vehicle Detection | `python task7_detect_multiple_vehicles.py` | Detects buses, cars, bikes, trucks with color codes |
| **8** | Labels & Confidence | `python task8_display_labels_confidence.py` | Labels with confidence % badges & table |
| **9** | Vehicle Count | `python task9_count_vehicles.py` | Aggregated count overlay HUD |
| **10** | Traffic Density | `python task10_traffic_density.py` | Categorizes into LOW, MEDIUM, or HIGH density |
| **11** | Congestion Alerts | `python task11_traffic_congestion.py` | Triggers alert when vehicle queue exceeds limit |
| **12** | Vehicle Tracking | `python task12_track_vehicles.py` | Persistent tracking IDs & trajectory lines |
| **13** | Speed Estimation | `python task13_estimate_vehicle_speed.py` | Real-time speed calculation in km/h |
| **14** | Traffic Signal Detection | `python task14_traffic_signal_status.py` | Detects active signal light: RED, YELLOW, or GREEN |
| **15** | Waiting Time Estimation | `python task15_traffic_waiting_time.py` | Computes delay in seconds & generates report |

---

## 🌐 Running the VisionPort Web Application

To start the full interactive web application and live inference server:

1. **Start the FastAPI Backend**:
   ```bash
   python visionport.py
   ```
2. **Open the Dashboard**:
   Navigate to [http://localhost:8000](http://localhost:8000) in your web browser.
3. **Upload an Image**:
   Drop any street photo or click **Run Quick Test with sample_traffic.jpg** to see real-time bounding boxes, vehicle breakdown, latency metrics, and density categorization.
