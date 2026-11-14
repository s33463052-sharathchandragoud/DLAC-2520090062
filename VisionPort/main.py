"""
VisionPort: Local FastAPI Evaluation Server
Provides live deep learning inference, static media serving,
and interactive evaluation dashboard.
"""

import io
import time
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
WEIGHTS_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

app = FastAPI(
    title="VisionPort Evaluation Server",
    description="Intelligent Vehicle Detection and Traffic Monitoring API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount samples directory for static access
if os.path.exists(SAMPLES_DIR):
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

print(f"[INFO] Loading YOLO model from '{WEIGHTS_PATH}'...")
model = YOLO(WEIGHTS_PATH)

# COCO Vehicle Classes: 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

def compute_density(count: int):
    if count <= 2:
        return "LOW", (0, 220, 0)
    elif count <= 6:
        return "MEDIUM", (0, 200, 255)
    else:
        return "HIGH", (0, 0, 255)

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the evaluation dashboard."""
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>VisionPort Evaluation Server Active</h1>")

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """Inference endpoint streaming annotated JPEG image."""
    t0 = time.time()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    results = model(img, classes=list(VEHICLE_CLASSES.keys()), verbose=False)
    boxes = results[0].boxes
    vehicle_count = len(boxes)
    density, color = compute_density(vehicle_count)
    latency_ms = round((time.time() - t0) * 1000, 1)

    annotated = results[0].plot()

    # Visual HUD banner
    cv2.rectangle(annotated, (15, 15), (420, 95), (20, 20, 20), -1)
    cv2.rectangle(annotated, (15, 15), (420, 95), color, 2)
    cv2.putText(annotated, "VisionPort Traffic Monitor", (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(annotated, f"Count: {vehicle_count} | Density: {density}", (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    _, encoded = cv2.imencode(".jpg", annotated)
    headers = {
        "X-Vehicle-Count": str(vehicle_count),
        "X-Density-Level": density,
        "X-Latency-Ms": str(latency_ms)
    }

    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type="image/jpeg",
        headers=headers
    )

@app.post("/api/detect")
async def detect_api(file: UploadFile = File(...)):
    """API endpoint returning structured detection JSON."""
    t0 = time.time()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    results = model(img, classes=list(VEHICLE_CLASSES.keys()), verbose=False)
    boxes = results[0].boxes
    breakdown = {name: 0 for name in VEHICLE_CLASSES.values()}
    detections = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = VEHICLE_CLASSES.get(cls_id, "Vehicle")
        breakdown[name] = breakdown.get(name, 0) + 1
        detections.append({
            "class_name": name,
            "confidence": round(conf, 4),
            "box": [x1, y1, x2, y2]
        })

    total_vehicles = len(detections)
    density, _ = compute_density(total_vehicles)
    latency_ms = round((time.time() - t0) * 1000, 1)

    return {
        "status": "success",
        "total_vehicles": total_vehicles,
        "traffic_density": density,
        "latency_ms": latency_ms,
        "breakdown": breakdown,
        "detections": detections
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("VisionPort Evaluation Server starting at http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
