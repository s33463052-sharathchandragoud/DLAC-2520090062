"""
VisionPort Vehicle Detector Module
Provides YOLOv8 vehicle detection and pure-Python centroid tracking.
"""

import math
import os
import numpy as np
import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(BASE_DIR, "yolov8n.pt")

VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=100):
        self.next_object_id = 1
        self.objects = {}
        self.disappeared = {}
        self.history = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.history[self.next_object_id] = [centroid]
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return []

        input_centroids = np.zeros((len(rects), 2), dtype=int)
        for i, (x1, y1, x2, y2, *_) in enumerate(rects):
            input_centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

        if len(self.objects) == 0:
            tracked = []
            for i in range(len(rects)):
                obj_id = self.register(tuple(input_centroids[i]))
                tracked.append((rects[i][:4], obj_id, rects[i][4], rects[i][5]))
            return tracked

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.zeros((len(object_centroids), len(input_centroids)))
        for i, (ox, oy) in enumerate(object_centroids):
            for j, (ix, iy) in enumerate(input_centroids):
                D[i, j] = math.hypot(ox - ix, oy - iy)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        tracked = []

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            centroid = tuple(input_centroids[col])
            self.objects[obj_id] = centroid
            self.disappeared[obj_id] = 0
            self.history[obj_id].append(centroid)

            tracked.append((rects[col][:4], obj_id, rects[col][4], rects[col][5]))
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            obj_id = self.register(tuple(input_centroids[col]))
            tracked.append((rects[col][:4], obj_id, rects[col][4], rects[col][5]))

        return tracked

class VisionPortDetector:
    def __init__(self, weights_path=DEFAULT_WEIGHTS):
        self.model = YOLO(weights_path)
        self.tracker = CentroidTracker()

    def detect_vehicles(self, image, conf_threshold=0.25):
        results = self.model(image, classes=list(VEHICLE_CLASSES.keys()), conf=conf_threshold, verbose=False)
        boxes = results[0].boxes
        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "box": [x1, y1, x2, y2],
                "class_id": cls_id,
                "class_name": VEHICLE_CLASSES.get(cls_id, "Vehicle"),
                "confidence": conf
            })
        return detections, results[0]

    def track_vehicles(self, frame, conf_threshold=0.25):
        detections, _ = self.detect_vehicles(frame, conf_threshold)
        rects = [[*d["box"], d["class_id"], d["confidence"]] for d in detections]
        tracked = self.tracker.update(rects)
        tracked_vehicles = []
        for box, track_id, cls_id, conf in tracked:
            tracked_vehicles.append({
                "id": track_id,
                "box": box,
                "class_id": cls_id,
                "class_name": VEHICLE_CLASSES.get(cls_id, "Vehicle"),
                "confidence": conf,
                "history": self.tracker.history.get(track_id, [])
            })
        return tracked_vehicles

    @staticmethod
    def calculate_traffic_density(vehicle_count):
        if vehicle_count <= 2:
            return "LOW", (0, 220, 0)
        elif vehicle_count <= 6:
            return "MEDIUM", (0, 200, 255)
        else:
            return "HIGH", (0, 0, 255)
