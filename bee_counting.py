import csv
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ocsort.ocsort import OCSort


VIDEO_DIR = "data/videos"
WEIGHTS = "checkpoints/yolov8s_bees.pt"
OUTPUT_CSV = "data/counts.csv"

COUNTING_LINE_Y = 0.55
POLY_DEGREE = 3
MIN_TRAJ_FRAMES = 5
DET_THRESH = 0.45
MAX_AGE = 30
MIN_HITS = 3
IOU_THRESHOLD = 0.3


def smooth_trajectory(centroids):
    frames = np.arange(len(centroids))
    ys = np.array([c[1] for c in centroids])
    coeffs = np.polyfit(frames, ys, POLY_DEGREE)
    return np.polyval(coeffs, frames)


def classify_trajectory(centroids, line_y):
    if len(centroids) < MIN_TRAJ_FRAMES:
        return "pass_through"

    smoothed_y = smooth_trajectory(centroids)

    crossed = any(
        (smoothed_y[i] < line_y) != (smoothed_y[i + 1] < line_y)
        for i in range(len(smoothed_y) - 1)
    )
    if not crossed:
        return "pass_through"

    if smoothed_y[0] < line_y and smoothed_y[-1] >= line_y:
        return "entrance"
    elif smoothed_y[0] >= line_y and smoothed_y[-1] < line_y:
        return "exit"
    return "pass_through"


def process_video(video_path, detector, tracker):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = frame_height * COUNTING_LINE_Y
    trajectories = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector(frame, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        dets = np.hstack([boxes, scores[:, None]])

        tracks = tracker.update(dets, frame) if len(dets) > 0 else []

        for track in tracks:
            x1, y1, x2, y2, track_id = track[:5]
            track_id = int(track_id)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            trajectories.setdefault(track_id, []).append((cx, cy))

    cap.release()

    counts = {"entrance": 0, "exit": 0, "pass_through": 0}
    for centroids in trajectories.values():
        counts[classify_trajectory(centroids, line_y)] += 1

    return counts


if __name__ == "__main__":
    detector = YOLO(WEIGHTS)
    tracker = OCSort(det_thresh=DET_THRESH, max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)

    rows = []
    for video_file in sorted(Path(VIDEO_DIR).glob("*.mp4")):
        counts = process_video(str(video_file), detector, tracker)
        rows.append({"file": video_file.name, **counts})
        print(f"{video_file.name}: entrance={counts['entrance']}  exit={counts['exit']}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "entrance", "exit", "pass_through"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} records to {OUTPUT_CSV}")
