"""
FRED Drone Detection - Step 4: Run Detection (Inference)
=========================================================
Use your trained YOLO model to detect drones on new event frames or video.

Run after Step 3 (training).
"""

from ultralytics import YOLO
import cv2
import os

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH   = "./runs/fred_drone_event/weights/best.pt"
CONF_THRESH  = 0.25    # Confidence threshold (lower = more detections)
IOU_THRESH   = 0.45    # NMS IoU threshold
IMG_SIZE     = 640
DEVICE       = 0       # GPU id, or "cpu"


# ── Detect on a single image ─────────────────────────────────────────────────

def detect_image(image_path, save_dir="./detections"):
    """Run detection on one event frame image and save the result."""
    os.makedirs(save_dir, exist_ok=True)
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source = image_path,
        conf   = CONF_THRESH,
        iou    = IOU_THRESH,
        imgsz  = IMG_SIZE,
        device = DEVICE,
        save   = True,
        project= save_dir,
        name   = "image_results",
    )

    for r in results:
        boxes = r.boxes
        print(f"\nDetected {len(boxes)} drone(s) in: {image_path}")
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            print(f"  Drone {i+1}: bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]  confidence={conf:.2f}")
    return results


# ── Detect on a video / folder of frames ─────────────────────────────────────

def detect_video(source, save_dir="./detections"):
    """
    source can be:
      - path to a .mp4 / .avi video file
      - path to a folder of PNG frames
      - 0 for webcam (live)
    """
    os.makedirs(save_dir, exist_ok=True)
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source  = source,
        conf    = CONF_THRESH,
        iou     = IOU_THRESH,
        imgsz   = IMG_SIZE,
        device  = DEVICE,
        stream  = True,    # Memory-efficient for long videos
        save    = True,
        project = save_dir,
        name    = "video_results",
    )

    frame_idx = 0
    for r in results:
        n_drones = len(r.boxes)
        print(f"Frame {frame_idx:05d}: {n_drones} drone(s) detected")
        frame_idx += 1


# ── Live detection with OpenCV display ───────────────────────────────────────

def detect_live(source=0):
    """
    Show real-time detection with bounding boxes drawn on screen.
    source=0 for webcam, or a video file path.
    """
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(source)

    print("Press Q to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO on this frame
        results = model.predict(
            source = frame,
            conf   = CONF_THRESH,
            iou    = IOU_THRESH,
            imgsz  = IMG_SIZE,
            device = DEVICE,
            verbose= False,
        )

        # Draw boxes on the frame
        annotated = results[0].plot()
        cv2.imshow("FRED Drone Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Batch evaluation: compute mAP on a folder ─────────────────────────────────

def evaluate(images_dir, labels_dir):
    """
    Evaluate detection accuracy (mAP50, mAP50:95) on a folder of images
    where ground-truth YOLO labels exist in labels_dir.
    """
    model   = YOLO(MODEL_PATH)
    metrics = model.val(
        data   = "./fred_yolo/dataset.yaml",
        split  = "test",
        imgsz  = IMG_SIZE,
        conf   = CONF_THRESH,
        iou    = IOU_THRESH,
        device = DEVICE,
    )
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50:95: {metrics.box.map:.4f}")
    return metrics


# ── Main: choose your mode ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python fred_step4_detect.py image  <path_to_event_frame.png>")
        print("  python fred_step4_detect.py video  <path_to_video.mp4>")
        print("  python fred_step4_detect.py live")
        print("  python fred_step4_detect.py eval")
    else:
        mode = sys.argv[1]
        if mode == "image":
            detect_image(sys.argv[2])
        elif mode == "video":
            detect_video(sys.argv[2])
        elif mode == "live":
            detect_live(source=0)
        elif mode == "eval":
            evaluate(
                images_dir = "./fred_yolo/images/test",
                labels_dir = "./fred_yolo/labels/test",
            )
