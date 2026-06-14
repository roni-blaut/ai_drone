"""
FRED Drone Detection - Step 3: Train YOLO v11
=============================================
Trains YOLO v11 on FRED event frames for drone detection.
Per the paper: YOLO on EVENT frames achieves 87.68 mAP50 — best of all methods.

Install requirements:
    pip install ultralytics

Run after Step 2 (annotation conversion).
"""

from ultralytics import YOLO
import os

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_YAML = "./fred_yolo/dataset.yaml"  # Created in Step 2
MODEL        = "yolo11n.pt"               # Pretrained YOLO v11 nano (small/fast)
                                           # Options: yolo11n / yolo11s / yolo11m / yolo11l / yolo11x
EPOCHS       = 100
IMG_SIZE     = 640
BATCH        = 16
DEVICE       = 0           # GPU id, or "cpu" if no GPU
PROJECT      = "./runs"    # Where results are saved
RUN_NAME     = "fred_drone_event"


# ── Train ────────────────────────────────────────────────────────────────────

def train():
    # Load YOLO v11 with ImageNet pretrained weights
    model = YOLO(MODEL)

    print(f"Training YOLO v11 on FRED event frames...")
    print(f"Dataset: {DATASET_YAML}")
    print(f"Epochs:  {EPOCHS}")
    print(f"Batch:   {BATCH}")

    results = model.train(
        data      = DATASET_YAML,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH,
        device    = DEVICE,
        project   = PROJECT,
        name      = RUN_NAME,
        # Recommended settings from the paper's evaluation protocol
        # Detection assessed at 33ms intervals (30 FPS)
        patience  = 20,          # Stop early if no improvement after 20 epochs
        save      = True,        # Save best + last checkpoints
        plots     = True,        # Save training plots
        val       = True,        # Run validation each epoch
        # Augmentation (helps with challenging conditions in FRED)
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        flipud    = 0.0,
        fliplr    = 0.5,
        mosaic    = 1.0,
    )

    print(f"\nTraining complete!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print(f"Validation mAP50:   {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    return results


# ── Validate ─────────────────────────────────────────────────────────────────

def validate(model_path="./runs/fred_drone_event/weights/best.pt"):
    """Evaluate the trained model on the test split."""
    model = YOLO(model_path)
    metrics = model.val(
        data   = DATASET_YAML,
        imgsz  = IMG_SIZE,
        device = DEVICE,
        split  = "test",
    )
    print(f"\nTest Results:")
    print(f"  mAP50    = {metrics.box.map50:.4f}")
    print(f"  mAP50:95 = {metrics.box.map:.4f}")
    print(f"  Precision = {metrics.box.mp:.4f}")
    print(f"  Recall    = {metrics.box.mr:.4f}")
    return metrics


if __name__ == "__main__":
    results = train()
    validate()
