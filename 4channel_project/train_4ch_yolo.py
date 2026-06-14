"""
train_4ch_yolo.py — Train YOLO v11 with 4-channel event camera input.

Key modification vs standard YOLO:
  The first Conv2d layer is changed from in_channels=3 (RGB)
  to in_channels=4 (our physics channels).

  Everything else — backbone, neck, head, loss, anchors — unchanged.

Usage:
    python train_4ch_yolo.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
from pathlib import Path

from config import (
    DATASET_DIR, RUNS_DIR, RUN_NAME,
    YOLO_MODEL, EPOCHS, IMG_SIZE, BATCH, DEVICE, PATIENCE,
    N_CHANNELS, IMG_W, IMG_H,
    DEBUG_MODE
)


# ── Custom Dataset ────────────────────────────────────────────────────────────

class FourChannelDroneDataset(Dataset):
    """
    Loads .npy channel stacks and YOLO .txt labels.

    Each sample:
      image : torch.Tensor (4, IMG_SIZE, IMG_SIZE) float32 [0,1]
      labels: torch.Tensor (N, 5) — [class, cx, cy, w, h] YOLO format
    """

    def __init__(self, images_dir, labels_dir, img_size=IMG_SIZE):
        self.img_size  = img_size
        self.img_files = sorted(glob.glob(os.path.join(images_dir, '*.png')))
        self.lbl_dir   = labels_dir
        print(f"  Dataset: {len(self.img_files)} samples from {images_dir}")
        if DEBUG_MODE and self.img_files:
            print(f"  [DEBUG] First 3 image paths:")
            for p in self.img_files[:3]:
                print(f"    {p}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, stem + '.txt')

        # Load 4-channel PNG (RGBA) → shape (4, H, W)
        from PIL import Image as PILImage
        img = PILImage.open(img_path)   # mode='RGBA'
        channels = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0  # (4, H, W)

        # Resize to YOLO input size
        channels = self._resize_channels(channels, self.img_size)

        # Load label
        labels = self._load_label(lbl_path)

        return torch.from_numpy(channels), labels

    def _resize_channels(self, channels, size):
        """Resize each channel to (size, size) using PIL."""
        from PIL import Image
        resized = []
        for ch in channels:
            img = Image.fromarray((ch * 255).astype(np.uint8), mode='L')
            img = img.resize((size, size), Image.BILINEAR)
            resized.append(np.array(img).astype(np.float32) / 255.0)
        return np.stack(resized, axis=0)   # (4, size, size)

    def _load_label(self, lbl_path):
        """Load YOLO format label. Returns tensor (N, 5) or empty (0, 5)."""
        if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)

        labels = []
        with open(lbl_path, 'r') as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) == 5:
                    labels.append(vals)

        if not labels:
            return torch.zeros((0, 5), dtype=torch.float32)

        return torch.tensor(labels, dtype=torch.float32)


def collate_fn(batch):
    """Custom collate to handle variable number of labels per image."""
    images, labels = zip(*batch)
    images = torch.stack(images, 0)

    # Add batch index to labels: [batch_idx, class, cx, cy, w, h]
    labeled = []
    for i, lbl in enumerate(labels):
        if lbl.shape[0] > 0:
            batch_col = torch.full((lbl.shape[0], 1), i, dtype=torch.float32)
            labeled.append(torch.cat([batch_col, lbl], dim=1))

    labels_out = torch.cat(labeled, 0) if labeled else torch.zeros((0, 6))
    return images, labels_out


# ── Modify YOLO first layer ───────────────────────────────────────────────────

def patch_yolo_input_channels(model, n_channels=N_CHANNELS):
    """
    Replace the first Conv2d layer to accept n_channels instead of 3.

    Copies existing weights for channel overlap, random-initialises new channels.
    This preserves as much pretrained knowledge as possible.

    Parameters
    ----------
    model      : ultralytics YOLO model
    n_channels : new number of input channels (default 4)

    Returns
    -------
    patched model
    """
    # Find the first Conv2d in the model
    first_conv = None
    first_conv_name = None

    for name, module in model.model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv      = module
            first_conv_name = name
            break

    if first_conv is None:
        print("WARNING: Could not find first Conv2d layer")
        return model

    old_in  = first_conv.in_channels
    old_out = first_conv.out_channels

    if old_in == n_channels:
        print(f"  First layer already has {n_channels} channels — no change needed")
        return model

    print(f"  Patching first conv: {old_in} → {n_channels} input channels")

    # Create new conv with same params but new in_channels
    new_conv = nn.Conv2d(
        in_channels  = n_channels,
        out_channels = old_out,
        kernel_size  = first_conv.kernel_size,
        stride       = first_conv.stride,
        padding      = first_conv.padding,
        bias         = first_conv.bias is not None
    )

    # Copy existing weights for min(old_in, n_channels) channels
    with torch.no_grad():
        min_ch = min(old_in, n_channels)
        new_conv.weight[:, :min_ch, :, :] = first_conv.weight[:, :min_ch, :, :]

        # Initialise extra channels with mean of existing weights
        if n_channels > old_in:
            mean_weight = first_conv.weight.mean(dim=1, keepdim=True)
            for c in range(old_in, n_channels):
                new_conv.weight[:, c:c+1, :, :] = mean_weight

        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)

    # Replace the module in the model
    # Navigate to parent module and replace
    parts = first_conv_name.split('.')
    parent = model.model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_conv)

    return model


# ── Training with Ultralytics API ─────────────────────────────────────────────

def train_with_ultralytics():
    """
    Use Ultralytics YOLO API for training.

    Automatically resumes from last checkpoint if one exists.
    Safe to interrupt and restart — no work is lost.

    Checkpoints saved after every epoch:
      runs/fred_4channel/weights/best.pt  ← best mAP50 so far
      runs/fred_4channel/weights/last.pt  ← most recent epoch
    """
    from ultralytics import YOLO

    # Ultralytics uses its own imread from ultralytics.utils.patches (not cv2.imread directly).
    # base.py sets cv2_flag=IMREAD_COLOR for channels≠1, which strips alpha on RGBA PNGs.
    # Patch both the source function and base.py's local binding to force IMREAD_UNCHANGED.
    import ultralytics.utils.patches as _ul_patches
    import ultralytics.data.base as _ul_base
    import cv2 as _cv2
    _orig_ul_imread = _ul_patches.imread

    def _imread_rgba(filename, flags=_cv2.IMREAD_COLOR):
        return _orig_ul_imread(filename, _cv2.IMREAD_UNCHANGED)

    _ul_patches.imread = _imread_rgba
    _ul_base.imread    = _imread_rgba

    yaml_path = os.path.join(DATASET_DIR, 'dataset.yaml')
    if not os.path.exists(yaml_path):
        print(f"ERROR: dataset.yaml not found at {yaml_path}")
        print("Run dataset_builder.py first")
        return

    # ── Check for existing checkpoint ────────────────────────────────────────
    last_pt = os.path.join(RUNS_DIR, RUN_NAME, 'weights', 'last.pt')
    best_pt = os.path.join(RUNS_DIR, RUN_NAME, 'weights', 'best.pt')

    if os.path.exists(last_pt):
        print(f"\nCheckpoint found: {last_pt}")
        print(f"Resuming training from last checkpoint...")
        model = YOLO(last_pt)
        resume = True
    else:
        print("No checkpoint found — starting fresh training...")
        print(f"Loading base model: {YOLO_MODEL}")
        model = YOLO(YOLO_MODEL)
        print(f"Patching input to {N_CHANNELS} channels...")
        model = patch_yolo_input_channels(model, N_CHANNELS)
        resume = False

    print(f"\nTraining settings:")
    print(f"  Dataset  : {yaml_path}")
    print(f"  Epochs   : {EPOCHS}")
    print(f"  Batch    : {BATCH}")
    print(f"  Device   : {DEVICE}")
    print(f"  Channels : {N_CHANNELS}")
    print(f"  Resume   : {resume}")
    print(f"  Saves to : {os.path.join(RUNS_DIR, RUN_NAME)}")

    if DEBUG_MODE:
        # Load one batch to show tensor shapes before training starts
        try:
            from torch.utils.data import DataLoader
            _ds = FourChannelDroneDataset(
                os.path.join(DATASET_DIR, 'images', 'train'),
                os.path.join(DATASET_DIR, 'labels', 'train'),
            )
            _loader = DataLoader(_ds, batch_size=min(2, len(_ds)),
                                 collate_fn=collate_fn)
            _imgs, _lbls = next(iter(_loader))
            print(f"  [DEBUG] First batch shape: images={tuple(_imgs.shape)}  "
                  f"labels={tuple(_lbls.shape)}")
            print(f"  [DEBUG] Pixel value range: "
                  f"min={_imgs.min():.3f}  max={_imgs.max():.3f}")
        except Exception as e:
            print(f"  [DEBUG] Could not inspect batch: {e}")

    results = model.train(
        data     = yaml_path,
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        device   = DEVICE,
        project  = RUNS_DIR,
        name     = RUN_NAME,
        patience = PATIENCE,
        resume   = resume,     # ← key: tells YOLO to continue from last epoch
        save     = True,       # save best.pt and last.pt after every epoch
        save_period = 10,      # also save every 10 epochs as extra backup
        plots    = True,
        val      = True,
        # Augmentation — no colour shifts (event frames have no colour).
        # hsv=0 skips RandomHSV entirely (avoids cv2.cvtColor BGR2HSV on 4-ch input).
        # mosaic is channel-aware (uses labels["img"].shape[2]) — safe to leave on.
        hsv_h    = 0.0,
        hsv_s    = 0.0,
        hsv_v    = 0.0,
        fliplr   = 0.5,
        mosaic   = 0.5,
    )

    print(f"\nTraining complete!")
    print(f"  Best model : {best_pt}")
    print(f"  Last model : {last_pt}")
    print(f"  mAP50      : {results.results_dict.get('metrics/mAP50(B)', 0)*100:.2f}%")
    print(f"  Paper base : 87.68%")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("FRED 4-Channel YOLO Training")
    print("=" * 60)

    # Verify dataset exists
    yaml_path = os.path.join(DATASET_DIR, 'dataset.yaml')
    if not os.path.exists(yaml_path):
        print(f"\nDataset not found at: {DATASET_DIR}")
        print("Run first:  python dataset_builder.py")
        sys.exit(1)

    # Check device
    if DEVICE != "cpu":
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            print("Training will be slow. Consider DEVICE='cpu' in config.py")

    train_with_ultralytics()
