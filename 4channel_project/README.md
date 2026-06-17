# FRED 4-Channel Drone Detection

Physics-driven multi-channel event camera input for improved drone detection.
Extends the FRED paper (87.68 mAP50) with 4 physically motivated channels.

## Project Concept

The FRED paper feeds YOLO one accumulated 33ms frame — discarding polarity,
temporal structure, and motor frequency. This project replaces that with:

| Channel | Content | Physical meaning |
|---------|---------|-----------------|
| 1 | Positive polarity events | Leading edge — where drone is going |
| 2 | Negative polarity events | Trailing edge — where drone came from |
| 3 | Rotor frequency map | Spinning motor signature (unique to drones) |
| 4 | Time surface | Most recent activity per pixel |

No channel is redundant. Each carries independent physical information.

## File Structure

```
4channel_project/
├── config.py           ← All settings; calls init_sequence() on import
├── zip_utils.py        ← Transparent zip/folder access (seq_glob, seq_imread…)
├── evt3_reader.py      ← Parse Prophesee EVT3 binary (zip-aware via BytesIO)
├── filters.py          ← Refractory + BAF noise filters
├── channels.py         ← Generate 4 channels from filtered events
├── dataset_builder.py  ← Build YOLO dataset (multi-sequence from splits.yaml)
├── make_catalog.py     ← Scan data_from_fred/*.zip → write catalog.yaml
├── train_4ch_yolo.py   ← Train YOLO with modified 4-channel input
├── evaluate.py         ← Evaluate and compare vs paper baseline
├── sync_check.py       ← Verify event↔RGB sync with side-by-side bbox overlay
├── raw_to_movie.py     ← Compare events.raw reconstruction vs Event/Frames/ video
├── verify_frames.py    ← Pixel-level alignment check (MAE score)
├── raw_label_check.py  ← Verify events.raw sync with Event_YOLO bounding boxes
├── environment.yml     ← conda environment (CPU PyTorch)
├── requirements.txt    ← pip requirements
└── CODE_GUIDE.md       ← full function-by-function documentation

data_from_fred/
├── splits.yaml         ← which sequence numbers go to train / val / test
├── catalog.yaml        ← auto-generated metadata for every zip (run make_catalog.py)
├── 7.zip               ← sequence data (or extracted as 7/)
└── 4.zip, 10.zip …     ← additional sequences (~100 total)
```

## Setup

```bash
conda env create -f environment.yml
conda activate drone_detect
```

Or with pip:
```bash
pip install -r requirements.txt
```

Data is read directly from `.zip` files — no extraction needed.
`config.py` auto-detects zip vs extracted folder via `zip_utils.init_sequence()`.

### Windows — fix OpenMP conflict before training
```powershell
# Per session:
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# Or permanently for the conda env:
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE -n drone_detect
```

## Here's the complete run order for all 3

There are 3 pipelines in this project. Run them from the `ai_drone/` root:

### Pipeline 1 — FRED event baseline (target: 87.68% mAP50)
Replicates the FRED paper result using pre-extracted 33ms event frames (3-channel PNG).
```powershell
cd ai_drone\Fred
python build_dataset.py
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python train.py
python evaluate.py
```

### Pipeline 2 — FRED RGB baseline (target: 76.23% mAP50)
Same YOLO model on the RGB camera stream for comparison.
```powershell
cd ai_drone\Fred
python build_dataset.py --mode rgb
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python train.py --mode rgb
python evaluate.py --mode rgb
```

### Data sync check — verify Event ↔ RGB alignment
```powershell
cd ai_drone\4channel_project
python sync_check.py                  # full sequence, 5 fps auto-play
python sync_check.py --start 9.8     # jump to drone segment (recommended)
python sync_check.py --save sync.mp4 # save side-by-side video
```
Shows Event/Frames/ (cyan bbox from Event_YOLO) beside PADDED_RGB/ (orange bbox
from RGB_YOLO). Console prints `Δcx`/`Δcy` per frame — near 0 means synced.
Controls: SPACE / D = next,  A = prev,  D = jump +10,  Q = quit.

### Pipeline 3 — 4-channel physics (target: > 87.68% mAP50)
Our approach: reads `events.raw` directly from zip, builds 4 physics-motivated channels.
```powershell
cd c:\ai_drone

# (First time or after adding zips) Generate sequence catalog:
python 4channel_project/make_catalog.py

# Edit data_from_fred/splits.yaml to assign sequences to train/val/test

cd 4channel_project
python dataset_builder.py       # reads splits.yaml, processes all assigned sequences
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python train_4ch_yolo.py        # auto-resumes from last.pt if interrupted
python evaluate.py              # compare vs 87.68% baseline
```

### Google Colab (Pipeline 3, T4 GPU — recommended)
1. Upload `ai_drone/` to Google Drive (keep folder structure intact)
2. Open `4channel_project/colab_run.ipynb` in Colab
3. Runtime → Change runtime type → T4 GPU
4. Run cells 1–10 in order (~20-30 min build + ~2-4 hrs training)

---

## Pipeline 3 — Detailed Run Steps

### Step 1 — Test the EVT3 reader
```bash
python evt3_reader.py
```
Expected output: header info, event counts for first window.

### Step 2 — Test channel generation
```bash
python channels.py
```
Expected output: 4 channel shapes + a preview PNG.

### Step 3 — Build training dataset
```powershell
# Delete old dataset first if rebuilding from scratch:
Remove-Item -Recurse -Force dataset

# (Optional) update catalog after adding new zips:
python make_catalog.py

# Build from all sequences in splits.yaml:
python dataset_builder.py
```
Reads `events.raw` from each zip in `splits.yaml`, generates 4 channels per 33ms window,
saves **4-channel RGBA PNG** files + YOLO labels into `dataset/images/` and `dataset/labels/`.
Split membership recorded in `dataset/train.txt`, `val.txt`, `test.txt`.
Writes `dataset.yaml` with `channels: 4` — Ultralytics reads this natively.

Frame names: `s{seq_num}_{t_us:012d}.png` — unique across all sequences.

Legacy single-sequence mode (seq 7 only): `python dataset_builder.py --single`

### Step 4 — Train
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python train_4ch_yolo.py
```
The script patches Ultralytics' imread to preserve the alpha channel (4-ch BGRA).
Training is safe to interrupt — auto-resumes from `last.pt`.
Best model saved to `./runs/fred_4channel/weights/best.pt`.

### Step 5 — Evaluate
```bash
python evaluate.py
```
Prints mAP50 vs paper's 87.68% baseline.

```bash
python evaluate.py --ablation
```
Tests each channel combination to show individual contribution.

## Key Design Decisions

**Why remove the standard accumulated frame (Channel 1 in FRED)?**
It equals positive + negative events combined — mathematically redundant
once you have Channels 1 and 2 separately.

**Why threshold=5 for rotor map?**
200Hz rotor × 0.033s ≈ 6.6 blade crossings per pixel per window.
Threshold of 5 captures this. Bird wing at 5Hz fires <1 time — below threshold.

**Why time surface instead of another temporal slice?**
Time surface carries recency information orthogonal to event count.
A pixel that fired once at t=32ms is more "alive" than one that fired
10 times at t=1ms. Event count channels cannot express this.

**Why only one architecture change (first layer)?**
Clean experimental design. Everything else is identical to FRED paper.
Any mAP50 improvement is directly attributable to the channel design.

## Expected Results

| Configuration | Expected mAP50 |
|---|---|
| FRED paper (1-channel) | 87.68% (baseline) |
| Ch1+Ch2 only (polarity) | ~88-89% |
| Ch1+Ch2+Ch3 (+ rotor) | ~89-91% |
| All 4 channels | ~90-92% |

These are estimates. Actual results depend on training time and GPU.

## Research Contribution

> "We replace the standard accumulated event frame with four physically
> motivated channels — positive polarity, negative polarity, rotor frequency
> map, and time surface — each carrying independent non-redundant information.
> This minimal architectural change (first conv layer only) achieves
> higher mAP50 than the FRED paper baseline while providing interpretable
> physical features that generalize to unseen drone models."
