# Drone Detection Project — Context for Claude

## What this project is

Physics-driven multi-channel event camera input for improved drone detection,
extending the FRED dataset paper (87.68 mAP50) with 4 physically motivated channels.

## The core idea

The FRED paper feeds YOLO one accumulated 33ms event frame — discarding polarity,
motion direction, and motor frequency. We replace it with 4 channels:

| Channel | Content | Physical meaning |
|---|---|---|
| 1 | Positive polarity events | Leading edge — where drone is going |
| 2 | Negative polarity events | Trailing edge — where drone came from |
| 3 | Rotor frequency map | Spinning motor signature (unique to drones) |
| 4 | Time surface | Most recent activity per pixel |

Only change to YOLO: first Conv2d layer from in_channels=3 to in_channels=4.
Ultralytics reads `channels: 4` from dataset.yaml and applies this automatically.

## Key confirmed result

PID oscillation detected in FRED sequence 7:
- Segment t=9.9–35.2s: peak at 9.14 Hz, SNR=2.77× ✓

## The 3 pipelines

| # | Folder | What | Target mAP50 |
|---|---|---|---|
| 1 | `Fred/` | FRED paper baseline — event frames (33ms PNGs, 3-ch) | 87.68% |
| 2 | `Fred/` | FRED paper baseline — RGB camera (30fps JPGs, 3-ch) | 76.23% |
| 3 | `4channel_project/` | Our 4-channel physics pipeline | > 87.68% |

## Project files

```
ai_drone/
├── CLAUDE.md                          ← this file
├── data_from_fred/                    ← FRED dataset sequences
│   ├── 7/                             ← sequence 7 data
│   │   ├── Event/events.raw           ← 127MB Prophesee EVT3 raw stream
│   │   ├── coordinates.txt            ← 3007 ground truth bbox annotations
│   │   ├── interpolated_coordinates.txt ← smoother float bboxes (preferred)
│   │   ├── Event/Frames/              ← pre-extracted 33ms PNGs (Pipeline 1)
│   │   ├── Event_YOLO/                ← YOLO labels for event frames
│   │   ├── PADDED_RGB/                ← padded RGB frames (Pipeline 2)
│   │   └── RGB_YOLO/                  ← YOLO labels for RGB frames
│   └── 4/                             ← sequence 4 data (same structure)
├── Fred/                              ← Pipelines 1 & 2 (paper baseline)
│   ├── build_dataset.py               ← copy frames + labels into YOLO layout
│   ├── train.py                       ← standard YOLO11n, no channel patch
│   └── evaluate.py                    ← compare vs paper mAP50
└── 4channel_project/                  ← Pipeline 3 (our approach)
    ├── config.py                      ← all settings (auto-detects Colab/local)
    ├── evt3_reader.py                 ← EVT3 binary parser
    ├── filters.py                     ← refractory + BAF noise filters
    ├── channels.py                    ← 4-channel generator
    ├── dataset_builder.py             ← build YOLO training data from events.raw
    ├── train_4ch_yolo.py              ← train with 4-channel input
    ├── evaluate.py                    ← compare vs paper baseline
    ├── sync_check.py                  ← verify event↔RGB sync with bbox overlay
    ├── raw_to_movie.py                ← compare events.raw vs Event/Frames/ video
    ├── verify_frames.py               ← pixel-level alignment check (MAE)
    └── colab_run.ipynb                ← Google Colab notebook (T4 GPU)
```

## Dataset format

- Pipeline 1 & 2: standard 3-channel PNG/JPG, `channels: 3` in dataset.yaml
- Pipeline 3: **4-channel RGBA PNG**, `channels: 4` in dataset.yaml
  - Ultralytics reads `channels: 4` and adjusts first Conv2d automatically
  - imread patch in train_4ch_yolo.py forces `cv2.IMREAD_UNCHANGED` to preserve alpha

## Known Windows issue — OpenMP conflict

Before training on Windows/conda, set:
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```
Or permanently: `conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE -n drone_detect`

## Training status

- Architecture confirmed: layer 0 is `[4, 16, 3, 2]` — 4 input channels ✓
- Dataset: 2523 train images, 647 val images (from sequence 7)
- Checkpoint system: auto-resumes from `runs/fred_4channel/weights/last.pt`
- imread fix: patches `ultralytics.utils.patches.imread` + `ultralytics.data.base.imread`

## Here's the complete run order for all 3

### Pipeline 1 — FRED event baseline (target: 87.68% mAP50)
```powershell
cd ai_drone\Fred
python build_dataset.py
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python train.py
python evaluate.py
```

### Pipeline 2 — FRED RGB baseline (target: 76.23% mAP50)
```powershell
cd ai_drone\Fred
python build_dataset.py --mode rgb
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python train.py --mode rgb
python evaluate.py --mode rgb
```

### Pipeline 3 — 4-channel physics (target: > 87.68% mAP50)
```powershell
cd ai_drone\4channel_project
python dataset_builder.py
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python train_4ch_yolo.py    # auto-resumes from last.pt if interrupted
python evaluate.py
```

### Data sync check (Event ↔ RGB bounding boxes)
```powershell
cd ai_drone\4channel_project
python sync_check.py                  # full sequence, auto-play at 5 fps
python sync_check.py --start 9.8     # jump to drone segment
python sync_check.py --save sync.mp4 # save side-by-side video
```
LEFT panel (cyan box) = Event/Frames/ + Event_YOLO labels.
RIGHT panel (orange box) = PADDED_RGB/ + RGB_YOLO labels.
Console prints `ev_cx/cy`, `rgb_cx/cy`, `Δcx/Δcy` — near 0 = synced.
Controls: SPACE=next  A=prev  D=+10  Q=quit

### Google Colab (Pipeline 3 only, T4 GPU)
1. Upload `ai_drone/` to Google Drive (keep folder structure)
2. Open `4channel_project/colab_run.ipynb` in Colab
3. Runtime → Change runtime type → T4 GPU
4. Run cells 1–10 in order

## Full research notes

See: `4channel_project/drone_detection_research_summary.md`
