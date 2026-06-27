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
ai_drone/                              ← git root (this folder)
├── CLAUDE.md                          ← this file
├── 2506.05163v1.pdf                   ← FRED paper (reference)
├── fred_step1_download.py             ← HuggingFace download (simplified pipeline)
├── fred_step2_convert.py              ← convert FRED annotations → YOLO format
├── fred_step3_train.py                ← train YOLO11n (simplified pipeline)
├── fred_step4_detect.py               ← run inference (simplified pipeline)
├── pid_annotation_fft.py              ← PID frequency analysis on annotation centroids
├── pid_annotation_fft.png             ← FFT output (9.14 Hz PID peak)
├── data_from_fred/                    ← FRED dataset sequences (zip or extracted folders)
│   ├── splits.yaml                    ← which sequence numbers go to train/val/test
│   ├── catalog.yaml                   ← auto-generated metadata for every zip sequence
│   ├── 7.zip                          ← sequence 7 (or extracted as 7/)
│   ├── 4.zip, 10.zip, 31.zip, 52.zip ← additional sequences (~100 total planned)
│   └── 7/                             ← example extracted layout (same as inside zip):
│       ├── Event/events.raw           ← 127MB Prophesee EVT3 raw stream
│       ├── Event/Frames/              ← pre-extracted 33ms PNGs (Pipeline 1)
│       ├── Event_YOLO/                ← YOLO labels for event frames
│       ├── PADDED_RGB/                ← padded RGB frames (Pipeline 2)
│       ├── RGB/                       ← raw RGB frames
│       ├── RGB_YOLO/                  ← YOLO labels for RGB frames
│       ├── Removed_frames/            ← frames excluded from dataset
│       ├── coordinates.txt            ← ground truth bbox annotations
│       ├── interpolated_coordinates.txt ← smoother float bboxes (preferred)
│       └── tracks.txt                 ← drone track metadata
├── Fred/                              ← Pipelines 1 & 2 (paper baseline)
│   ├── build_dataset.py               ← read frames+labels from zip → YOLO layout on disk
│   ├── train.py                       ← standard YOLO11n, no channel patch
│   └── evaluate.py                    ← compare vs paper mAP50
└── 4channel_project/                  ← Pipeline 3 (our approach)
    ├── config.py                      ← all settings; calls init_sequence() on import
    ├── zip_utils.py                   ← transparent zip/folder access (seq_glob, seq_imread…)
    ├── evt3_reader.py                 ← EVT3 binary parser (zip-aware via BytesIO)
    ├── filters.py                     ← refractory + BAF noise filters
    ├── channels.py                    ← 4-channel generator
    ├── dataset_builder.py             ← build YOLO dataset from events.raw (multi-seq)
    ├── make_catalog.py                ← scan all zips → write data_from_fred/catalog.yaml
    ├── gdrive.py                      ← Google Drive folder scan + lazy zip download
    ├── train_4ch_yolo.py              ← train with 4-channel input
    ├── evaluate.py                    ← compare vs paper baseline
    ├── runs/detect/                   ← inference output (bounding box overlays)
    ├── README.md                      ← project overview
    ├── requirements.txt               ← pip dependencies
    ├── environment.yml                ← conda environment
    ├── yolo11n.pt                     ← YOLO base weights
    ├── docs/                          ← documentation & research notes
    │   ├── CODE_GUIDE.md              ← developer guide for 4-channel pipeline
    │   ├── EVT3_READER_FIXES.md       ← EVT3 parser bug history
    │   ├── drone_detection_research_summary.md ← full research notes
    │   ├── FRED_EventCamera_Discussion.md ← research discussion notes
    │   ├── flow_chart.md              ← pipeline flow diagram
    │   ├── TODO.md                    ← team task list
    │   └── channel_preview.png        ← sample 4-channel output image
    ├── notebooks/                     ← Jupyter notebooks
    │   ├── colab_run.ipynb            ← Google Colab notebook (T4 GPU)
    │   └── drone_detection_notebook.ipynb ← exploration notebook
    └── tools/                         ← diagnostic & visualization utilities
        ├── sync_check.py              ← verify event↔RGB sync with bbox overlay
        ├── raw_label_check.py         ← verify events.raw sync with Event_YOLO labels
        ├── raw_to_movie.py            ← compare events.raw vs Event/Frames/ video
        ├── verify_frames.py           ← pixel-level alignment check (MAE)
        ├── debug_filter_preview.py    ← visualize noise filter effects
        ├── find_offset.py             ← find time offset between event/RGB streams
        ├── inspect_raw.py             ← inspect EVT3 raw file contents
        ├── make_filter_movie.py       ← render filter comparison video
        └── view_raw_events.py         ← live viewer for raw event stream
```

## Zip file access — no extraction needed

All scripts read FRED data directly from `.zip` files (`7.zip`, `4.zip`, etc.) via
`zip_utils.py`. If a folder (e.g. `data_from_fred/7/`) exists on disk it is used instead,
otherwise the matching `.zip` is opened transparently.

`zip_utils.py` provides drop-in replacements:
- `seq_glob(dir, pattern)` — like `glob.glob`
- `seq_imread(path, flags)` — like `cv2.imread`
- `seq_open_lines(path)` — like `open(path).readlines()`
- `seq_exists(path)` — like `os.path.exists` (handles virtual directories in zips)
- `init_sequence(seq_dir)` — call once; auto-detects zip vs folder

`config.py` calls `init_sequence(SEQUENCE_DIR)` on import, so all tools that import
`config` automatically get zip access for sequence 7.

## Multi-sequence dataset (Pipeline 3)

Sequences are assigned to splits via `data_from_fred/splits.yaml`:
```yaml
train: [4, 7, 10, 31]
val:   [52]
test:  []
```
Each zip is used as a whole unit for one split — no per-frame random splitting.

**Auto-assign splits by percentage** (writes splits.yaml automatically):
```powershell
python 4channel_project/make_catalog.py --auto-split --train 70 --val 20 --test 10
```
Sequences are sorted numerically and distributed by the given percentages (reproducible, no randomness).

Generated 4-channel PNGs go into `4channel_project/dataset/images/` (flat folder).
Split membership is recorded in `dataset/train.txt`, `val.txt`, `test.txt`.
`dataset/dataset.yaml` references these txt files (Ultralytics txt-path format).

Frame naming: `s{seq_num}_{t_start_us:012d}.png` — globally unique across sequences.

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
- Dataset: multi-sequence from splits.yaml (train: seqs 4,7,10,31 / val: seq 52)
- Checkpoint system: auto-resumes from `runs/fred_4channel/weights/last.pt`
- imread fix: patches `ultralytics.utils.patches.imread` + `ultralytics.data.base.imread`

## Here's the complete run order for all 3

### Pipeline 0 — HuggingFace simplified (download → train → detect)
```powershell
# Run from c:\ai_drone
python fred_step1_download.py         # download FRED from HuggingFace
python fred_step2_convert.py          # convert annotations to YOLO format
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python fred_step3_train.py            # train YOLO11n
python fred_step4_detect.py           # run inference
```

### Pipeline 1 — FRED event baseline (target: 87.68% mAP50)
```powershell
cd Fred
python build_dataset.py               # reads directly from zip, copies frames to disk
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python train.py
python evaluate.py
```

### Pipeline 2 — FRED RGB baseline (target: 76.23% mAP50)
```powershell
cd Fred
python build_dataset.py --mode rgb
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python train.py --mode rgb
python evaluate.py --mode rgb
```

### Pipeline 3 — 4-channel physics (target: > 87.68% mAP50)
```powershell
cd c:\ai_drone

# (First time or when adding new sequences) Update catalog:
python 4channel_project/make_catalog.py   # writes data_from_fred/catalog.yaml

# Option A — assign splits manually: edit data_from_fred/splits.yaml
# Option B — assign splits automatically by percentage:
python 4channel_project/make_catalog.py --auto-split --train 70 --val 20 --test 10

cd 4channel_project
python dataset_builder.py             # processes all sequences from splits.yaml
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python train_4ch_yolo.py              # auto-resumes from last.pt if interrupted
python evaluate.py
```

Verify dataset split counts without regenerating:
```powershell
python 4channel_project/dataset_builder.py --check
```

Single-sequence legacy mode (seq 7 only, random 80/20 split):
```powershell
python 4channel_project/dataset_builder.py --single
```

### Sequence catalog
```powershell
# Run from c:\ai_drone
python 4channel_project/make_catalog.py
```
Scans all `data_from_fred/*.zip`, reads metadata from inside each zip (ts_shift_us,
frame counts, sizes), and writes `data_from_fred/catalog.yaml`.
Re-running merges new data but preserves manually written `description` and
`drive_file_id` fields.
Edit `catalog.yaml` to add descriptions like: `description: "indoor flight, low light"`.

### Google Drive download (optional)

Dataset source: `https://drive.google.com/drive/folders/1pISIErXOx76xmCqkwhS3-azWOMlTKZMp`

If zips are already in `data_from_fred/` (manually downloaded), skip this entirely.
Three download modes — all require `pip install gdown`:

```powershell
# Option A — download ALL zips at once (simplest, no API key needed):
python 4channel_project/make_catalog.py --download-all

# Option B — selective: scan first, then download only what splits.yaml needs:
python 4channel_project/make_catalog.py --scan-drive      # saves file IDs → catalog.yaml
python 4channel_project/dataset_builder.py --download     # downloads only missing zips

# Option C — scan with a free Google API key (most reliable listing):
python 4channel_project/make_catalog.py --scan-drive --api-key AIza...
python 4channel_project/dataset_builder.py --download
```
`--scan-drive` uses gdown's built-in folder parser (no API key needed by default).
`--api-key`: free key from console.cloud.google.com → Enable Drive API → Credentials.
Zips already present locally are never re-downloaded.

### Data sync check (Event ↔ RGB bounding boxes)
```powershell
cd 4channel_project
python tools/sync_check.py                  # full sequence, auto-play at 5 fps
python tools/sync_check.py --start 9.8     # jump to drone segment
python tools/sync_check.py --save sync.mp4 # save side-by-side video
```
LEFT panel (cyan box) = Event/Frames/ + Event_YOLO labels.
RIGHT panel (orange box) = PADDED_RGB/ + RGB_YOLO labels.
Console prints `ev_cx/cy`, `rgb_cx/cy`, `Δcx/Δcy` — near 0 = synced.
Controls: SPACE=next  A=prev  D=+10  Q=quit

### Raw event data vs Event_YOLO label sync check
```powershell
cd 4channel_project
python tools/raw_label_check.py                  # full sequence 7
python tools/raw_label_check.py --start 9.8     # jump to drone segment
python tools/raw_label_check.py --seq 4         # sequence 4 (reads from 4.zip)
python tools/raw_label_check.py --save out.mp4  # also save video
```
Renders frames live from `events.raw` (no pre-extracted PNGs) and overlays the
matching `Event_YOLO/` bounding box by timestamp.
- **CYAN box** = Event_YOLO label drawn on the raw event frame
- **GREEN** banner = label timestamp matched (Δ shown in µs)
- **ORANGE** banner = label file exists but empty (drone out of frame)
- **RED** banner = no Event_YOLO file within ±16ms of this window
Applies both EVT3 fixes automatically: 24-bit rollover (via EVT3Reader) and
ts_shift clock offset (`raw_t - ts_shift_us` → Event_YOLO time).
Console prints `cx cy w h` and timestamp delta for every frame.
Controls: SPACE=pause  A/←=prev  D/→=+10  Q=quit

### Event reconstruction vs Frames/ comparison
```powershell
cd 4channel_project
python tools/raw_to_movie.py                        # side-by-side: Frames/ PNG vs raw reconstruction
python tools/raw_to_movie.py --start 9.8 --end 20  # drone segment only
python tools/raw_to_movie.py --seq 4               # sequence 4
```
LEFT = Event/Frames/ PNG, RIGHT = reconstructed from events.raw. Also saves `raw_events.mp4`.
Controls: SPACE=pause  Q/ESC=quit

### Pixel-level alignment check
```powershell
cd 4channel_project
python tools/verify_frames.py             # 20 frames from drone segment; prints MAE per frame
python tools/verify_frames.py --n 50      # check 50 frames
python tools/verify_frames.py --start 0   # include countdown region
```
MAE < 5 = aligned, ~20–40 = off by 1 frame, ~60–80 = badly misaligned.

### Filter before/after preview
```powershell
cd 4channel_project
python tools/debug_filter_preview.py      # saves side-by-side PNG to current dir
```
Reads one 33ms window, generates 4-channel images before/after refractory filter.

### Find event/RGB time offset
```powershell
cd 4channel_project
python tools/find_offset.py              # prints first EVT3 timestamp vs first Frames/ filename
```
Difference between printed values is the ts_shift_us offset.

### Inspect EVT3 data quality
```powershell
cd 4channel_project
python tools/inspect_raw.py                                              # uses config.py RAW_FILE
python tools/inspect_raw.py --raw ../data_from_fred/7/Event/events.raw  # explicit path
python tools/inspect_raw.py --raw ../data_from_fred/7/Event/events.raw --ann ../data_from_fred/7/coordinates.txt
```
Auto-detects hot pixels, dead zones, rate spikes, polarity bias. Prints "clean data starts at t=X.Xs".

### Filter comparison movie (live viewer)
```powershell
cd 4channel_project
python tools/make_filter_movie.py                       # drone segment 9.87s–35s
python tools/make_filter_movie.py --start 0 --end 10   # first 10 seconds
python tools/make_filter_movie.py --delay 50            # ms per frame (default 33)
```
4-channel 2×2 grid: left = raw events, right = after refractory filter.
Controls: SPACE=pause/resume  →=step  Q/ESC=quit

### Live raw event viewer
```powershell
cd 4channel_project
python tools/view_raw_events.py                       # sequence 7 (default)
python tools/view_raw_events.py --seq 4               # sequence 4
python tools/view_raw_events.py --seq 4 --delay 100   # slow down
```
GREEN highlight = annotated drone frames, RED = removed/gap, YELLOW = pre/post flight.
Controls: SPACE=pause  →/D=step forward  ←/A=step back  Q/ESC=quit

### Adding new tools to tools/

All scripts in `tools/` must add this at the top (after the docstring, before local imports) so they can import core modules from the project root:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```
Run them from `4channel_project/` as `python tools/<script>.py`.

### PID oscillation analysis
```powershell
# Run from c:\ai_drone
python pid_annotation_fft.py          # FFT on annotation centroids → pid_annotation_fft.png
```
Detects PID wobble frequency in ground-truth bboxes. Confirmed result: 9.14 Hz peak, SNR=2.77× at t=9.9–35.2s in sequence 7.

### Google Colab (Pipeline 3 only, T4 GPU)
1. Upload `ai_drone/` to Google Drive (keep folder structure)
2. Open `4channel_project/notebooks/colab_run.ipynb` in Colab
3. Runtime → Change runtime type → T4 GPU
4. Run cells 1–10 in order

## Full research notes

See: `4channel_project/docs/drone_detection_research_summary.md`
