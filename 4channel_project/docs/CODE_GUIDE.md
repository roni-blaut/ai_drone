# 4-Channel Project — Code Guide

Complete explanation of every file, every function, and how data flows through the pipeline.

---
data from :
https://drive.google.com/drive/folders/1pISIErXOx76xmCqkwhS3-azWOMlTKZMp?usp=share_link

https://github.com/miccunifi/FRED/tree/main


## Workflow

**Scenario A — zips already in data_from_fred/**
```
make_catalog.py --auto-split → splits.yaml + catalog.yaml  (one command does both)
dataset_builder.py           → dataset/                    (4-channel PNGs + YOLO labels)
train_4ch_yolo.py            → best.pt                     (trained model)
evaluate.py                  → mAP50
```
`--auto-split` writes `splits.yaml` first, then runs the full catalog scan so
`catalog.yaml` already reflects the new split assignments. One command, both files.

To refresh the catalog without changing splits (e.g. after adding more zips):
```
make_catalog.py              → catalog.yaml only  (splits.yaml unchanged)
```

**Scenario B — zips on Google Drive**
```
make_catalog.py --download-all               → downloads all zips, then Scenario A
  OR
make_catalog.py --scan-drive                 → catalog.yaml with drive_file_id per sequence
  edit splits.yaml to pick which sequences
dataset_builder.py --download                → downloads missing zips, then builds dataset/
train_4ch_yolo.py + evaluate.py
```

**Key file roles:**
- `catalog.yaml` — metadata index (frame counts, Drive IDs). Read by `--download`, not by training.
- `splits.yaml` — which sequence numbers → train / val / test. Read by `dataset_builder.py`.
- `dataset/` — the actual training data (generated PNGs). Read by YOLO.

### What dataset_builder.py does per sequence

Given `splits.yaml` with `train: [0, 1, 4, 7, 10, 31]`, it loops in that order:

```
For each seq in [0, 1, 4, 7, 10, 31]:
  open data_from_fred/{seq}.zip   ← read-only, never extracted
  read Event/events.raw (in-memory via BytesIO)
  slice into 33ms windows starting at t=9.8s (drone appears)

  for each window:
    skip if:  zero events / in Removed_frames/ / all events filtered out
    otherwise:
      fast_filter()          → remove noise
      generate_channels()    → 4 arrays (pos / neg / rotor / time surface)
      save → dataset/images/s{seq}_{t_us:012d}.png   (4-ch RGBA, 720×1280)
      save → dataset/labels/s{seq}_{t_us:012d}.txt   (YOLO bbox or empty)
      append path → train.txt

  zip file untouched — all output goes to dataset/
```

Result in `dataset/`:
```
images/s0_000009800000.png          labels/s0_000009800000.txt
images/s0_000009833000.png          labels/s0_000009833000.txt
...                                 ...
images/s31_000025400000.png         labels/s31_000025400000.txt

train.txt  ← absolute paths to all images from seqs 0,1,4,7,10,31
val.txt    ← absolute paths to images from val sequences
test.txt   ← absolute paths to images from test sequences
dataset.yaml  ← channels:4, nc:1, names:[drone]
```

YOLO reads `train.txt` / `val.txt` / `test.txt` directly — it never sees `splits.yaml` or `catalog.yaml`.
Filename prefix `s{seq}` guarantees no collisions across sequences.

---

## Pipeline Overview

```
data_from_fred/splits.yaml    data_from_fred/N.zip (or folder N/)
        │                              │
        │                      zip_utils.py  ← transparent zip/folder access
        │                              │
        ▼                              ▼
dataset_builder.py            evt3_reader.py  ← parse EVT3 binary (zip-aware)
(multi-sequence loop)                 │
        │                      filters.py     ← noise removal
        │                             │
        │                      channels.py    ← 4-channel generator
        │                             │
        ▼                             ▼
dataset/images/*.png   ← 4-ch RGBA PNG (frame named s{seq}_{t_us:012d})
dataset/labels/*.txt   ← YOLO bbox label
dataset/train.txt      ← paths of train images (whole sequences)
dataset/val.txt        ← paths of val images
        │
        ▼
train_4ch_yolo.py      ← patch YOLO first layer → train → best.pt
        │
        ▼
evaluate.py            ← mAP50 vs paper baseline + ablation study
```

All paths live in `config.py`. `zip_utils.init_sequence()` is called there on
import — all downstream scripts get transparent zip access automatically.

---

## config.py

**Purpose:** Single place for all settings. Auto-detects which environment you are running on.

### Environment detection

```python
ENV = _detect_env()
```

Checks in this order:
1. `DRONE_ENV` environment variable (manual override)
2. `google.colab` import succeeds → Colab
3. Linux + GPU + no display → NVIDIA server
4. Everything else → Local PC

Sets paths and training parameters accordingly:

| ENV | SEQUENCE_DIR | BATCH | DEVICE |
|---|---|---|---|
| local | ../data_from_fred/7 | 8 | GPU or CPU |
| colab | /content/drive/MyDrive/ai_drone/data_from_fred/7 | 16 | 0 (T4) |
| nvidia | /data/fred/7 | 32 | 0 |
| cpu | ../data_from_fred/7 | 4 | cpu |

### Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `WINDOW_US` | 33333 | 33ms window = 30fps (matches FRED paper) |
| `ROTOR_THRESHOLD` | 5 | Min events/pixel to count as rotor |
| `REFRACTORY_US` | 1000 | 1ms refractory period per pixel |
| `BAF_RADIUS_PX` | 3 | BAF neighbourhood radius in pixels |
| `BAF_DELTA_US` | 10000 | 10ms BAF time window |
| `N_CHANNELS` | 4 | Number of input channels to YOLO |
| `BATCH` | 8 | Training batch size |
| `EPOCHS` | 100 | Training epochs |

---

## zip_utils.py

**Purpose:** Transparent access to FRED sequence data from `.zip` files or extracted folders.
All scripts use the `seq_*` helpers instead of raw `glob`/`open`/`cv2.imread` calls.

### Module-level state

`_ACTIVE_SEQ` — module-level `ZipSequence` instance (or `None` for real filesystem).
Set by `init_sequence()`. All `seq_*` helpers check this automatically.

### Function: `init_sequence(seq_dir)`

Call once with the sequence directory (e.g. `../data_from_fred/7`).
- If `seq_dir` exists as a folder → uses real filesystem (`_ACTIVE_SEQ = None`)
- If `seq_dir` doesn't exist but `seq_dir + '.zip'` does → opens `ZipSequence`
- Calling again with the same zip is a no-op (zip stays open)

Called automatically by `config.py` on import for the default sequence.
Scripts with `--seq` flag call it again after `args = parser.parse_args()`.

### Helper functions

| Function | Replaces | Notes |
|---|---|---|
| `seq_glob(dir, pattern)` | `glob.glob(os.path.join(dir, pattern))` | returns fake filesystem paths |
| `seq_imread(path, flags)` | `cv2.imread(path, flags)` | decodes image from zip on demand |
| `seq_open_lines(path)` | `open(path).readlines()` | returns list of lines from zip member |
| `seq_open_binary(path)` | `open(path, 'rb')` | returns `io.BytesIO` (seekable) |
| `seq_exists(path)` | `os.path.exists(path)` | also checks virtual directories in zip |

All helpers fall back to real filesystem if the file exists on disk.

### Class: `ZipSequence`

Wraps `zipfile.ZipFile`. Key method: `_to_member(path)` converts an absolute
filesystem path back to a zip member name using `os.path.relpath`.

`ts_shift_us` is read from `Event/events.raw.tmp_index` inside the zip during `__init__`.

---

## evt3_reader.py

**Purpose:** Parse the Prophesee EVT3 binary format into a structured numpy array of events.
Supports both real filesystem and zip mode (via `zip_utils._ACTIVE_SEQ`).

### Zip-aware initialization

In `__init__`, if `zip_utils._ACTIVE_SEQ` is set and the file doesn't exist on disk:
- Loads entire `events.raw` into `io.BytesIO` (seekable, ~127 MB RAM)
- Gets `ts_shift_us` from `_ACTIVE_SEQ.ts_shift_us` (already parsed from zip index)
- All subsequent reads use `self._bio` instead of opening the file

### EVT3 format recap

The file is a stream of 2-byte (16-bit) words. Each word has a type in bits 15–12:

```
0x0  ADDR_Y   → sets current Y coordinate
0x2  ADDR_X   → fires event (x, current_y, current_t, polarity)
0x6  TIME_LOW → low 12 bits of timestamp
0x8  TIME_HIGH→ high bits of timestamp
```

Each event is reconstructed by combining the most recent ADDR_Y, ADDR_X, and timestamp words.

### Class: `EVT3Reader`

**`__init__(filepath, chunk_size=2_000_000)`**
- Opens the file, finds where the ASCII header ends
- `chunk_size` = how many 16-bit words to process per chunk (default 4MB)

**`read_header()`**
- Returns dict of metadata from the ASCII header
- Example keys: sensor type, resolution, recording date

**`read_all(max_events=None)`**
- Reads entire file into one numpy array
- Warning: loads ~10M+ events into RAM for 127MB file
- Use `read_window()` instead for targeted extraction

**`read_window(t_start_us, t_end_us)`**
- Reads only events in a specific time window
- Streams the file and stops early once past `t_end_us`
- Much faster than `read_all()` for short windows
- Returns structured numpy array with fields: `x, y, t, p`

**`iter_windows(window_us, t_start, t_end)`**
- Generator that yields `(t_start, events)` for each consecutive time window
- Used by `dataset_builder.py` to process the whole file efficiently
- Maintains a buffer across chunks so no events are missed at boundaries

**`_get_file()`**
- Returns `self._bio` (zip mode) or opens `self.filepath` (disk mode)
- Used by `_find_header_end()` and `read_header()` for transparent access

**`_find_header_end()`**
- Scans file for end of ASCII header (lines starting with `%`)
- Returns byte offset where binary data begins
- Uses `_get_file()` — works in both zip and disk mode

**`_iter_chunks()`**
- Dispatches to `_stream_chunks(self._bio)` (zip) or `_stream_chunks(f)` (disk)

**`_stream_chunks(f)`**
- Inner EVT3 decode loop — reads 16-bit words from file-like object `f`
- Maintains decoder state (current_y, time_high, rollover_offset) across chunks
- Yields structured event arrays one chunk at a time

### Output format

```python
EVENT_DTYPE = np.dtype([
    ('x', np.uint16),   # pixel column  0–1279
    ('y', np.uint16),   # pixel row     0–719
    ('t', np.int64),    # timestamp     microseconds
    ('p', np.uint8),    # polarity      0=negative, 1=positive
])
```

---

## filters.py

**Purpose:** Remove noise from raw event stream. Two filters applied in order.

### Why filters are needed

Raw event stream contains:
- **Hot pixels**: defective pixels firing thousands of times per second
- **Thermal noise**: random isolated events with no neighbours
- **Real events**: drone edges, moving objects

After both filters ~98% of raw events are removed. What remains is almost purely the drone.

### Function: `refractory_filter(events, tau_us=1000)`

Kills hot pixels. Each pixel has a memory of when it last fired.

```
For each event at (x, y, t):
  if (t - last_fire[y, x]) < tau_us:
      DISCARD  ← fired too recently, must be noise
  else:
      KEEP
      last_fire[y, x] = t
```

- `tau_us`: refractory period in microseconds (default 1ms)
- Cost: O(1) per event — just one array lookup
- Run this FIRST because it is the cheapest filter

### Function: `baf_filter(events, radius=3, delta_t_us=10000)`

Background Activity Filter. Keeps events that have neighbours.

```
For each event at (x, y, t):
  Check all pixels in (2*radius+1) × (2*radius+1) neighbourhood
  If ANY neighbour fired within last delta_t_us:
      KEEP  ← part of a real moving object
  else:
      DISCARD  ← isolated noise
```

- `radius`: neighbourhood radius in pixels (default 3px = 7×7 box)
- `delta_t_us`: time window to check (default 10ms)
- Cost: O(radius²) per event — more expensive than refractory
- Run this SECOND

### Function: `apply_filters(events, tau_us, radius, delta_t_us)`

Convenience wrapper — applies refractory then BAF in correct order.

### Function: `fast_filter(events, tau_us)`

Refractory only — no BAF. Faster, good enough for training data generation.
Use this in `dataset_builder.py` where speed matters more than filter quality.

---

## channels.py

**Purpose:** Convert a filtered event array into 4 physics-based input channels.

### Function: `generate_channels(events, t_start_us, t_end_us)`

Main entry point. Calls all 4 channel generators and stacks the result.

```python
channels = generate_channels(events, t_start, t_end)
# channels.shape == (4, 720, 1280)  dtype float32
# Each channel normalized to [0, 1]
```

Internally calls:
1. `_channel_positive_polarity(events)`
2. `_channel_negative_polarity(events)`
3. `_channel_rotor_map(events)`
4. `_channel_time_surface(events, t_start_us, t_end_us)`

Then stacks with `np.stack([ch1, ch2, ch3, ch4], axis=0)`.

### Function: `_channel_positive_polarity(events)`

**Channel 1 — Leading edge**

Counts events where `polarity == 1` per pixel.
A positive event means pixel brightness INCREASED — the drone is moving INTO this pixel.
Shows the front face of the drone — where it is going.

```python
pos = events[events['p'] == 1]
np.add.at(frame, (pos['y'], pos['x']), 1.0)
```

### Function: `_channel_negative_polarity(events)`

**Channel 2 — Trailing edge**

Counts events where `polarity == 0` per pixel.
A negative event means pixel brightness DECREASED — the drone just LEFT this pixel.
Shows the back face of the drone — where it came from.

Together channels 1 and 2 give YOLO direction of motion without any optical flow calculation.

### Function: `_channel_rotor_map(events, threshold=5)`

**Channel 3 — Spinning motor signature**

Counts ALL events per pixel (both polarities), then zeros out any pixel below `threshold`.

```python
count = np.zeros((720, 1280))
np.add.at(count, (events['y'], events['x']), 1.0)
rotor_map = count.copy()
rotor_map[rotor_map < threshold] = 0
```

Why threshold=5:
```
Drone rotor at 200Hz × 0.033s = 6.6 crossings per pixel → ABOVE threshold
Bird wing at 5Hz × 0.033s     = 0.16 crossings per pixel → BELOW threshold
Tree branch at 0.5Hz           = 0.016 crossings per pixel → BELOW threshold
```

This channel is zero for birds and trees. Only spinning machinery produces it.

### Function: `_channel_time_surface(events, t_start_us, t_end_us)`

**Channel 4 — Recency map**

Stores the TIMESTAMP of the most recent event at each pixel, normalized to [0, 1].

```
Fired at t=33ms (end of window):   pixel = 1.0  (just happened)
Fired at t=16ms (middle):          pixel = 0.5
Fired at t=1ms  (start):           pixel = 0.03
Never fired:                       pixel = 0.0
```

The drone glows bright because it JUST fired those pixels.
Background noise that fired early in the window fades out automatically.

### Function: `_channel_time_surface_fast(events, t_start_us, t_end_us)`

Vectorized version — faster for large event counts.
Uses numpy assignment (later events overwrite earlier ones for same pixel).

### Function: `_normalize(frame)`

Divides frame by its maximum value to get [0, 1] range.
Returns zeros if frame is empty (no events).

### Function: `channels_to_rgb_preview(channels)`

Creates a 2×2 grid preview image for debugging:
```
[Ch1 positive | Ch2 negative]
[Ch3 rotor    | Ch4 surface ]
```
Returns BGR image for OpenCV display/save.

---

## dataset_builder.py

**Purpose:** Read `events.raw` from one or more sequences, generate 4 channels per 33ms window, match annotations, save as YOLO training data.

### Output structure (multi-sequence mode, default)

```
dataset/
├── images/       ← all 4-channel RGBA PNGs, flat (no train/val subdirs)
│   ├── s4_000009800000.png
│   ├── s7_000009800000.png   (same timestamp, different seq → no collision)
│   └── …
├── labels/       ← YOLO .txt labels, flat
│   └── …
├── train.txt     ← absolute paths of train images (whole sequences per splits.yaml)
├── val.txt       ← absolute paths of val images
├── test.txt      ← absolute paths of test images (may be empty)
└── dataset.yaml  ← channels: 4, references train.txt / val.txt / test.txt
```

Frame naming: `s{seq_num}_{t_us:012d}.png` — globally unique across all sequences.

### Function: `check_dataset(output_dir)`

Prints a summary of a built dataset without regenerating any images:
- Total `.png` / `.txt` file counts in `images/` and `labels/`
- Line counts for `train.txt`, `val.txt`, `test.txt` + which sequences each contains
- Filename collision check (all names must be unique)

Run with `python dataset_builder.py --check`.

### Function: `build_multi_sequence(splits_yaml, output_dir, window_us, download=False)`

**Default entry point.** Reads `data_from_fred/splits.yaml`, calls `_process_sequence()`
for each sequence in each split, writes `train.txt` / `val.txt` / `test.txt` index files,
then writes `dataset.yaml` via `_write_yaml_multi()`.

`download=True` loads `catalog.yaml` and calls `_ensure_zip()` before each sequence,
fetching missing zips from Drive on demand. Pass `--download` on the CLI to enable.

### Function: `_process_sequence(seq_num, raw_file, coords_file, output_dir, window_us)`

Extracted inner loop — processes one sequence:
1. Load annotations via `seq_open_lines()` (works from zip)
2. Derive removed windows from annotation gaps
3. Stream events via `EVT3Reader.iter_windows()`
4. Per window: filter → generate channels → save PNG + label
5. Returns list of absolute image paths for this sequence

### Function: `_find_coords(seq_dir)`

Returns `interpolated_coordinates.txt` if it exists in the zip/folder,
otherwise `coordinates.txt`. Uses `seq_exists()`.

### Function: `build_dataset(...)` *(legacy, single-sequence)*

Random 80/20 per-frame split within sequence 7. Saves to `images/train/`, `images/val/`.
Run with `python dataset_builder.py --single` to use this mode.

### Function: `load_annotations(coords_file)`

Parses `coordinates.txt` (or `interpolated_coordinates.txt`) into a sorted list of
`(time_us, x1, y1, x2, y2)`. Uses `seq_open_lines()` — works from zip.

### Function: `find_annotation(annotations, t_start_us, t_end_us, max_gap_us=100000)`

Binary search (`np.searchsorted`) for nearest annotation to a window.
Returns `(x1, y1, x2, y2)` or `None` if no annotation within `max_gap_us`.

### Function: `bbox_to_yolo(x1, y1, x2, y2)`

Absolute pixel coords → YOLO normalized `cx, cy, w, h` (clamped to [0,1]).

### Function: `_write_yaml_multi(output_dir)`

Writes `dataset.yaml` using Ultralytics txt-path format:
```yaml
path:  /absolute/path/to/dataset
train: train.txt
val:   val.txt
test:  test.txt
channels: 4
nc: 1
names: ['drone']
```

### Function: `_load_catalog()`

Loads `data_from_fred/catalog.yaml` and returns the `sequences` dict (or `{}` if missing).
Used by `build_multi_sequence(download=True)` to look up `drive_file_id` values.

### Function: `_ensure_zip(seq_num, catalog, data_dir)`

Called before `init_sequence()` when `--download` is active. Does nothing if the zip
or extracted folder already exists locally. If the zip is missing, looks up
`drive_file_id` in `catalog` and calls `gdrive.download_zip()`. Raises
`FileNotFoundError` with instructions if no ID is available.

### Function: `print_dataset_stats(output_dir)`

Auto-detects flat vs split-subdir layout. For flat (multi-sequence): counts
lines in each `*.txt` index file. For legacy: counts files in `images/train/` etc.

### CLI flags

| Flag | Effect |
|---|---|
| *(none)* | Multi-sequence build from splits.yaml |
| `--download` | Auto-download missing zips from Drive before building |
| `--check` | Print dataset stats, no image generation |
| `--single` | Legacy single-sequence mode (seq 7, random 80/20) |

---

## make_catalog.py

**Purpose:** Scan `data_from_fred/*.zip`, read metadata from inside each zip, write
`data_from_fred/catalog.yaml`. Run manually after adding new zips. Does **not** generate
images and does **not** download anything — metadata only.

### Default run (no flags) — `main()`

Opens every `.zip` in `data_from_fred/`, reads without extracting:
- `ts_shift_us` from the raw index header
- `n_event_frames` / `n_event_yolo` / `n_rgb_frames` / `n_rgb_yolo` — file counts inside the zip
- `zip_size_mb` from the file size on disk
- `split` — looked up from `splits.yaml` (train/val/test/unassigned)

Merges with any existing `catalog.yaml` so manually written `description` and
`drive_file_id` fields are preserved. Re-run any time you add new zips or update
`splits.yaml`.

```powershell
python 4channel_project/make_catalog.py
```

### Constant

`DRIVE_FOLDER_ID = "1pISIErXOx76xmCqkwhS3-azWOMlTKZMp"` — the public Google Drive
folder that holds the FRED sequence zips.

### What it reads from each zip (via `ZipSequence`)

- `ts_shift_us` — from `Event/events.raw.tmp_index` (already parsed by `ZipSequence._read_ts_shift`)
- `n_event_frames` — count of `Event/Frames/*.png` members
- `n_event_yolo` — count of `Event_YOLO/*.txt` members
- `n_rgb_frames` — count of `PADDED_RGB/*.jpg` members
- `zip_size_mb` — from `os.path.getsize`
- `split` — looked up from `data_from_fred/splits.yaml`

### Merge behaviour

On re-run, existing `description` and `drive_file_id` fields are **preserved** — only
auto-generated fields are updated. Edit these manually in `catalog.yaml` as needed.

### Function: `auto_split(train_pct, val_pct, test_pct)`

Auto-assigns all local `.zip` sequences to train/val/test by percentage.
Sequences are sorted numerically (reproducible). Writes `data_from_fred/splits.yaml`.

```powershell
python make_catalog.py --auto-split --train 70 --val 20 --test 10
```

### Function: `update_drive_ids(folder_id)`

Scans a public Google Drive folder via `gdrive.scan_folder()`, adds `drive_file_id`
to each matching entry in `catalog.yaml`. Preserves all other fields.

```powershell
python make_catalog.py --scan-drive               # no API key (uses gdown)
python make_catalog.py --scan-drive --api-key AIza...   # reliable API key method
```

### CLI flags

| Flag | Default | Effect |
|---|---|---|
| `--auto-split` | off | auto-assign splits, write splits.yaml |
| `--train N` | 70 | train % for --auto-split |
| `--val N` | 20 | val % for --auto-split |
| `--test N` | 10 | test % for --auto-split |
| `--scan-drive` | off | scan Drive folder, add drive_file_id to catalog |
| `--download-all` | off | download all zips from Drive (no API key, uses gdown) |
| `--folder-id ID` | DRIVE_FOLDER_ID | override Drive folder ID |
| `--api-key KEY` | None | Google API key for reliable folder listing |

---

## gdrive.py

**Purpose:** Google Drive folder scan and bulk/lazy zip download. Requires `pip install gdown`.

### Function: `scan_folder(folder_id, api_key=None)`

Lists `.zip` files in a public Google Drive folder. Tries three methods in order:
1. **gdown** (`skip_download=True`) — reliable, no API key needed (default)
2. **Drive API v3** — if `api_key` is provided; handles large folders with pagination
3. **HTML parsing** — last resort fallback; prints fix instructions if it fails

Returns `{seq_id: drive_file_id}` dict. Called by `make_catalog.py --scan-drive`.

### Function: `download_folder_all(folder_id, data_dir)`

Downloads all `.zip` files from the Drive folder to `data_dir/` in one call using
`gdown.download_folder(resume=True)`. No API key needed. Already-local files are
skipped automatically via `resume=True`. Called by `make_catalog.py --download-all`.

### Function: `download_zip(seq_num, drive_file_id, data_dir)`

Downloads `{seq_num}.zip` to `data_dir/` using `gdown.download()`. Verifies the file
exists after download and prints its size. The caller (`_ensure_zip`) checks for local
presence first — this function is only called when the zip is genuinely missing.

---

## train_4ch_yolo.py

**Purpose:** Modify YOLO v11's first layer to accept 4 channels and train on the generated dataset.

### Class: `FourChannelDroneDataset`

PyTorch Dataset that loads 4-channel PNG files and YOLO labels.
(Used for manual training loops; standard Ultralytics training uses the native loader.)

**`__init__(images_dir, labels_dir, img_size)`**
- Finds all `.png` files in `images_dir`
- Stores path to labels directory

**`__getitem__(idx)`**
- Opens RGBA PNG via PIL → shape `(H, W, 4)` uint8
- Transposes and normalises → `(4, 720, 1280)` float32 [0,1]
- Resizes to `(4, IMG_SIZE, IMG_SIZE)` using bilinear interpolation
- Loads corresponding `.txt` label
- Returns `(tensor, labels)`

**`_resize_channels(channels, size)`**
- Resizes each of the 4 channels independently using PIL
- Maintains float32 [0,1] range

**`_load_label(lbl_path)`**
- Reads YOLO `.txt` label file
- Returns tensor shape `(N, 5)` — N boxes, each `[class, cx, cy, w, h]`
- Returns empty `(0, 5)` tensor if file is empty or missing

### Function: `collate_fn(batch)`

Custom PyTorch collate function needed because each image can have a different number of bounding boxes.

Adds batch index as first column: `[batch_idx, class, cx, cy, w, h]`

### Function: `patch_yolo_input_channels(model, n_channels=4)`

Safety patch — modifies the first `Conv2d` layer to accept `n_channels` inputs
if it hasn't already been set by Ultralytics from `dataset.yaml`.

In normal operation (`channels: 4` in yaml), Ultralytics sets this automatically
and the function detects `old_in == n_channels` and skips. The patch runs only
when loading a model that was not originally built with 4-channel input.

```python
# Old first layer (standard YOLO):
Conv2d(in_channels=3, out_channels=16, kernel_size=3, ...)

# After patch / yaml-native loading:
Conv2d(in_channels=4, out_channels=16, kernel_size=3, ...)
```

Confirmed in training output: `[4, 16, 3, 2]` at layer 0 ✓

Weight initialisation strategy:
- Copies existing weights for the channels that overlap
- Initialises new channels with the mean of existing weights
- This preserves as much pretrained knowledge as possible

### Function: `train_with_ultralytics()`

Uses Ultralytics YOLO API for training.

1. Checks for `runs/fred_4channel/weights/last.pt` — resumes if found
2. Otherwise loads YOLO v11 nano pretrained weights and patches first layer
3. Calls `model.train()` with settings from `config.py`
4. Saves `best.pt` (best mAP50) and `last.pt` (latest epoch) automatically
5. Also saves a checkpoint every 10 epochs (`save_period=10`)

Augmentation settings (event-camera appropriate):
- `hsv_h=0, hsv_s=0` — no colour shifts (event frames have no colour)
- `hsv_v=0.3` — slight brightness variation
- `fliplr=0.5` — horizontal flip
- `mosaic=0.5` — mosaic augmentation

**Windows note:** Set `KMP_DUPLICATE_LIB_OK=TRUE` before running to avoid
OpenMP DLL conflict (`OMP Error #15`) between conda's MKL and PyTorch's OpenMP.

---

## evaluate.py

**Purpose:** Measure mAP50 of the trained model and compare against the FRED paper baseline (87.68%).

### Function: `evaluate(model_path)`

Runs `model.val()` on the validation split and prints:

```
mAP50      : XX.XX%
mAP50:95   : XX.XX%
Precision  : XX.XX%
Recall     : XX.XX%

Paper baseline (1-channel): 87.68%
Improvement: +X.XX% ✓
```

### Function: `run_ablation()`

Tests each channel combination separately to identify which channel contributes most:

| Label | Channels | Description |
|---|---|---|
| A | [Ch1] | Positive polarity only |
| B | [Ch1, Ch2] | Polarity pair |
| C | [Ch1, Ch2, Ch3] | Polarity + rotor |
| D | [Ch1, Ch2, Ch3, Ch4] | Full proposal |
| E | [Ch3] | Rotor map alone |
| F | [Ch4] | Time surface alone |

Results show exactly how much each channel contributes to the final mAP50.

### Function: `visualize_predictions(model_path, n_samples)`

Saves a 2×2 panel image for each validation sample showing all 4 channels with the predicted bounding box drawn on each.

Output saved to `./predictions/` folder.

---

## colab_run.ipynb

**Purpose:** Google Colab notebook — runs the entire pipeline from Drive mount to final evaluation in 10 cells.

| Cell | What it does |
|---|---|
| 1 | Install ultralytics, mount Drive, verify GPU |
| 2 | Copy project files from Drive to fast local SSD |
| 3 | Verify all paths and files exist |
| 4 | Test EVT3 reader on first window |
| 5 | Test noise filters |
| 6 | Generate and display all 4 channels visually |
| 7 | Build full training dataset |
| 8 | Train YOLO |
| 9 | Evaluate vs paper baseline |
| 10 | Save trained model back to Drive |

No manual path changes needed — `config.py` auto-detects Colab.

---

## tools/

Diagnostic and visualization utilities. All scripts must be run from `4channel_project/`
as `python tools/<script>.py`. Each script adds the parent directory to `sys.path` so
it can import the core modules (`config`, `evt3_reader`, `zip_utils`, etc.).

| Script | Purpose |
|---|---|
| `sync_check.py` | Side-by-side Event/Frames+label vs PADDED_RGB+label; prints Δcx/Δcy |
| `raw_label_check.py` | Render frames from `events.raw`, overlay Event_YOLO bbox; verify sync |
| `raw_to_movie.py` | Compare Event/Frames/ PNG (left) vs raw reconstruction (right) |
| `verify_frames.py` | Pixel-level MAE check between Frames/ PNGs and raw reconstruction |
| `view_raw_events.py` | Live viewer for raw event stream with annotated/removed frame highlights |
| `debug_filter_preview.py` | 4-channel before/after refractory filter comparison PNG |
| `find_offset.py` | Print first EVT3 timestamp vs first Frames/ filename to find ts_shift |
| `inspect_raw.py` | EVT3 data quality report: hot pixels, rate spikes, polarity bias |
| `make_filter_movie.py` | 2×2 grid video comparing raw vs filtered event channels |

### tools/raw_label_check.py

Renders 33ms windows directly from `events.raw` (no pre-extracted PNGs) and overlays
the matching `Event_YOLO/` bounding box. Useful for verifying timestamp sync after
any change to `ts_shift_us` or the EVT3 reader.

Banner colours:
- **GREEN** — label matched, box drawn; prints `cx cy w h Δt`
- **ORANGE** — label file exists but empty (drone out of frame)
- **RED** — no Event_YOLO file within ±16ms
- **GREY** — window outside annotated range

```powershell
cd 4channel_project
python tools/raw_label_check.py                  # sequence 7, full run
python tools/raw_label_check.py --start 9.8     # jump to drone segment
python tools/raw_label_check.py --seq 4         # sequence 4 (from 4.zip)
python tools/raw_label_check.py --save out.mp4  # also save video
```
Controls: `SPACE`=pause/resume  `A/←`=prev  `D/→`=+10  `Q/ESC`=quit

### Adding new tools

Every script in `tools/` must include this at the top (after the docstring):
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

---

## Data formats used across files

### Structured event array (output of evt3_reader, input to filters and channels)
```python
dtype = [('x', uint16), ('y', uint16), ('t', int64), ('p', uint8)]
shape = (N,)   # N = number of events in window
```

### Channel stack (output of channels.py)
```python
dtype = float32
shape = (4, 720, 1280)   # 4 channels, height, width
values = [0.0, 1.0]      # normalized
```

### Saved image format (output of dataset_builder.py)
```python
# Saved as RGBA PNG — PIL mode='RGBA', uint8 per channel
shape = (720, 1280, 4)   # H, W, C — PIL/numpy convention
dtype = uint8             # values 0–255
# Loaded back: divide by 255 → float32 [0,1], transpose → (4, H, W)
```

### YOLO label file (one line per drone)
```
0 0.312500 0.208333 0.054688 0.076389
│ │        │        │        └── height (normalized)
│ │        │        └────────── width (normalized)
│ │        └─────────────────── cy center y (normalized)
│ └──────────────────────────── cx center x (normalized)
└────────────────────────────── class id (0 = drone)
```

### YOLO training output
```
runs/fred_4channel/
├── weights/
│   ├── best.pt    ← best validation mAP50
│   └── last.pt    ← last epoch
├── results.csv    ← training metrics per epoch
└── plots/         ← loss curves, PR curve, confusion matrix
```

---

## Known Issues and Fixes

| Issue | Symptom | Fix |
|---|---|---|
| OpenMP DLL conflict | `OMP: Error #15` on Windows | `$env:KMP_DUPLICATE_LIB_OK="TRUE"` |
| Old .npy dataset | `No images found` error | Delete `dataset/` folder and rerun `dataset_builder.py` |
| Wrong channel count | Layer 0 shows `[3, 16, 3, 2]` | Ensure `channels: 4` in `dataset.yaml` and no old checkpoint loaded |
| fbgemm.dll error | PyTorch DLL load failure | Install Visual C++ Redistributable from aka.ms/vs/17/release/vc_redist.x64.exe |
| Old train/val subdir layout | `train.txt` not found | Delete `dataset/` and rebuild with `python dataset_builder.py` |
| Sequence not found | `FileNotFoundError: Zip file not found` | Add `N.zip` to `data_from_fred/` or update `splits.yaml` |
| Drive scan finds no files | `HTML parsing found no .zip files` | `--scan-drive` now uses gdown by default (no API key needed); if still failing, add `--api-key AIza...` |
| Drive download arg error | `unexpected keyword argument 'remaining_ok'` | Update gdrive.py — fixed in commit a995d75 |
| Drive download fails | `ImportError: gdown is required` | `pip install gdown` |
| Drive file IDs missing | `no drive_file_id in catalog` | Run `python make_catalog.py --scan-drive` |

---

*Code Guide — 4-Channel Drone Detection Project — Updated June 2026*
