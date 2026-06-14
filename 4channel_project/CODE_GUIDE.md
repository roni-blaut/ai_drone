# 4-Channel Project — Code Guide

Complete explanation of every file, every function, and how data flows through the pipeline.

---
data from :
https://drive.google.com/drive/folders/1pISIErXOx76xmCqkwhS3-azWOMlTKZMp?usp=share_link

https://github.com/miccunifi/FRED/tree/main


## Pipeline Overview

```
events.raw (127MB raw binary)
      │
      ▼
evt3_reader.py       ← parse binary → structured event array (x, y, t, polarity)
      │
      ▼
filters.py           ← remove noise → clean event array
      │
      ▼
channels.py          ← generate 4 channels → numpy array (4, 720, 1280)
      │
      ▼
dataset_builder.py   ← save 4-ch RGBA PNG + YOLO labels → train/val folders
      │
      ▼
train_4ch_yolo.py    ← patch YOLO first layer → train → best.pt
      │
      ▼
evaluate.py          ← mAP50 vs paper baseline + ablation study
```

All settings live in `config.py`. No other file needs path changes.

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

## evt3_reader.py

**Purpose:** Parse the Prophesee EVT3 binary format into a structured numpy array of events.

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

**`_find_header_end()`**
- Internal: scans file for end of ASCII header (lines starting with `%`)
- Returns byte offset where binary data begins

**`_iter_chunks()`**
- Internal generator: reads file in chunks and decodes EVT3 words
- Maintains decoder state (current_y, time_high, current_t) across chunks
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

**Purpose:** Read the full `events.raw` file, generate 4 channels for every 33ms window, match with ground-truth annotations, and save as YOLO training data.

### Output structure

```
dataset/
├── images/
│   ├── train/   ← 4-channel RGBA PNG files (H×W×4, uint8)
│   └── val/
├── labels/
│   ├── train/   ← YOLO format .txt files
│   └── val/
└── dataset.yaml ← includes "channels: 4"
```

Images are saved as RGBA PNG (not .npy). Ultralytics reads `channels: 4` from
`dataset.yaml` natively and routes 4-channel images through the network correctly.

### Function: `load_annotations(coords_file)`

Parses `coordinates.txt` into a sorted list of `(time_us, x1, y1, x2, y2)`.

Converts annotation time from seconds to microseconds for alignment with event timestamps.

### Function: `find_annotation(annotations, t_start_us, t_end_us, max_gap_us=100000)`

For a given time window, finds the nearest annotation.

Uses `np.searchsorted` for fast binary search across annotation timestamps.
Returns `(x1, y1, x2, y2)` or `None` if no annotation within `max_gap_us` (100ms).

### Function: `bbox_to_yolo(x1, y1, x2, y2)`

Converts absolute pixel coordinates to YOLO normalized format:

```
cx = ((x1 + x2) / 2) / img_width     # center x, normalized 0–1
cy = ((y1 + y2) / 2) / img_height    # center y, normalized 0–1
w  = (x2 - x1) / img_width           # width, normalized 0–1
h  = (y2 - y1) / img_height          # height, normalized 0–1
```

### Function: `build_dataset(...)`

Main function. Iterates through every 33ms window in `events.raw`:

```
For each window (t_start, events):
  1. Skip if no events
  2. Find matching annotation (or note as negative example)
  3. Apply fast_filter (refractory only)
  4. Generate 4 channels → shape (4, 720, 1280) float32
  5. Assign to train or val (random split, 80/20)
  6. Save as RGBA PNG → images/train/ or images/val/
     (channels transposed to (H, W, 4), scaled ×255, saved via PIL)
  7. Save .txt label → labels/train/ or labels/val/
```

Each PNG is named `seq_t{timestamp}.png` so files sort in time order.
Each `.txt` label has one line: `0 cx cy w h` (class 0 = drone).
Empty `.txt` file = no drone in this frame.

### Function: `print_dataset_stats(output_dir)`

Prints summary after building:
```
train:  2400 images, 2200 with drone (92%)
val  :   600 images,  550 with drone (92%)
```

### Function: `_write_yaml(output_dir)`

Writes `dataset.yaml` that YOLO needs for training:
```yaml
path:  /absolute/path/to/dataset
train: images/train
val:   images/val
channels: 4
nc: 1
names: ['drone']
```

The `channels: 4` key is read by Ultralytics via:
```python
super().__init__(*args, channels=self.data.get("channels", 3), ...)
```
This sets the first Conv2d to 4 input channels without any manual patching.

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

---

*Code Guide — 4-Channel Drone Detection Project — Updated June 2026*
