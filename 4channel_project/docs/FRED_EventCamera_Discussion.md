# FRED Event Camera & Drone Detection — Discussion Summary

---

## 1. FRED Dataset Overview

**FRED (Florence RGB-Event Drone Dataset)** — [https://miccunifi.github.io/FRED/](https://miccunifi.github.io/FRED/)

- Multimodal dataset for drone detection, tracking, and trajectory forecasting
- Combines RGB video (30fps / 33ms) and event camera streams
- 7+ hours of annotated drone footage, 5 drone models
- Challenging scenarios: rain, high dynamic range, night, indoor
- Spatio-temporally synchronized RGB and event frames

---

## 2. Will Training with a Video/Movie Model Give Better Results Than Per-Frame YOLO?

**Yes, almost certainly — with nuances.**

### Why video models help:
- **Motion continuity** — drones are small and fast; velocity/trajectory across frames is a strong discriminator
- **Temporal consistency** — false positives (birds, noise) are rarely temporally consistent
- Single-frame YOLO at 33ms discards all temporal information

### Tradeoffs:
| Factor | Single-frame YOLO | Video Model |
|---|---|---|
| Latency | Very low | Higher |
| Training complexity | Simple | More complex |
| Edge deployment | Easy | Harder |
| Temporal context | None | Strong |

### Key insight:
The biggest gains come from **fusing event data temporally**, not just upgrading the RGB backbone. The event camera operates at microsecond resolution — far beyond what any 30fps video model can exploit.

---

## 3. Will 1ms Frame Rate (1000fps RGB) Help?

**Marginal gain, high cost.**

| | 30fps (33ms) | 1000fps (1ms) |
|---|---|---|
| Motion blur | High | Near zero |
| Light per frame | Normal | Very low |
| Data rate | Manageable | 33× heavier |
| Low-light performance | Decent | Poor |

### Core insight:
1ms RGB essentially reinvents what the event camera already does — but worse:
- Event camera: **microsecond resolution**, no light starvation, sparse/efficient output
- 1ms RGB: expensive, dark, bandwidth-heavy

### Video model behavior at high frame rates:
- At 1000fps, consecutive frames barely differ → temporal advantage of video models **shrinks**
- Video models are **more valuable at 30fps** where inter-frame motion is large and informative

### Recommended architecture:
1. Event stream for high-speed motion/timing
2. RGB at 30fps for appearance/texture
3. Fuse both with a temporal model

---

## 4. Will Higher Temporal Resolution from the Event Camera Help?

**Yes — but only with the right representation.**

### How event cameras work:
| | RGB Camera | Event Camera |
|---|---|---|
| Output | Full frames at fixed rate | Asynchronous pixel-level changes |
| Temporal resolution | 33ms | ~1 microsecond |
| Spatial resolution | High (dense) | Lower (sparse, changed pixels only) |

### Event accumulation strategies:
| Method | Pro | Con |
|---|---|---|
| Fixed time window (1ms) | High temporal resolution | Very sparse, hard for YOLO |
| Fixed event count | Consistent density | Variable time window |
| Voxel grid (time-binned) | Rich temporal structure | More complex |
| Event frame (33ms sum) | Dense, YOLO-friendly | Loses all timing info |

**Sweet spot:** 5–10ms accumulation windows — dense enough for detection, fine enough to capture motion.

---

## 5. What is a Voxel Grid?

**Voxel = Pixel + Time** (from "Volumetric Pixel")

### Raw event stream:
```
(x, y, t, polarity)
(x, y, t, polarity)
...
```

### Voxel grid = 3D array:
```
Shape: [Time_bins, Height, Width]

Example:
- Resolution: 346 × 260
- Time window: 25ms
- Bins: 5 (one bin = 5ms each)
→ Array: [5, 260, 346]
```

Each cell `[t_bin, y, x]` = count of events at pixel (x,y) during that time bin.

### Visual intuition — stack of sparse frames:
```
Bin 1 (0–5ms):   [dots where motion happened]
Bin 2 (5–10ms):  [dots shifted slightly]
Bin 3 (10–15ms): [dots shifted more]
...
```

- **Drone** → smooth, consistent trail across bins
- **Bird** → pulsing, alternating pattern from wing flaps

### Why not just sum to one frame?
Summing loses all timing information. The bird flapping signature disappears. The voxel grid preserves **when** events happened — your key drone vs. bird discriminator.

---

## 6. Full Pipeline Architecture (Custom Build)

### Detection target: Drones only (reject birds and other objects)
### Event representation: Raw events (x, y, t, polarity)
### Runtime: Demo offline → future edge deployment

---

### Drone vs. Bird Discriminating Features:
| Feature | Drone | Bird |
|---|---|---|
| Motion pattern | Smooth, mechanical, hovering | Flapping, organic, gliding |
| Event polarity pattern | Symmetric, rigid body | Asymmetric (wing flap bursts) |
| Trajectory | Straight lines, sharp turns | Curved, soaring, erratic |
| Event rate | Steady | Pulsed (~3–10Hz flapping) |

---

### Recommended Pipeline:

```
Raw Events (x,y,t,p)
        │
        ▼
  Voxel Grid (5-10ms bins)
        │
        ├──────────────────────────┐
        ▼                          ▼
  YOLO Detection            Temporal Classifier
  "Where is the object?"    "Drone or bird?"
        │                          │
        └──────────┬────────────────┘
                   ▼
            Final Decision
         (drone / not drone)
```

---

### Stage 1: Voxel Grid Accumulator
- Accumulate into 5–10ms time bins
- Shape: `[2T, H, W]` (separate positive/negative polarity)
- Preserves flapping signature for bird rejection

### Stage 2: YOLO Detection (spatial)
- YOLOv8n or YOLOv9t (nano/tiny)
- Input: voxel grid tensor
- Pre-train on FRED, fine-tune on own data
- Goal: find candidate bounding boxes only

### Stage 3: Temporal Classifier (bird rejection)
- Input: event time series inside bounding box, 100–300ms window
- Architecture: small 1D CNN or LSTM over polarity histogram per time bin
- Detects periodic polarity bursts = bird flapping signature
- Lightweight but powerful discriminator

### Stage 4: Trajectory Filter (optional)
- Kalman filter per tracked object
- Drones: predictable kinematic model
- Birds: higher jerk (rate of acceleration change)
- Kills sporadic false positives

---

### Edge Deployment Path:
| Stage | Edge-friendly approach |
|---|---|
| Voxel grid | Fixed-size tensor, CPU-friendly |
| YOLO detection | YOLOv8n, INT8 quantized, TensorRT |
| Temporal classifier | Tiny LSTM or 1D CNN |
| Kalman filter | Pure CPU, negligible cost |

**Target hardware:** NVIDIA Jetson Orin Nano or Hailo-8 accelerator

---

## 7. Why YOLO + Temporal Model, Not a Pure Video Model?

### The two-stage logic:
```
Video model — tries to do everything in one shot
    → Heavy, slow, hard to deploy

YOLO + temporal classifier — splits the problem
    → YOLO:  "where is the object?"    (fast, spatial)
    → LSTM:  "is it a drone or bird?"  (light, temporal)
```

### When pure video model makes sense:
- Powerful GPU, no edge requirement
- Large, texture-rich objects
- RGB only (no event camera)

### For this use case (small drones, event camera, edge target):
**Two-stage approach wins.**

---

## Immediate Next Steps

1. Write the **voxel grid accumulator** — get this right before any model work
2. **Visualize** voxel grids for drone vs. bird sequences — the difference is visible
3. Train **YOLO on event voxels** using FRED annotations
4. Build the **temporal classifier** — the bird rejection secret weapon

---

*Conversation exported from Claude — June 2026*
