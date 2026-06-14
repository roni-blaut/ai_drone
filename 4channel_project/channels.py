"""
channels.py — Generate the 4 physics-based input channels from filtered events.

Channel 1 — Positive polarity (leading edge)
    Events where brightness INCREASED.
    Shows where the drone is moving INTO.
    Physically: the bow wave of the drone.

Channel 2 — Negative polarity (trailing edge)
    Events where brightness DECREASED.
    Shows where the drone just WAS.
    Physically: the wake of the drone.

Channel 3 — Rotor frequency map
    Pixels that fired more than ROTOR_THRESHOLD times in the window.
    These are rotor blade crossings — the spinning motor signature.
    Nothing in nature produces this except spinning machinery.

Channel 4 — Time surface
    Most recent event timestamp per pixel, normalized 0→1.
    Recent = bright (1.0). Old = dark (0.0). No event = 0.
    Shows WHERE the action is happening RIGHT NOW.

Output: numpy array of shape (4, IMG_H, IMG_W) dtype float32
        Each channel independently normalized to [0, 1].

Usage:
    from channels import generate_channels
    ch = generate_channels(events, t_start_us, t_end_us)
    # ch.shape == (4, 720, 1280)
"""

import numpy as np
from config import IMG_W, IMG_H, ROTOR_THRESHOLD, DEBUG_MODE, DEBUG_SAMPLES

_generate_debug_count = 0


def generate_channels(events, t_start_us, t_end_us):
    """
    Generate all 4 channels from a filtered event array.

    Parameters
    ----------
    events    : structured numpy array (x, y, t, p) — already filtered
    t_start_us: window start timestamp (microseconds)
    t_end_us  : window end timestamp (microseconds)

    Returns
    -------
    numpy float32 array of shape (4, IMG_H, IMG_W)
    Values normalized to [0, 1] per channel.
    """
    # Filter to this window
    mask = (events['t'] >= t_start_us) & (events['t'] < t_end_us)
    evs  = events[mask]

    ch1 = _channel_positive_polarity(evs)
    ch2 = _channel_negative_polarity(evs)
    ch3 = _channel_rotor_map(evs)
    ch4 = _channel_time_surface(evs, t_start_us, t_end_us)

    # Stack into (4, H, W)
    stack = np.stack([ch1, ch2, ch3, ch4], axis=0).astype(np.float32)

    global _generate_debug_count
    if DEBUG_MODE and _generate_debug_count < DEBUG_SAMPLES:
        names = ["Positive", "Negative", "Rotor   ", "TimeSurf"]
        print(f"  [DEBUG] Channels from {len(evs):,} events "
              f"(window #{_generate_debug_count + 1}):")
        for i, name in enumerate(names):
            nonzero = int((stack[i] > 0).sum())
            print(f"    Ch{i+1} {name}: max={stack[i].max():.3f}  "
                  f"active_pixels={nonzero:,}")
        _generate_debug_count += 1

    return stack


# ── Channel 1: Positive polarity ─────────────────────────────────────────────

def _channel_positive_polarity(events):
    """
    Count positive polarity events per pixel.
    Positive = pixel brightness increased = drone edge moving INTO pixel.
    """
    frame = np.zeros((IMG_H, IMG_W), dtype=np.float32)

    pos = events[events['p'] == 1]
    if len(pos) > 0:
        np.add.at(frame, (pos['y'], pos['x']), 1.0)

    return _normalize(frame)


# ── Channel 2: Negative polarity ─────────────────────────────────────────────

def _channel_negative_polarity(events):
    """
    Count negative polarity events per pixel.
    Negative = pixel brightness decreased = drone edge moving AWAY FROM pixel.
    """
    frame = np.zeros((IMG_H, IMG_W), dtype=np.float32)

    neg = events[events['p'] == 0]
    if len(neg) > 0:
        np.add.at(frame, (neg['y'], neg['x']), 1.0)

    return _normalize(frame)


# ── Channel 3: Rotor frequency map ───────────────────────────────────────────

def _channel_rotor_map(events, threshold=None):
    """
    Pixels that fired more than ROTOR_THRESHOLD times = spinning motor.

    A pixel at 200Hz rotor frequency fires ~6 times in 33ms.
    A bird wing pixel fires ~1 time in 33ms.
    A tree pixel fires ~0-1 times in 33ms.

    Threshold of 5 captures drones, rejects birds and trees.
    """
    if threshold is None:
        threshold = ROTOR_THRESHOLD

    # Count total events per pixel (both polarities)
    count = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    if len(events) > 0:
        np.add.at(count, (events['y'], events['x']), 1.0)

    # Zero out everything below threshold
    rotor_map = count.copy()
    rotor_map[rotor_map < threshold] = 0.0

    return _normalize(rotor_map)


# ── Channel 4: Time surface ───────────────────────────────────────────────────

def _channel_time_surface(events, t_start_us, t_end_us):
    """
    Most recent event timestamp per pixel, normalized to [0, 1].

    Pixel that fired at t=t_end_us  → value = 1.0  (just happened)
    Pixel that fired at t=t_start_us → value = 0.0  (long ago)
    Pixel that never fired           → value = 0.0

    This tells YOLO: "this is where activity is happening RIGHT NOW"
    rather than just "this is where events happened at some point."
    """
    # Start all pixels at -1 (never fired)
    surface = np.full((IMG_H, IMG_W), -1.0, dtype=np.float32)

    if len(events) == 0:
        return np.zeros((IMG_H, IMG_W), dtype=np.float32)

    # Process events in time order (they should already be sorted)
    # For each pixel, keep only the MOST RECENT timestamp
    # Using a loop is slow but correct; for speed see vectorized version below
    for ev in events:
        x, y, t = int(ev['x']), int(ev['y']), int(ev['t'])
        if t > surface[y, x]:
            surface[y, x] = float(t)

    # Normalize: -1 (never fired) stays 0, fired timestamps → [0, 1]
    duration = float(t_end_us - t_start_us)
    result   = np.zeros((IMG_H, IMG_W), dtype=np.float32)

    fired = surface >= 0
    if fired.any():
        result[fired] = (surface[fired] - t_start_us) / duration
        result = np.clip(result, 0.0, 1.0)

    return result


def _channel_time_surface_fast(events, t_start_us, t_end_us):
    """
    Vectorized version of time surface — faster for large event counts.
    Uses the fact that later events overwrite earlier ones for same pixel.
    """
    surface = np.full((IMG_H, IMG_W), -1.0, dtype=np.float32)

    if len(events) == 0:
        return np.zeros((IMG_H, IMG_W), dtype=np.float32)

    # Vectorized assignment — later events in array naturally overwrite earlier
    # This works because events are time-ordered and np assignment overwrites
    surface[events['y'], events['x']] = events['t'].astype(np.float32)

    duration = float(t_end_us - t_start_us)
    result   = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    fired    = surface >= 0
    if fired.any():
        result[fired] = (surface[fired] - t_start_us) / duration
        result = np.clip(result, 0.0, 1.0)

    return result


# ── Normalization helper ──────────────────────────────────────────────────────

def _normalize(frame):
    """Normalize frame to [0, 1]. Returns zeros if frame is empty."""
    max_val = frame.max()
    if max_val > 0:
        return frame / max_val
    return frame


# ── Visualization helper ──────────────────────────────────────────────────────

def channels_to_rgb_preview(channels):
    """
    Create an RGB preview image from 4 channels for debugging.

    Layout: 2x2 grid
      [Ch1 positive | Ch2 negative]
      [Ch3 rotor    | Ch4 surface ]

    Returns numpy uint8 array of shape (IMG_H*2, IMG_W*2, 3)
    """
    import cv2

    def to_uint8(ch):
        return (ch * 255).astype(np.uint8)

    ch1 = to_uint8(channels[0])
    ch2 = to_uint8(channels[1])
    ch3 = to_uint8(channels[2])
    ch4 = to_uint8(channels[3])

    top    = np.hstack([ch1, ch2])
    bottom = np.hstack([ch3, ch4])
    grid   = np.vstack([top, bottom])

    # Convert to BGR for OpenCV display
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)

    # Add labels
    h, w = IMG_H, IMG_W
    labels = [
        ((10, 30),  "Ch1: Positive polarity (leading edge)"),
        ((w+10, 30), "Ch2: Negative polarity (trailing edge)"),
        ((10, h+30), "Ch3: Rotor frequency map"),
        ((w+10, h+30), "Ch4: Time surface"),
    ]
    for (x, y), text in labels:
        cv2.putText(grid_bgr, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    return grid_bgr


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, '..')
    from evt3_reader import EVT3Reader
    from filters import fast_filter

    fpath = sys.argv[1] if len(sys.argv) > 1 else "../data_from_fred/7/Event/events.raw"

    reader = EVT3Reader(fpath)

    # Test on annotation time range (drone is present from t=9.87s)
    T_START = 9_870_000   # 9.87s in microseconds
    T_END   = T_START + 33_333

    print(f"Reading window t={T_START/1e6:.3f}s – {T_END/1e6:.3f}s...")
    events = reader.read_window(T_START, T_END)
    print(f"  Raw events: {len(events):,}")

    events = fast_filter(events)
    print(f"  After filter: {len(events):,}")

    print("\nGenerating 4 channels...")
    channels = generate_channels(events, T_START, T_END)
    print(f"  Output shape: {channels.shape}")
    for i, name in enumerate(["Positive", "Negative", "Rotor", "TimeSurface"]):
        ch = channels[i]
        n_nonzero = (ch > 0).sum()
        print(f"  Ch{i+1} {name:12s}: max={ch.max():.3f}  "
              f"nonzero_pixels={n_nonzero:,}")

    # Save preview
    try:
        import cv2
        preview = channels_to_rgb_preview(channels)
        out_path = "./channel_preview.png"
        cv2.imwrite(out_path, preview)
        print(f"\nPreview saved: {out_path}")
    except ImportError:
        print("\n(install opencv-python to see preview)")
