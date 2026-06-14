"""
filters.py — Noise filters for raw event streams.

Two filters applied in order (cheapest first):

  1. Refractory Period Filter
     Each pixel ignores a second event within tau microseconds of its last.
     Kills hot pixels. O(1) per event.

  2. Background Activity Filter (BAF)
     Keeps event only if a neighbour pixel fired within delta_t microseconds.
     Kills isolated thermal/photon noise. O(radius^2) per event.

Usage:
    from filters import refractory_filter, baf_filter

    events = refractory_filter(events, tau_us=1000)
    events = baf_filter(events, radius=3, delta_t_us=10000)
"""

import numpy as np
from config import IMG_W, IMG_H, REFRACTORY_US, BAF_RADIUS_PX, BAF_DELTA_US, DEBUG_MODE, DEBUG_SAMPLES

# Call counters — limit verbose debug output to the first DEBUG_SAMPLES windows
_refractory_debug_count = 0
_baf_debug_count        = 0


# ── Step 1: Refractory Period Filter ─────────────────────────────────────────

def refractory_filter(events, tau_us=None):
    """
    Remove events at a pixel that fired too recently.

    For each event at (x, y, t):
      - If (t - last_fire_time[y, x]) < tau_us  → discard
      - Otherwise → keep, update last_fire_time[y, x] = t

    Parameters
    ----------
    events  : structured numpy array with fields (x, y, t, p)
    tau_us  : refractory period in microseconds. Default from config.

    Returns
    -------
    filtered structured numpy array
    """
    if tau_us is None:
        tau_us = REFRACTORY_US

    # Per-pixel last fire timestamp (initialised to -infinity)
    last_t = np.full((IMG_H, IMG_W), -10_000_000, dtype=np.int64)

    keep = np.zeros(len(events), dtype=bool)

    for i, ev in enumerate(events):
        x, y, t = int(ev['x']), int(ev['y']), int(ev['t'])
        if (t - last_t[y, x]) >= tau_us:
            keep[i]      = True
            last_t[y, x] = t

    n_before = len(events)
    n_after  = keep.sum()
    print(f"  Refractory filter: {n_before:,} → {n_after:,} "
          f"(removed {n_before - n_after:,} = "
          f"{100*(n_before-n_after)/max(n_before,1):.1f}%)")

    global _refractory_debug_count
    if DEBUG_MODE and _refractory_debug_count < DEBUG_SAMPLES:
        rejected = events[~keep]
        if len(rejected) > 0:
            xy = np.column_stack([rejected['x'], rejected['y']])
            unique_xy, counts = np.unique(xy, axis=0, return_counts=True)
            top_n   = min(10, len(counts))
            top_idx = np.argsort(-counts)[:top_n]
            print(f"  [DEBUG] Top hot pixels (window #{_refractory_debug_count + 1}):")
            for i in top_idx:
                print(f"    pixel ({unique_xy[i, 0]:4d}, {unique_xy[i, 1]:4d})"
                      f" → {counts[i]:,} events removed")
        _refractory_debug_count += 1

    return events[keep]


# ── Step 2: Background Activity Filter (BAF) ─────────────────────────────────

def baf_filter(events, radius=None, delta_t_us=None):
    """
    Keep event only if a neighbour pixel fired recently.

    For each event at (x, y, t):
      Check all pixels in the (2*radius+1) x (2*radius+1) neighbourhood.
      If ANY neighbour fired within the last delta_t_us → keep.
      Otherwise → discard (isolated noise).

    Parameters
    ----------
    events     : structured numpy array with fields (x, y, t, p)
    radius     : spatial neighbourhood radius in pixels. Default from config.
    delta_t_us : time window in microseconds. Default from config.

    Returns
    -------
    filtered structured numpy array
    """
    if radius     is None: radius     = BAF_RADIUS_PX
    if delta_t_us is None: delta_t_us = BAF_DELTA_US

    # Per-pixel last fire timestamp
    last_t = np.full((IMG_H, IMG_W), -10_000_000, dtype=np.int64)

    keep = np.zeros(len(events), dtype=bool)

    for i, ev in enumerate(events):
        x, y, t = int(ev['x']), int(ev['y']), int(ev['t'])

        # Define neighbourhood bounds (clamped to image)
        x0 = max(0,      x - radius)
        x1 = min(IMG_W - 1, x + radius)
        y0 = max(0,      y - radius)
        y1 = min(IMG_H - 1, y + radius)

        # Check if any neighbour fired recently
        neighbourhood = last_t[y0:y1+1, x0:x1+1]
        if (t - neighbourhood).min() <= delta_t_us:
            keep[i] = True

        # Always update this pixel's last time
        last_t[y, x] = t

    n_before = len(events)
    n_after  = keep.sum()
    print(f"  BAF filter:         {n_before:,} → {n_after:,} "
          f"(removed {n_before - n_after:,} = "
          f"{100*(n_before-n_after)/max(n_before,1):.1f}%)")

    global _baf_debug_count
    if DEBUG_MODE and _baf_debug_count < DEBUG_SAMPLES:
        pct_kept = 100 * n_after / max(n_before, 1)
        print(f"  [DEBUG] BAF acceptance: {pct_kept:.1f}% passed  "
              f"({n_after:,} signal events, {n_before - n_after:,} noise)")
        _baf_debug_count += 1

    return events[keep]


# ── Combined filter pipeline ──────────────────────────────────────────────────

def apply_filters(events, tau_us=None, radius=None, delta_t_us=None):
    """
    Apply both filters in the correct order (refractory first, BAF second).

    Parameters
    ----------
    events     : raw structured numpy array from EVT3Reader
    tau_us     : refractory period in microseconds
    radius     : BAF spatial radius in pixels
    delta_t_us : BAF time window in microseconds

    Returns
    -------
    cleaned structured numpy array
    """
    if len(events) == 0:
        return events

    print(f"  Filtering {len(events):,} events...")
    events = refractory_filter(events, tau_us=tau_us)
    events = baf_filter(events, radius=radius, delta_t_us=delta_t_us)
    return events


# ── Fast version (no per-pixel BAF — just refractory) ─────────────────────────

def fast_filter(events, tau_us=None):
    """
    Refractory filter only — faster, good enough for training data generation.

    BAF is O(N * radius^2) which is slow for large event counts.
    Refractory alone removes ~60-70% of noise very cheaply.
    Use this when speed matters more than filter quality.
    """
    return refractory_filter(events, tau_us=tau_us)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    from evt3_reader import EVT3Reader

    fpath = sys.argv[1] if len(sys.argv) > 1 else "../data_from_fred/7/Event/events.raw"

    reader = EVT3Reader(fpath)
    print("Reading first 33ms window...")
    events = reader.read_window(0, 33333)
    print(f"Raw events: {len(events):,}")

    print("\nApplying filters...")
    filtered = apply_filters(events)
    print(f"\nFinal: {len(filtered):,} events remain")
