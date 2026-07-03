"""
debug_filter_preview.py â€” Visual before/after filter comparison.

Reads one 33ms window, generates 4-channel images before and after
the refractory filter, and saves a side-by-side PNG.

Usage:
    cd 4channel_project
    python debug_filter_preview.py
"""

import sys
import os
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ai_drone/
sys.path.insert(0, os.path.join(_ROOT, 'common'))
sys.path.insert(0, os.path.join(_ROOT, '4channel_project'))

from evt3_reader import EVT3Reader
from filters import fast_filter
from channels import generate_channels, channels_to_rgb_preview

_DATA    = os.path.join(_ROOT, 'data_from_fred')
RAW_FILE = os.path.join(_DATA, '7', 'Event', 'events.raw')

try:
    import cv2
except ImportError:
    print("ERROR: install opencv-python first:  pip install opencv-python")
    sys.exit(1)

# â”€â”€ Settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Start time â€” drone is visible from ~9.87s in sequence 7
T_START_US = 9_870_000
WINDOW_US  = 33_333
T_END_US   = T_START_US + WINDOW_US

OUT_PATH   = "./debug_before_after.png"

# â”€â”€ Load events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

print(f"Reading window t={T_START_US/1e6:.3f}s â€“ {T_END_US/1e6:.3f}s ...")
reader = EVT3Reader(RAW_FILE)
raw_events = reader.read_window(T_START_US, T_END_US)
print(f"  Raw events : {len(raw_events):,}")

# â”€â”€ Filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

clean_events = fast_filter(raw_events)
removed      = len(raw_events) - len(clean_events)
print(f"  After filter: {len(clean_events):,}  "
      f"(removed {removed:,} = {100*removed/max(len(raw_events),1):.1f}%)")

# â”€â”€ Generate 4-channel images â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

print("\nGenerating channels...")
ch_before = generate_channels(raw_events,   T_START_US, T_END_US)
ch_after  = generate_channels(clean_events, T_START_US, T_END_US)

img_before = channels_to_rgb_preview(ch_before)
img_after  = channels_to_rgb_preview(ch_after)

# â”€â”€ Add title banners â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def add_banner(img, title, subtitle=""):
    bar = np.zeros((50, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title,    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    cv2.putText(bar, subtitle, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    return np.vstack([bar, img])

img_before = add_banner(
    img_before,
    f"BEFORE filter  [{len(raw_events):,} events]",
    f"t = {T_START_US/1e6:.3f}s  |  All raw events including noise"
)
img_after = add_banner(
    img_after,
    f"AFTER filter   [{len(clean_events):,} events]",
    f"Removed {removed:,} events ({100*removed/max(len(raw_events),1):.1f}%)  |  Clean signal only"
)

# â”€â”€ Combine side by side â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compare = np.hstack([img_before, img_after])
cv2.imwrite(OUT_PATH, compare)
print(f"\nSaved: {os.path.abspath(OUT_PATH)}")
print("Open that PNG to see before vs after the filter.")
