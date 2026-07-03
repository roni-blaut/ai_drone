"""
find_offset.py â€” Find the time offset between events.raw and Event/Frames/.

Prints the first real timestamp in events.raw and the first Frames/ filename.
The difference is the exact offset needed to align them.

Usage:
    cd 4channel_project
    python find_offset.py
"""
import sys, os
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ai_drone/
sys.path.insert(0, os.path.join(_ROOT, 'common'))
sys.path.insert(0, os.path.join(_ROOT, '4channel_project'))

import argparse
from evt3_reader import EVT3Reader
from zip_utils import init_sequence, seq_glob

parser = argparse.ArgumentParser()
parser.add_argument('--seq', type=str, default='7', help='Sequence number (default: 7)')
args = parser.parse_args()

BASE       = os.path.join(_ROOT, 'data_from_fred', args.seq)
RAW_FILE   = os.path.join(BASE, 'Event', 'events.raw')
FRAMES_DIR = os.path.join(BASE, 'Event', 'Frames')

init_sequence(BASE)

# â”€â”€ First event timestamp from events.raw â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

print("Reading first events from events.raw...")
reader = EVT3Reader(RAW_FILE)

first_raw_t  = None
first_10     = []

for chunk in reader._iter_chunks():
    if len(chunk) == 0:
        continue
    for ev in chunk[:10]:
        first_10.append(int(ev['t']))
    first_raw_t = first_10[0]
    break

print(f"\n  First 10 raw timestamps:")
for i, t in enumerate(first_10):
    print(f"    [{i}]  {t:>12,} Âµs  =  {t/1e6:.6f} s")

# â”€â”€ First Frames/ filename â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

frames        = seq_glob(FRAMES_DIR, '*.png')
frame_ts      = sorted([int(os.path.basename(p).split('_frame_')[1][:-4]) for p in frames])
first_frame_t = frame_ts[0]
last_frame_t  = frame_ts[-1]

print(f"\n  First Frames/ file : Video_7_frame_{first_frame_t}.png  =  {first_frame_t/1e6:.6f} s")
print(f"  Last  Frames/ file : Video_7_frame_{last_frame_t}.png  =  {last_frame_t/1e6:.6f} s")
print(f"  Total Frames/      : {len(frames)} files")

# â”€â”€ Compute offset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

offset = first_raw_t - first_frame_t
print(f"\n  Offset (raw - frames)  = {offset:,} Âµs  =  {offset/1e6:.3f} s")
print(f"  â†’ raw timestamps are {abs(offset)/1e6:.3f}s {'AHEAD of' if offset > 0 else 'BEHIND'} Frames/ timestamps")
print()
if offset > 0:
    print(f"  To align: subtract {offset:,} Âµs from raw timestamps")
    print(f"            OR add   {offset:,} Âµs to Frames/ lookup")
else:
    print(f"  To align: add {abs(offset):,} Âµs to raw timestamps")
    print(f"            OR subtract {abs(offset):,} Âµs from Frames/ lookup")
