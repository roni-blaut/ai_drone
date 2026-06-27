"""
inspect_raw.py — General-purpose EVT3 data quality inspector.

Works on ANY events.raw file. Annotations file is optional.

Auto-detects junk by looking for:
  1. Scene change   — sudden event-rate spike at start (countdown, screen, flicker)
  2. Hot pixels     — single pixels firing constantly across entire recording
  3. Dead zones     — sensor regions that never fire
  4. Rate spikes    — bursts of noise at any point in time
  5. Polarity bias  — sustained >80% positive or negative (sensor fault)

Prints a verdict:  "Your clean data starts at t=X.Xs"

Usage:
    cd 4channel_project
    python inspect_raw.py                              # uses config.py RAW_FILE
    python inspect_raw.py --raw ../data_from_fred/7/Event/events.raw
    python inspect_raw.py --raw ../data_from_fred/7/Event/events.raw --ann ../data_from_fred/7/coordinates.txt
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evt3_reader import EVT3Reader
from config import RAW_FILE, IMG_W, IMG_H

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("ERROR: pip install matplotlib")
    sys.exit(1)

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--raw', type=str, default=RAW_FILE,
                    help='Path to events.raw')
parser.add_argument('--ann', type=str, default=None,
                    help='Path to coordinates.txt (optional)')
args = parser.parse_args()

print(f"Inspecting: {args.raw}")
print(f"File size : {os.path.getsize(args.raw)/1e6:.1f} MB\n")

# ── Load annotations (optional) ───────────────────────────────────────────────

ann_start = ann_end = None
if args.ann and os.path.exists(args.ann):
    ann_times = []
    with open(args.ann) as f:
        for line in f:
            if line.strip():
                t_sec = float(line.split(':')[0].strip())
                ann_times.append(t_sec)
    ann_start = min(ann_times)
    ann_end   = max(ann_times)
    print(f"Annotations: {len(ann_times)} frames  "
          f"t={ann_start:.2f}s – {ann_end:.2f}s\n")
else:
    print("No annotations file — running without ground truth.\n")

# ── Stream full file ──────────────────────────────────────────────────────────

print("Streaming file (may take ~60s for 127MB)...")

BUCKET_US   = 1_000_000
reader      = EVT3Reader(args.raw)

bucket_counts = {}
bucket_pos    = {}
bucket_neg    = {}
spatial_map   = np.zeros((IMG_H, IMG_W), dtype=np.float32)
# Per-second spatial concentration score (high = text/screen, low = natural scene)
bucket_concentration = {}

total_events = 0
t_first = t_last = None

for t_start, events in reader.iter_windows(BUCKET_US):
    if len(events) == 0:
        continue

    t_sec = int(t_start // BUCKET_US)
    n     = len(events)

    bucket_counts[t_sec] = n
    bucket_pos[t_sec]    = int((events['p'] == 1).sum())
    bucket_neg[t_sec]    = int((events['p'] == 0).sum())

    # Spatial concentration: what fraction of events are in the top-1% busiest pixels?
    sec_map = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    np.add.at(sec_map, (events['y'], events['x']), 1)
    np.add.at(spatial_map, (events['y'], events['x']), 1)

    flat = sec_map.flatten()
    if flat.max() > 0:
        top1_threshold  = np.percentile(flat[flat > 0], 99)
        top1_events     = sec_map[sec_map >= top1_threshold].sum()
        concentration   = top1_events / max(n, 1)
        bucket_concentration[t_sec] = float(concentration)

    total_events += n
    if t_first is None: t_first = t_start / 1e6
    t_last = (t_start + BUCKET_US) / 1e6

    if t_sec % 10 == 0:
        print(f"  t={t_sec:3d}s  {n/1e6:.2f}M events")

print(f"\nTotal events : {total_events:,}")
print(f"Time range   : {t_first:.2f}s – {t_last:.2f}s")

# ── Junk detection ────────────────────────────────────────────────────────────

times  = sorted(bucket_counts.keys())
counts = np.array([bucket_counts[t] for t in times])

mean_rate  = float(np.median(counts))   # median is robust to outliers
std_rate   = float(np.std(counts))

issues     = []   # list of (t_sec, severity, description)

for t in times:
    c    = bucket_counts[t]
    pos  = bucket_pos[t]
    neg  = bucket_neg[t]
    conc = bucket_concentration.get(t, 0)
    pct_pos = 100 * pos / max(c, 1)

    # Rule 1: event rate spike (>3σ above median)
    if c > mean_rate + 3 * std_rate:
        issues.append((t, 'HIGH', f"t={t}s  rate spike: {c/1e6:.1f}M events (mean={mean_rate/1e6:.1f}M)"))

    # Rule 2: spatial concentration >40% in top-1% pixels → text/screen/countdown
    if conc > 0.40:
        issues.append((t, 'HIGH', f"t={t}s  screen/text detected: {conc*100:.0f}% events in top-1% pixels"))

    # Rule 3: polarity heavily biased >80% one side
    if pct_pos > 80:
        issues.append((t, 'WARN', f"t={t}s  polarity bias: {pct_pos:.0f}% positive (possible sensor fault or bright flash)"))
    elif pct_pos < 20:
        issues.append((t, 'WARN', f"t={t}s  polarity bias: {100-pct_pos:.0f}% negative"))

# ── Hot pixel report ──────────────────────────────────────────────────────────

flat         = spatial_map.flatten()
hot_thresh   = np.percentile(flat[flat > 0], 99.9)
hot_mask     = spatial_map > hot_thresh
hot_yx       = np.argwhere(hot_mask)
dead_count   = int((spatial_map == 0).sum())

# ── Print verdict ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)

# Group issues by type
high_issues = [i for i in issues if i[1] == 'HIGH']
warn_issues = [i for i in issues if i[1] == 'WARN']

if high_issues:
    print(f"\n[!] {len(high_issues)} HIGH issues found:")
    for t, sev, msg in high_issues[:20]:
        print(f"    {msg}")
    if len(high_issues) > 20:
        print(f"    ... and {len(high_issues)-20} more")

if warn_issues:
    print(f"\n[~] {len(warn_issues)} warnings:")
    for t, sev, msg in warn_issues[:10]:
        print(f"    {msg}")

# Find the clean start: first second with NO high issues
high_times = sorted(set(t for t, sev, _ in high_issues if sev == 'HIGH'))
if high_times:
    # Find longest gap after last junk second
    last_junk = max(high_times)
    clean_start = last_junk + 1
    print(f"\n  Junk detected from t=0s to t={last_junk}s")
    print(f"  --> Recommended clean start: t={clean_start}s")
    if ann_start:
        print(f"  --> Annotation start       : t={ann_start:.1f}s")
        if abs(clean_start - ann_start) < 3:
            print(f"  --> MATCHES annotation start (within 3s)  [GOOD]")
        else:
            print(f"  --> WARNING: junk ends at {clean_start}s "
                  f"but annotations start at {ann_start:.1f}s")
else:
    print("\n  No junk detected — data looks clean from t=0s")

print(f"\n  Hot pixels  : {len(hot_yx):,} pixels fire >99.9% of the time")
print(f"  Dead pixels : {dead_count:,} ({100*dead_count/(IMG_W*IMG_H):.1f}% of sensor)")
print(f"  Polarity    : "
      f"{100*sum(bucket_pos.values())/max(total_events,1):.1f}% pos / "
      f"{100*sum(bucket_neg.values())/max(total_events,1):.1f}% neg")
print("=" * 60)

# ── Plots ─────────────────────────────────────────────────────────────────────

conc_vals = [bucket_concentration.get(t, 0) for t in times]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fname = os.path.basename(args.raw)
fig.suptitle(f"Data Quality — {fname}", fontsize=13, fontweight='bold')

# Plot 1: Event rate + junk markers
ax = axes[0, 0]
ax.plot(times, counts / 1e6, color='steelblue', linewidth=0.8, label='Event rate')
ax.axhline(mean_rate / 1e6, color='orange', linestyle='--', linewidth=1,
           label=f'Median {mean_rate/1e6:.1f}M')
ax.axhline((mean_rate + 3*std_rate) / 1e6, color='red', linestyle='--',
           linewidth=1, label='+3σ junk threshold')
if ann_start:
    ax.axvspan(ann_start, ann_end, alpha=0.12, color='green',
               label=f'Annotated {ann_start:.0f}s–{ann_end:.0f}s')
for t, sev, _ in high_issues:
    ax.axvline(t, color='red', alpha=0.3, linewidth=1.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Events/sec (M)")
ax.set_title("Event rate  (red lines = junk seconds)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Plot 2: Spatial concentration score
ax = axes[0, 1]
ax.plot(times, [c * 100 for c in conc_vals], color='purple', linewidth=0.8)
ax.axhline(40, color='red', linestyle='--', linewidth=1,
           label='40% = screen/text threshold')
if ann_start:
    ax.axvspan(ann_start, ann_end, alpha=0.12, color='green')
ax.set_xlabel("Time (s)")
ax.set_ylabel("% events in top-1% pixels")
ax.set_title("Spatial concentration\n(high = text/screen/countdown, low = natural scene)")
ax.legend(fontsize=8)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

# Plot 3: Spatial heat map
ax = axes[1, 0]
im = ax.imshow(np.log1p(spatial_map), cmap='hot', aspect='auto')
plt.colorbar(im, ax=ax, label='log(events)')
for y, x in hot_yx[:8]:
    ax.plot(x, y, 'c+', markersize=8, markeredgewidth=2)
ax.set_title(f"Spatial heat map  "
             f"({len(hot_yx)} hot pixels marked cyan, "
             f"{dead_count:,} dead pixels)")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Plot 4: Polarity balance
ax = axes[1, 1]
pos_pct = [100 * bucket_pos[t] / max(bucket_counts[t], 1) for t in times]
ax.plot(times, pos_pct, color='tomato', linewidth=0.8, label='Positive %')
ax.plot(times, [100 - p for p in pos_pct], color='cornflowerblue',
        linewidth=0.8, label='Negative %')
ax.axhline(50, color='gray', linestyle='--', linewidth=1)
ax.axhline(80, color='red',  linestyle=':', linewidth=1, label='80% bias threshold')
ax.axhline(20, color='red',  linestyle=':', linewidth=1)
if ann_start:
    ax.axvspan(ann_start, ann_end, alpha=0.12, color='green')
ax.set_ylim(0, 100)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Polarity %")
ax.set_title("Polarity balance  (healthy = near 50/50)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "./raw_inspection.png"
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"\nPlot saved: {os.path.abspath(out)}")
plt.show()
