"""
PID Frequency Analysis v3 — annotation centroid FFT
====================================================
Instead of using event image centroids (which are saturated/symmetric),
run FFT directly on the ground-truth annotation bbox center time series.

If the drone has a 10Hz PID wobble, the annotation centroid will show
that oscillation on top of the overall flight trajectory.

Approach:
  1. Load all annotations (t, cx, cy)
  2. Fit + subtract the smooth flight trajectory (Savitzky-Golay)
  3. FFT the residual
  4. Look for 10Hz peak
"""

import numpy as np
import os
from scipy.signal import savgol_filter, welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COORDS_FILE = "/sessions/magical-gracious-brahmagupta/mnt/ai_drone/7/coordinates.txt"
OUT_DIR     = "/sessions/magical-gracious-brahmagupta/mnt/outputs"

# ── 1. Load all annotations ───────────────────────────────────────────────────
times, cxs, cys = [], [], []
with open(COORDS_FILE) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        time_part, coords_part = line.split(":")
        vals = [v.strip() for v in coords_part.split(",")]
        x1, y1, x2, y2 = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
        t = float(time_part.strip())
        times.append(t)
        cxs.append((x1 + x2) / 2)
        cys.append((y1 + y2) / 2)

times = np.array(times)
cxs   = np.array(cxs)
cys   = np.array(cys)

# Sort by time
order = np.argsort(times)
times = times[order]; cxs = cxs[order]; cys = cys[order]

print(f"Total annotations: {len(times)}")
print(f"Time range: {times[0]:.2f}s – {times[-1]:.2f}s")
dt_median = np.median(np.diff(times))
fs_ann = 1.0 / dt_median
print(f"Annotation rate: {fs_ann:.2f} Hz  (dt median={dt_median*1000:.1f} ms)")

# ── 2. Find segments of roughly uniform sampling ────────────────────────────
# Large gaps indicate drone not visible — split there
gaps = np.where(np.diff(times) > 0.1)[0]  # gaps > 100ms
print(f"\nLarge gaps (>100ms): {len(gaps)}")
if len(gaps) > 0:
    for g in gaps[:5]:
        print(f"  t={times[g]:.2f}s → gap {(times[g+1]-times[g])*1000:.0f}ms")

# ── 3. Analyse the full series with Savitzky-Golay detrending ────────────────
# SG filter with window covering ~0.5s = smooth slow motion, keep >2Hz oscillations
sg_window = max(int(fs_ann * 0.5), 5)
if sg_window % 2 == 0: sg_window += 1   # must be odd

trend_x = savgol_filter(cxs, sg_window, 3)
trend_y = savgol_filter(cys, sg_window, 3)

resid_x = cxs - trend_x
resid_y = cys - trend_y

print(f"\nSG window: {sg_window} samples ({sg_window/fs_ann:.2f}s)")
print(f"Residual X: std={resid_x.std():.3f}px  range=[{resid_x.min():.2f}, {resid_x.max():.2f}]")
print(f"Residual Y: std={resid_y.std():.3f}px  range=[{resid_y.min():.2f}, {resid_y.max():.2f}]")

# ── 4. FFT on full residual ────────────────────────────────────────────────────
n = len(resid_x)
fft_x = np.abs(np.fft.rfft(resid_x * np.hanning(n)))
fft_y = np.abs(np.fft.rfft(resid_y * np.hanning(n)))
freqs = np.fft.rfftfreq(n, d=dt_median)

print("\nTop FFT peaks (full sequence) — X residual:")
top5 = np.argsort(fft_x)[-8:][::-1]
for i in top5:
    print(f"  {freqs[i]:.2f} Hz  amp={fft_x[i]:.3f}")

print("\nTop FFT peaks (full sequence) — Y residual:")
top5 = np.argsort(fft_y)[-8:][::-1]
for i in top5:
    print(f"  {freqs[i]:.2f} Hz  amp={fft_y[i]:.3f}")

# ── 5. Welch PSD for better spectral resolution ──────────────────────────────
nperseg = min(256, len(resid_x)//4)
fx_w, px_w = welch(resid_x, fs=fs_ann, nperseg=nperseg)
fy_w, py_w = welch(resid_y, fs=fs_ann, nperseg=nperseg)

print("\nWelch PSD peaks — X residual:")
top_w = np.argsort(px_w)[-8:][::-1]
for i in top_w:
    print(f"  {fx_w[i]:.2f} Hz  PSD={px_w[i]:.4f}")

# Check 10Hz SNR in Welch
band = (fx_w >= 8) & (fx_w <= 12)
noise = (fx_w >= 2) & (fx_w <= 6)
snr_x_w = px_w[band].max() / px_w[noise].mean() if band.any() and noise.any() else 0
snr_y_w = py_w[band].max() / py_w[noise].mean() if band.any() and noise.any() else 0
print(f"\nWelch 10Hz SNR — X: {snr_x_w:.2f}x   Y: {snr_y_w:.2f}x")

# ── 6. Per-segment analysis (split at gaps > 50ms) ─────────────────────────
print("\n── Per-segment analysis ──────────────────────────")
gap_idx = list(np.where(np.diff(times) > 0.05)[0] + 1)
seg_starts = [0] + gap_idx
seg_ends   = gap_idx + [len(times)]
segs = [(s, e) for s, e in zip(seg_starts, seg_ends) if (e - s) >= 60]
print(f"Segments >= 60 samples: {len(segs)}")

best_snr = 0; best_seg = None; best_freq = None
for s, e in segs:
    t_s = times[s:e]; cx_s = cxs[s:e]; cy_s = cys[s:e]
    dt_s = np.median(np.diff(t_s)); fs_s = 1.0/dt_s
    if fs_s < 15: continue   # need Nyquist > 10Hz
    
    sg_w = max(int(fs_s * 0.5), 5)
    if sg_w % 2 == 0: sg_w += 1
    if sg_w >= len(cx_s): continue
    
    tr_x = savgol_filter(cx_s, sg_w, 3)
    tr_y = savgol_filter(cy_s, sg_w, 3)
    rx = cx_s - tr_x; ry = cy_s - tr_y
    
    npers = min(128, len(rx)//4)
    if npers < 16: continue
    fw, px = welch(rx, fs=fs_s, nperseg=npers)
    fw, py = welch(ry, fs=fs_s, nperseg=npers)
    
    b = (fw >= 7) & (fw <= 13)
    ns = (fw >= 1.5) & (fw <= 5)
    if not b.any() or not ns.any(): continue
    
    snr = max(px[b].max(), py[b].max()) / ((px[ns].mean()+py[ns].mean())/2)
    peak_f = fw[b][np.argmax(px[b])]
    
    if snr > best_snr:
        best_snr = snr; best_seg = (s,e); best_freq = peak_f
    
    if snr > 1.5:
        dur = t_s[-1] - t_s[0]
        print(f"  t={t_s[0]:.1f}-{t_s[-1]:.1f}s ({dur:.1f}s)  SNR={snr:.2f}x  peak={peak_f:.2f}Hz  std_x={rx.std():.2f}px")

print(f"\nBest segment SNR: {best_snr:.2f}x at {best_freq:.2f}Hz")

# ── 7. Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle("FRED Sequence 7 — PID Frequency Analysis (Annotation Centroid)", fontsize=13)

# Raw centroid
ax = axes[0,0]
ax.plot(times - times[0], cxs, lw=0.5, label='cx'); ax.plot(times-times[0], cys, lw=0.5, label='cy')
ax.set_title("Annotation centroid over time"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (px)")
ax.legend()

# Residual X
ax = axes[0,1]
ax.plot(times - times[0], resid_x, lw=0.5)
ax.set_title("Residual X (after SG detrend)"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Residual (px)")

# Residual Y
ax = axes[0,2]
ax.plot(times - times[0], resid_y, lw=0.5)
ax.set_title("Residual Y (after SG detrend)"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Residual (px)")

# FFT X
ax = axes[1,0]
mask = freqs <= 15
ax.semilogy(freqs[mask], fft_x[mask] + 1e-9)
ax.axvline(10, color='r', linestyle='--', label='10 Hz')
ax.set_title("FFT — Residual X"); ax.set_xlabel("Frequency (Hz)"); ax.legend()

# Welch PSD X
ax = axes[1,1]
mask_w = fx_w <= 15
ax.semilogy(fx_w[mask_w], px_w[mask_w] + 1e-12)
ax.axvline(10, color='r', linestyle='--', label='10 Hz')
ax.set_title(f"Welch PSD — X  (10Hz SNR={snr_x_w:.1f}x)"); ax.set_xlabel("Frequency (Hz)"); ax.legend()

# Welch PSD Y
ax = axes[1,2]
ax.semilogy(fy_w[mask_w], py_w[mask_w] + 1e-12)
ax.axvline(10, color='r', linestyle='--', label='10 Hz')
ax.set_title(f"Welch PSD — Y  (10Hz SNR={snr_y_w:.1f}x)"); ax.set_xlabel("Frequency (Hz)"); ax.legend()

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "pid_annotation_fft.png")
plt.savefig(out_path, dpi=120)
print(f"\nPlot saved: {out_path}")

if best_snr > 2.0:
    print(f"\n✓  PID oscillation DETECTED: {best_freq:.1f} Hz (SNR={best_snr:.1f}x)")
elif best_snr > 1.3:
    print(f"\n~  Weak signal near {best_freq:.1f} Hz (SNR={best_snr:.1f}x) — possible PID but noisy")
else:
    print(f"\n✗  No clear PID frequency peak found (best SNR={best_snr:.2f}x at {best_freq:.2f}Hz)")
    print("   Annotation resolution may be too coarse to capture sub-pixel oscillations.")
