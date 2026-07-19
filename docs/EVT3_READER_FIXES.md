# EVT3 Reader Fixes

Three bugs were found and fixed related to reading Prophesee EVT3 raw files
and their companion sequence zips: two in `evt3_reader.py` causing incorrect
timestamps, and one in `zip_utils.py` causing zip member lookups to fail for
a non-standard zip layout (affecting `EVT3Reader` and every other zip reader
alike, since they all go through `ZipSequence`).

---

## Fix 1 — 24-bit Timestamp Rollover

### The problem

The EVT3 format encodes timestamps using two word types:

| Word type | Bits used | What it sets |
|---|---|---|
| `TIME_HIGH` (0x8) | bits [11:0] | high 12 bits of timestamp → bits [23:12] |
| `TIME_LOW`  (0x6) | bits [11:0] | low  12 bits of timestamp → bits [11:0]  |

Total timestamp width = **24 bits**.

Maximum value = 2²⁴ − 1 = **16,777,215 µs = 16.777 seconds**.

For a 118-second recording, the counter resets to 0 **seven times**:

```
t =  0.0s – 16.8s  → timestamps 0 → 16,777,215
t = 16.8s – 33.6s  → timestamps 0 → 16,777,215  (rollover 1)
t = 33.6s – 50.3s  → timestamps 0 → 16,777,215  (rollover 2)
t = 50.3s – 67.1s  → timestamps 0 → 16,777,215  (rollover 3)
t = 67.1s – 83.9s  → timestamps 0 → 16,777,215  (rollover 4)
t = 83.9s – 100.7s → timestamps 0 → 16,777,215  (rollover 5)
t = 100.7s – 118s  → timestamps 0 → 16,777,215  (rollover 6 & 7)
```

**Without rollover detection:** events from t=9.8s, t=26.6s, t=43.4s, t=60.1s,
t=76.9s, t=93.6s, and t=110.4s all have the same raw timestamp (~9.8s).
When accumulated into one frame → **multiple drone positions overlap**.

### The fix

Detect when `TIME_HIGH` decreases (counter wrapped back to 0) and add
2²⁴ = 16,777,216 µs to a running offset:

```python
ROLLOVER_ADD = 1 << 24   # 16,777,216 µs

elif word_type == 0x8:          # TIME_HIGH
    new_time_high = int(word & 0xFFF) << 12
    if new_time_high < prev_time_high:      # rollover detected
        rollover_offset += ROLLOVER_ADD
    prev_time_high = new_time_high
    time_high  = new_time_high
    current_t  = rollover_offset + time_high | (current_t & 0xFFF)

elif word_type == 0x6:          # TIME_LOW
    current_t  = rollover_offset + time_high | int(word & 0xFFF)
```

**Result:** timestamps now increase monotonically from 0 to ~118,000,000 µs
across the full recording. Each drone position appears in exactly one frame.

---

## Fix 2 — Clock Offset (ts_shift_us)

### The problem

The Prophesee SDK applies a clock synchronisation offset (`ts_shift_us`) when
generating derived files such as `Event/Frames/` and `Event_YOLO/`.

This offset is stored in the companion index file:
```
7/Event/events.raw.tmp_index
  % ts_shift_us 1163264       ← 1,163,264 µs = 1.163 seconds
```

**Without correction:** a raw event physically recorded at time T has timestamp
`T + 1,163,264 µs` in events.raw, but the matching `Frames/` PNG is named with
timestamp `T`. The two timelines are **1.163 seconds apart**.

This appeared visually as:
- The raw reconstruction showing the same content as `Frames/` but **~1 second late**
- `verify_frames.py` finding no matching `Frames/` PNG for any raw window

### The fix

`EVT3Reader.__init__` reads `ts_shift_us` from the `.tmp_index` file:

```python
def _read_ts_shift(self):
    index_path = self.filepath + '.tmp_index'
    if not os.path.exists(index_path):
        return 0
    with open(index_path, 'rb') as f:
        for line in f:
            if not line.startswith(b'%'):
                break
            if b'ts_shift_us' in line:
                return int(line.decode().split()[-1])
    return 0
```

Scripts that compare raw events to `Frames/` (e.g. `raw_to_movie.py`) then:
1. **Skip** the first `ts_shift_us` µs of raw data (pre-recording junk)
2. **Convert** raw window time → Frames/ time: `frames_t = raw_t - ts_shift_us`

```python
SKIP_US = reader.ts_shift_us          # 1,163,264 µs
t_start_us = SKIP_US                  # start reading here
...
frames_t = t_start - SKIP_US          # convert for Frames/ filename lookup
orig_path, _ = nearest_frame(frames_t + WINDOW_US)
```

**Result:** raw reconstruction and `Frames/` PNG show the same content at the
same displayed timestamp. `verify_frames.py` reports MAE < 10 across all frames.

---

## Fix 3 — Wrapped Zip Folder Structure

### The problem

`ZipSequence` (`common/zip_utils.py`) converts filesystem paths to zip member
names by taking the path relative to the sequence directory
(`_to_member()`), assuming every sequence zip stores its files at the zip
root — `Event/events.raw`, `coordinates.txt`, etc. — same as the
`common/zip_utils.py` docstring states.

`data_from_fred/2.zip` breaks that assumption: it nests **all** of its files
one level deeper, under an extra `2/` folder (`2/coordinates.txt`,
`2/Event/events.raw`, ...), while every other sequence zip (0, 1, 3-49) is
laid out normally.

**Without correction:** `_to_member()` computes `coordinates.txt` for
sequence 2's annotation file, but the zip only contains `2/coordinates.txt`
— the lookup misses, `_find_coords()` falls back to the plain
`coordinates.txt` path (also missing), and `load_annotations()` crashes with:
```
KeyError: "There is no item named 'coordinates.txt' in the archive"
```
`EVT3Reader.open_binary()` and `ZipSequence._read_ts_shift()` (which reads
`Event/events.raw.tmp_index`) would fail the same way for this sequence,
since both go through `_to_member()` / a hardcoded root-relative member name.

### The fix

`ZipSequence.__init__` calls a new `_detect_root_prefix()` after building
`self._names`:

```python
def _detect_root_prefix(self):
    if 'coordinates.txt' in self._names or 'Event/events.raw' in self._names:
        return ''
    tops = {n.split('/', 1)[0] for n in self._names}
    if len(tops) == 1:
        prefix = next(iter(tops)) + '/'
        if any(n.startswith(prefix) for n in self._names):
            return prefix
    return ''
```

If the well-known root markers exist directly in the zip, behavior is
unchanged (`self._root_prefix = ''`). Otherwise, if every member shares one
common top-level path component, that component is treated as an implicit
wrapper directory and stored as `self._root_prefix`. `_to_member()` and
`_read_ts_shift()` both prepend `self._root_prefix` when resolving member
names, so every other method (`open_binary`, `open_lines`, `read_bytes`,
`imread`, `glob`, `exists`) picks up the fix automatically since they all
funnel through `_to_member()`.

**Result:** sequence 2 loads annotations, raw events, and `ts_shift_us`
exactly like every other sequence, with no special-casing by sequence number
— any future zip packaged the same way is handled automatically.

---

## Summary

| Fix | Root cause | Symptom | Solution |
|---|---|---|---|
| Rollover | 24-bit counter resets every 16.777s | Multiple overlapping drone positions | Track `TIME_HIGH` decreases, add 2²⁴ per rollover |
| ts_shift | SDK applies 1.163s clock offset to Frames/ | Raw 1 second behind Frames/ | Skip first `ts_shift_us` µs; offset Frames/ lookup |
| Wrapped zip folder | seq 2's zip nests everything under an extra `2/` folder | `KeyError: "There is no item named 'coordinates.txt' in the archive"` | Auto-detect a shared top-level wrapper folder and strip it in `ZipSequence._to_member()` |

Fixes 1-2 are in `evt3_reader.py` and `raw_to_movie.py`; fix 3 is in
`zip_utils.py`. All downstream scripts (`build_dataset.py`,
`make_filter_movie.py`, `view_raw_events.py`) benefit automatically because
they all use `EVT3Reader` and `ZipSequence`.
