"""
evt3_reader.py — Parse Prophesee EVT3 raw binary file.

EVT3 format: stream of 2-byte (16-bit) words.
Each word type is determined by bits 15-12:

  0x0  ADDR_Y      → sets current Y coordinate
  0x2  ADDR_X      → fires event at (x, current_y, current_t, polarity)
                     bit 11 = polarity, bits 10-0 = x
  0x6  TIME_LOW    → sets low 12 bits of timestamp (microseconds)
  0x8  TIME_HIGH   → sets high bits of timestamp
  other            → ignored (vector events, triggers, etc.)

Usage:
    from evt3_reader import EVT3Reader
    reader = EVT3Reader("events.raw")
    events = reader.read_window(t_start_us=0, t_end_us=33333)
    # events is structured numpy array with fields: x, y, t, p
"""

import numpy as np
import os
from config import DEBUG_MODE, DEBUG_SAMPLES

_read_window_debug_count = 0

# Structured dtype for one event
EVENT_DTYPE = np.dtype([
    ('x', np.uint16),   # pixel column  0-1279
    ('y', np.uint16),   # pixel row     0-719
    ('t', np.int64),    # timestamp     microseconds
    ('p', np.uint8),    # polarity      0=negative, 1=positive
])


class EVT3Reader:
    """
    Reads a Prophesee EVT3 .raw file.

    Parameters
    ----------
    filepath : str
        Path to events.raw file.
    chunk_size : int
        Number of 16-bit words to process per chunk.
        Default 2M words = 4MB per chunk.
    """

    def __init__(self, filepath, chunk_size=2_000_000):
        self.filepath    = filepath
        self.chunk_size  = chunk_size
        self.header_end  = self._find_header_end()
        self.file_size   = os.path.getsize(filepath)
        self.ts_shift_us = self._read_ts_shift()
        print(f"EVT3Reader: {filepath}")
        print(f"  Header ends at byte {self.header_end}")
        print(f"  Binary data: {(self.file_size - self.header_end) / 1e6:.1f} MB")
        if self.ts_shift_us:
            print(f"  ts_shift_us : {self.ts_shift_us:,} µs ({self.ts_shift_us/1e6:.3f} s)")

    # ── ts_shift reader ───────────────────────────────────────────────────────

    def _read_ts_shift(self):
        """Read ts_shift_us from companion .tmp_index file.

        This offset aligns camera-internal timestamps with the synchronized
        clock used by Frames/ filenames, coordinates.txt and Event_YOLO/.
        """
        index_path = self.filepath + '.tmp_index'
        if not os.path.exists(index_path):
            return 0
        with open(index_path, 'rb') as f:
            for line in f:
                if not line.startswith(b'%'):
                    break
                line = line.decode('ascii', errors='ignore').strip()
                if 'ts_shift_us' in line:
                    try:
                        return int(line.split()[-1])
                    except ValueError:
                        pass
        return 0

    # ── Header detection ──────────────────────────────────────────────────────

    def _find_header_end(self):
        """Scan for end of ASCII header (lines starting with %)."""
        with open(self.filepath, 'rb') as f:
            pos = 0
            for line in f:
                if not line.startswith(b'%'):
                    break
                pos += len(line)
        return pos

    def read_header(self):
        """Return header as a dict of key→value pairs."""
        info = {}
        with open(self.filepath, 'rb') as f:
            for line in f:
                if not line.startswith(b'%'):
                    break
                line = line.decode('ascii', errors='ignore').strip().lstrip('%').strip()
                if ':' in line:
                    k, v = line.split(':', 1)
                    info[k.strip()] = v.strip()
        return info

    # ── Full file read ────────────────────────────────────────────────────────

    def read_all(self, max_events=None):
        """
        Read entire file and return all events as structured numpy array.

        Warning: 127MB file → ~10M+ events → uses significant RAM.
        Use read_window() for targeted extraction.
        """
        print("Reading all events (this may take 30-60s for 127MB)...")
        all_events = []

        for chunk_events in self._iter_chunks():
            all_events.append(chunk_events)
            if max_events and sum(len(e) for e in all_events) >= max_events:
                break

        if not all_events:
            return np.array([], dtype=EVENT_DTYPE)

        result = np.concatenate(all_events)
        print(f"  Total events read: {len(result):,}")
        return result

    # ── Window read (efficient) ───────────────────────────────────────────────

    def read_window(self, t_start_us, t_end_us):
        """
        Read only events in time window [t_start_us, t_end_us).

        Streams the file and stops early once past t_end_us.
        Much faster than read_all() for short windows.

        Parameters
        ----------
        t_start_us : int
            Window start in microseconds.
        t_end_us : int
            Window end in microseconds.

        Returns
        -------
        numpy structured array with fields (x, y, t, p)
        """
        result = []

        for chunk_events in self._iter_chunks():
            if len(chunk_events) == 0:
                continue

            # Filter to window
            mask = (chunk_events['t'] >= t_start_us) & (chunk_events['t'] < t_end_us)
            if mask.any():
                result.append(chunk_events[mask])

            # Stop early if we are past the window
            if chunk_events['t'][-1] > t_end_us:
                break

        if not result:
            return np.array([], dtype=EVENT_DTYPE)

        out = np.concatenate(result)

        global _read_window_debug_count
        if DEBUG_MODE and _read_window_debug_count < DEBUG_SAMPLES:
            print(f"  [DEBUG] read_window {t_start_us/1e6:.3f}s–{t_end_us/1e6:.3f}s: "
                  f"{len(out):,} events  "
                  f"pos={int((out['p']==1).sum()):,}  "
                  f"neg={int((out['p']==0).sum()):,}")
            _read_window_debug_count += 1

        return out

    # ── Iterator over time windows ────────────────────────────────────────────

    def iter_windows(self, window_us, t_start=None, t_end=None):
        """
        Yield (t_start, events) for each consecutive time window.

        Parameters
        ----------
        window_us : int
            Window duration in microseconds (e.g. 33333 for 30fps).
        t_start : int or None
            Start time. If None, uses first event timestamp.
        t_end : int or None
            End time. If None, reads to end of file.

        Yields
        ------
        (window_start_us, events_array)
        """
        # Buffer events across chunks
        buffer = []
        current_window_start = t_start

        for chunk_events in self._iter_chunks():
            if len(chunk_events) == 0:
                continue

            # Set start time from first event if not specified
            if current_window_start is None:
                current_window_start = int(chunk_events['t'][0])

            if t_end and chunk_events['t'][0] > t_end:
                break

            buffer.append(chunk_events)

            # Check if we have enough events to fill at least one window
            all_buffered = np.concatenate(buffer)
            t_max_buffered = int(all_buffered['t'][-1])

            while t_max_buffered >= current_window_start + window_us:
                t_window_end = current_window_start + window_us
                mask = (all_buffered['t'] >= current_window_start) & \
                       (all_buffered['t'] < t_window_end)
                window_events = all_buffered[mask]

                if t_end is None or current_window_start < t_end:
                    yield current_window_start, window_events

                current_window_start = t_window_end
                # Keep only remaining events in buffer
                remaining_mask = all_buffered['t'] >= current_window_start
                if remaining_mask.any():
                    buffer = [all_buffered[remaining_mask]]
                    all_buffered = buffer[0]
                else:
                    buffer = []
                    break

        # Flush remaining buffer
        if buffer and current_window_start is not None:
            all_buffered = np.concatenate(buffer)
            t_window_end = current_window_start + window_us
            if t_end is None or current_window_start < t_end:
                mask = all_buffered['t'] >= current_window_start
                if mask.any():
                    yield current_window_start, all_buffered[mask]

    # ── Internal chunk iterator ───────────────────────────────────────────────

    def _iter_chunks(self):
        """
        Internal generator: yield decoded events chunk by chunk.
        Maintains decoder state (current_y, timestamp) across chunks.
        """
        current_y        = 0
        time_high        = 0
        prev_time_high   = 0
        rollover_offset  = 0      # adds 2^24 µs on each TIME_HIGH rollover
        current_t        = 0
        ROLLOVER_ADD     = 1 << 24   # 16,777,216 µs = 16.777 s

        xs, ys, ts, ps = [], [], [], []

        with open(self.filepath, 'rb') as f:
            f.seek(self.header_end)

            while True:
                raw_bytes = f.read(self.chunk_size * 2)
                if not raw_bytes:
                    break

                # Pad to even number of bytes
                if len(raw_bytes) % 2 != 0:
                    raw_bytes = raw_bytes[:-1]

                words = np.frombuffer(raw_bytes, dtype=np.uint16)

                for word in words:
                    word_type = (word >> 12) & 0xF

                    if word_type == 0x0:            # ADDR_Y
                        current_y = word & 0x7FF

                    elif word_type == 0x2:          # ADDR_X → fire event
                        pol = (word >> 11) & 0x1
                        x   = word & 0x7FF
                        xs.append(x)
                        ys.append(current_y)
                        ts.append(current_t)
                        ps.append(pol)

                    elif word_type == 0x8:          # TIME_HIGH
                        new_time_high = int(word & 0xFFF) << 12
                        # Detect rollover: TIME_HIGH jumped back to a smaller value
                        if new_time_high < prev_time_high:
                            rollover_offset += ROLLOVER_ADD
                        prev_time_high = new_time_high
                        time_high  = new_time_high
                        current_t  = rollover_offset + time_high | (current_t & 0xFFF)

                    elif word_type == 0x6:          # TIME_LOW
                        current_t  = rollover_offset + time_high | int(word & 0xFFF)

                    # Other types (vector events, triggers) → skip

                # Yield events from this chunk
                if xs:
                    chunk = np.empty(len(xs), dtype=EVENT_DTYPE)
                    chunk['x'] = xs
                    chunk['y'] = ys
                    chunk['t'] = np.array(ts, dtype=np.int64)
                    chunk['p'] = ps
                    yield chunk
                    xs, ys, ts, ps = [], [], [], []

        # Yield any remaining events
        if xs:
            chunk = np.empty(len(xs), dtype=EVENT_DTYPE)
            chunk['x'] = xs
            chunk['y'] = ys
            chunk['t'] = np.array(ts, dtype=np.int64)
            chunk['p'] = ps
            yield chunk


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    fpath = sys.argv[1] if len(sys.argv) > 1 else "../data_from_fred/7/Event/events.raw"

    reader = EVT3Reader(fpath)

    print("\nHeader info:")
    for k, v in reader.read_header().items():
        print(f"  {k}: {v}")

    print("\nReading first window (0–33ms)...")
    events = reader.read_window(0, 33333)
    print(f"  Events in first window: {len(events):,}")
    if len(events) > 0:
        print(f"  Time range: {events['t'].min()}–{events['t'].max()} us")
        print(f"  X range: {events['x'].min()}–{events['x'].max()}")
        print(f"  Y range: {events['y'].min()}–{events['y'].max()}")
        print(f"  Positive events: {(events['p']==1).sum():,}")
        print(f"  Negative events: {(events['p']==0).sum():,}")

    print("\nCounting windows in first 5 seconds...")
    count = 0
    for t_start, evs in reader.iter_windows(33333, t_end=5_000_000):
        count += 1
        print(f"  Window t={t_start/1e6:.3f}s: {len(evs):,} events")
    print(f"  Total windows: {count}")
