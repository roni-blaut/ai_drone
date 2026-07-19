"""
zip_utils.py — Transparent zip/filesystem access for FRED sequence data.

When a sequence folder (e.g. data_from_fred/7/) does not exist on disk but the
matching zip file (data_from_fred/7.zip) does, all file access is routed through
the zip.  When the folder exists, normal filesystem calls are used unchanged.

Usage:
    from zip_utils import init_sequence, seq_glob, seq_imread, seq_open_lines

    init_sequence("../data_from_fred/7")   # auto-detects zip vs folder
    files = seq_glob(EVENT_YOLO_DIR, "*.txt")
    img   = seq_imread(png_path)
    lines = seq_open_lines(coords_path)
"""

import os
import io
import zipfile
import fnmatch
import glob as _glob

import numpy as np

_ACTIVE_SEQ = None   # ZipSequence instance, or None when using real filesystem


class ZipSequence:
    """
    Wraps a zipfile.ZipFile and translates filesystem paths into zip member names.

    The zip is expected to have NO top-level sequence-number prefix — entries
    start directly at Event/events.raw, coordinates.txt, etc.
    The caller provides seq_dir (e.g. ../data_from_fred/7) as the virtual root.
    """

    def __init__(self, zip_path, seq_dir):
        self._zip_path = os.path.normpath(zip_path)
        self._seq_dir  = os.path.normpath(seq_dir)
        self._zf       = zipfile.ZipFile(zip_path, 'r')
        self._names    = set(self._zf.namelist())
        self._root_prefix = self._detect_root_prefix()
        self.ts_shift_us = self._read_ts_shift()

    # ── Path conversion ───────────────────────────────────────────────────────

    def _detect_root_prefix(self):
        """
        Most sequence zips store files at the zip root (Event/, coordinates.txt, ...).
        A few are packaged with everything nested one level deeper under a single
        wrapper folder (e.g. seq 2's zip has '2/coordinates.txt', '2/Event/...').
        Detect that case so path lookups still resolve correctly.
        """
        if 'coordinates.txt' in self._names or 'Event/events.raw' in self._names:
            return ''
        tops = {n.split('/', 1)[0] for n in self._names}
        if len(tops) == 1:
            prefix = next(iter(tops)) + '/'
            if any(n.startswith(prefix) for n in self._names):
                return prefix
        return ''

    def _to_member(self, path):
        """Convert an absolute or relative filesystem path → zip member name."""
        rel = os.path.relpath(os.path.normpath(path), self._seq_dir)
        return self._root_prefix + rel.replace('\\', '/')

    # ── ts_shift from companion index file inside zip ─────────────────────────

    def _read_ts_shift(self):
        m = self._root_prefix + 'Event/events.raw.tmp_index'
        if m not in self._names:
            return 0
        content = self._zf.read(m).decode('ascii', errors='ignore')
        for line in content.splitlines():
            if not line.startswith('%'):
                break
            if 'ts_shift_us' in line:
                try:
                    return int(line.split()[-1])
                except ValueError:
                    pass
        return 0

    # ── File access ───────────────────────────────────────────────────────────

    def open_binary(self, path):
        """Return seekable io.BytesIO for a binary file (e.g. events.raw)."""
        return io.BytesIO(self._zf.read(self._to_member(path)))

    def open_lines(self, path):
        """Return list of text lines (keepends=True) for a text file."""
        raw = self._zf.read(self._to_member(path))
        return raw.decode('utf-8', errors='replace').splitlines(keepends=True)

    def read_bytes(self, path):
        """Return raw bytes for a file (for binary copy-out)."""
        return self._zf.read(self._to_member(path))

    def imread(self, path, flags):
        """Decode an image from the zip and return a numpy array."""
        import cv2
        buf = np.frombuffer(self._zf.read(self._to_member(path)), np.uint8)
        return cv2.imdecode(buf, flags)

    def glob(self, directory, pattern):
        """
        Return fake filesystem paths matching glob pattern inside the zip.

        Paths are returned as real-looking filesystem strings so that
        os.path.basename(), filename parsing, etc. all work unchanged.
        When these paths are passed back to seq_imread / seq_open_lines,
        _to_member() converts them back to zip member names.
        """
        prefix  = self._to_member(directory).rstrip('/') + '/'
        matches = fnmatch.filter(self._names, prefix + pattern)
        return sorted(
            os.path.join(self._seq_dir, m.replace('/', os.sep))
            for m in matches
        )

    def exists(self, path):
        return self._to_member(path) in self._names

    def close(self):
        self._zf.close()


# ── Module-level API ──────────────────────────────────────────────────────────

def init_sequence(seq_dir):
    """
    Set up zip or filesystem access for a sequence directory.

    - If seq_dir exists as a folder on disk → use normal filesystem (_ACTIVE_SEQ = None).
    - If seq_dir does not exist but seq_dir + '.zip' does → open ZipSequence.
    - Raises FileNotFoundError if neither exists.
    - Calling again with the same zip path is a no-op (zip stays open).
    """
    global _ACTIVE_SEQ
    seq_dir = os.path.normpath(seq_dir)

    if os.path.isdir(seq_dir):
        _ACTIVE_SEQ = None   # folder present — use real filesystem
        return

    zip_path = seq_dir + '.zip'
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(
            f"Sequence folder not found : {seq_dir}\n"
            f"Zip file not found        : {zip_path}\n"
            f"Place either the extracted folder or the .zip in data_from_fred/"
        )

    # Avoid re-opening the same zip
    if (_ACTIVE_SEQ is not None
            and os.path.normpath(_ACTIVE_SEQ._zip_path) == os.path.normpath(zip_path)):
        return

    if _ACTIVE_SEQ is not None:
        _ACTIVE_SEQ.close()

    _ACTIVE_SEQ = ZipSequence(zip_path, seq_dir)
    print(f"[zip_utils] Reading from {os.path.basename(zip_path)}  "
          f"(ts_shift={_ACTIVE_SEQ.ts_shift_us} µs)")


# ── Helper functions — filesystem or zip, transparent ────────────────────────

def seq_glob(directory, pattern='*'):
    """glob.glob replacement that works for zip or real filesystem."""
    if _ACTIVE_SEQ:
        return _ACTIVE_SEQ.glob(directory, pattern)
    return sorted(_glob.glob(os.path.join(directory, pattern)))


def seq_imread(path, flags=None):
    """cv2.imread replacement that works for zip or real filesystem."""
    import cv2
    if flags is None:
        flags = cv2.IMREAD_COLOR
    if _ACTIVE_SEQ and not os.path.isfile(path):
        return _ACTIVE_SEQ.imread(path, flags)
    return cv2.imread(path, flags)


def seq_open_lines(path):
    """open(path).readlines() replacement that works for zip or real filesystem."""
    if _ACTIVE_SEQ and not os.path.isfile(path):
        return _ACTIVE_SEQ.open_lines(path)
    with open(path) as f:
        return f.readlines()


def seq_open_binary(path):
    """open(path, 'rb') replacement that works for zip or real filesystem."""
    if _ACTIVE_SEQ and not os.path.isfile(path):
        return _ACTIVE_SEQ.open_binary(path)
    return open(path, 'rb')


def seq_exists(path):
    """os.path.exists replacement that checks zip members too (files and virtual dirs)."""
    if _ACTIVE_SEQ:
        if os.path.isfile(path) or os.path.isdir(path):
            return True
        member = _ACTIVE_SEQ._to_member(path)
        if member in _ACTIVE_SEQ._names:
            return True
        # Virtual directory: any member has this as a path prefix
        prefix = member.rstrip('/') + '/'
        return any(n.startswith(prefix) for n in _ACTIVE_SEQ._names)
    return os.path.exists(path)


# ── Multi-sequence zip cache (for Pipeline 1 / 2 training) ───────────────────

_ZIP_CACHE: dict = {}   # normalized seq_dir → ZipSequence


def _get_seq_zip(seq_dir):
    """Return a cached ZipSequence for seq_dir, or None if it's a real folder."""
    key = os.path.normpath(seq_dir)
    if key in _ZIP_CACHE:
        return _ZIP_CACHE[key]
    if os.path.isdir(key):
        return None
    zip_path = key + '.zip'
    if os.path.isfile(zip_path):
        zs = ZipSequence(zip_path, key)
        _ZIP_CACHE[key] = zs
        return zs
    return None


def multi_seq_imread(path, flags=None):
    """
    cv2.imread replacement that works across multiple sequence zips simultaneously.

    Intended for YOLO training (Pipeline 1/2) where images from several sequences
    are loaded in random order.  The path must contain data_from_fred/{seq_num}/
    so the correct zip can be auto-detected.

    Falls back to cv2.imread for real on-disk files.
    """
    import cv2
    if flags is None:
        flags = cv2.IMREAD_COLOR
    if os.path.isfile(path):
        return cv2.imread(path, flags)

    norm  = os.path.normpath(path)
    parts = norm.split(os.sep)
    for i, part in enumerate(parts):
        if part == 'data_from_fred' and i + 1 < len(parts):
            seq_dir = os.sep.join(parts[:i + 2])
            zs = _get_seq_zip(seq_dir)
            if zs:
                try:
                    return zs.imread(path, flags)
                except Exception:
                    pass
            break

    return cv2.imread(path, flags)
