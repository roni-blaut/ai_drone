"""
gdrive.py — Google Drive folder scan and lazy zip download for FRED sequences.

Usage:
    # Scan public Drive folder for file IDs:
    from gdrive import scan_folder
    id_map = scan_folder("1pISIErXOx76xmCqkwhS3-azWOMlTKZMp")
    # Returns: {"4": "1AbCd...", "7": "1EfGh...", ...}

    # Download one zip (only when not already local):
    from gdrive import download_zip
    download_zip(seq_num=4, drive_file_id="1AbCd...", data_dir="../data_from_fred")
"""

import os
import re
import requests


def scan_folder(folder_id, timeout=30):
    """
    List .zip files in a public Google Drive folder without an API key.

    Returns dict {seq_id: file_id} where seq_id is the filename without .zip.
    Returns empty dict if the folder page cannot be parsed (Drive HTML may change).
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36")
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        print(f"[gdrive] ERROR fetching folder page: {e}")
        return {}

    # Google Drive embeds file metadata as JSON arrays in the page HTML.
    # We look for file ID strings (28+ char base64url) adjacent to .zip filenames.
    matches = re.findall(r'"([\w-]{28,})"[^"]*?"([^"]+\.zip)"', resp.text)
    result = {}
    for fid, name in matches:
        seq_id = name.replace('.zip', '')
        result[seq_id] = fid

    if not result:
        print("[gdrive] WARNING: No .zip files found in folder page.")
        print("         Google may have changed their HTML format.")
        print("         Add drive_file_id fields to catalog.yaml manually.")

    return result


def download_zip(seq_num, drive_file_id, data_dir, quiet=False):
    """
    Download {seq_num}.zip from Google Drive to data_dir/ using gdown.

    Only call this when the zip is not already present locally — callers should
    check os.path.isfile(zip_path) first.  Returns the local path.
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for Drive download.\n"
            "Install with: pip install gdown"
        )

    out_path = os.path.join(data_dir, f"{seq_num}.zip")
    if not quiet:
        print(f"  [gdrive] Downloading seq {seq_num}.zip → {out_path}")

    url = f"https://drive.google.com/uc?id={drive_file_id}"
    gdown.download(url, out_path, quiet=quiet)

    if not os.path.isfile(out_path):
        raise RuntimeError(
            f"Download failed: {out_path} not found after gdown.\n"
            f"Check that the Drive file is publicly accessible."
        )

    size_mb = os.path.getsize(out_path) // (1024 * 1024)
    print(f"  [gdrive] Downloaded {seq_num}.zip ({size_mb} MB)")
    return out_path
