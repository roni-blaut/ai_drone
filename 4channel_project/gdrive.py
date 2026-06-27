"""
gdrive.py — Google Drive folder scan and lazy zip download for FRED sequences.

Scanning requires a free Google API key (no billing needed):
  1. Go to console.cloud.google.com
  2. Create project → Enable "Google Drive API"
  3. Credentials → Create credentials → API key
  4. Pass it with --api-key: python make_catalog.py --scan-drive --api-key AIza...

Usage:
    # Scan public Drive folder for file IDs (recommended: with API key):
    from gdrive import scan_folder
    id_map = scan_folder("1pISIErXOx76xmCqkwhS3-azWOMlTKZMp", api_key="AIza...")
    # Returns: {"4": "1AbCd...", "7": "1EfGh...", ...}

    # Download one zip (only when not already local):
    from gdrive import download_zip
    download_zip(seq_num=4, drive_file_id="1AbCd...", data_dir="../data_from_fred")
"""

import os
import re
import requests


def scan_folder(folder_id, api_key=None, timeout=30):
    """
    List .zip files in a public Google Drive folder.

    api_key : free Google API key — enables the reliable Drive API v3 method.
              Without it, falls back to HTML page parsing which often fails.
              Get one free at console.cloud.google.com (Drive API, no billing).

    Returns dict {seq_id: file_id} where seq_id is the filename without .zip.
    """
    if api_key:
        return _scan_folder_api(folder_id, api_key, timeout)
    return _scan_folder_html(folder_id, timeout)


def _scan_folder_api(folder_id, api_key, timeout=30):
    """
    List folder contents via Google Drive API v3.
    Reliable, handles pagination, works for any public shared folder.
    """
    result = {}
    page_token = None
    base_url = "https://www.googleapis.com/drive/v3/files"

    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken, files(id, name)",
            "pageSize": 1000,
            "key": api_key,
            "includeItemsFromAllDrives": "true",
            "supportsAllDrives": "true",
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            resp = requests.get(base_url, params=params, timeout=timeout)
            resp.raise_for_status()
        except Exception as e:
            print(f"[gdrive] ERROR calling Drive API: {e}")
            return {}

        data = resp.json()

        if "error" in data:
            err = data["error"]
            print(f"[gdrive] Drive API error {err.get('code')}: {err.get('message')}")
            print("         Check that the API key is valid and the Drive API is enabled.")
            return {}

        for f in data.get("files", []):
            name = f.get("name", "")
            if name.endswith(".zip"):
                result[name.replace(".zip", "")] = f["id"]

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    print(f"[gdrive] Found {len(result)} .zip file(s) in Drive folder")
    return result


def _scan_folder_html(folder_id, timeout=30):
    """
    Fallback: parse the public Drive folder HTML page for file IDs.
    Fragile — may stop working if Google changes their page format.
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

    matches = re.findall(r'"([\w-]{28,})"[^"]*?"([^"]+\.zip)"', resp.text)
    result = {}
    for fid, name in matches:
        result[name.replace(".zip", "")] = fid

    if not result:
        print("[gdrive] WARNING: HTML parsing found no .zip files.")
        print("         Google Drive pages require JavaScript — HTML parsing is unreliable.")
        print()
        print("         RECOMMENDED FIX — use a free Google API key:")
        print("           1. Go to https://console.cloud.google.com")
        print("           2. Create project → Enable 'Google Drive API'")
        print("           3. Credentials → Create credentials → API key")
        print("           4. Run: python make_catalog.py --scan-drive --api-key AIza...")
        print()
        print("         ALTERNATIVE — add drive_file_id fields manually to catalog.yaml:")
        print("           Open the Drive folder, click a file, copy the ID from the URL")
        print("           (the long string between /d/ and /view), then add to catalog.yaml:")
        print("             sequences:")
        print('               "7":')
        print('                 drive_file_id: "1AbCdEfGhIjKlMnOpQrStUvWxYz012345"')

    return result


def download_folder_all(folder_id, data_dir, quiet=False):
    """
    Download ALL .zip files from a public Google Drive folder using gdown.

    No API key needed — gdown handles the folder listing internally.
    Skips files that already exist locally.
    Returns list of downloaded zip paths.
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required.\n"
            "Install with: pip install gdown"
        )

    os.makedirs(data_dir, exist_ok=True)

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"[gdrive] Downloading all zips from Drive folder → {data_dir}/")
    print("[gdrive] This may take a while for large folders (~100 sequences)")

    downloaded = gdown.download_folder(
        url=url,
        output=data_dir,
        quiet=quiet,
        use_cookies=False,
        remaining_ok=True,
    )

    zips = [p for p in (downloaded or []) if p.endswith(".zip")]
    print(f"[gdrive] Done — {len(zips)} zip(s) in {data_dir}/")
    return zips


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
