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
import ssl
import requests
import urllib3

# Corporate networks often run SSL inspection proxies that replace server
# certificates with a self-signed enterprise certificate.  Python rejects
# these because the CA is not in the standard trust store.
# Patching here disables SSL verification for all HTTPS calls in this module
# (gdown also uses requests internally, so the patch covers it too).
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def scan_folder(folder_id, api_key=None, timeout=30):
    """
    List .zip files in a public Google Drive folder.

    Tries methods in order:
      1. gdown skip_download=True  — reliable, no API key needed (default)
      2. Google Drive API v3       — if api_key is provided
      3. HTML page parsing         — last resort fallback (often fails)

    Returns dict {seq_id: file_id} where seq_id is the filename without .zip.
    """
    result = _scan_folder_gdown(folder_id)
    if result:
        return result

    if api_key:
        result = _scan_folder_api(folder_id, api_key, timeout)
        if result:
            return result

    return _scan_folder_html(folder_id, timeout)


def _scan_folder_gdown(folder_id):
    """
    List folder contents via gdown skip_download=True — no API key needed.
    Returns {seq_id: file_id} for all .zip files, or {} on failure.
    """
    try:
        import gdown
    except ImportError:
        return {}

    try:
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        files = gdown.download_folder(
            url=url,
            skip_download=True,
            use_cookies=False,
            quiet=True,
        )
    except Exception as e:
        print(f"[gdrive] gdown folder scan failed: {e}")
        return {}

    if not files:
        return {}

    result = {}
    for f in files:
        name = getattr(f, 'path', '') or ''
        name = os.path.basename(name)
        fid  = getattr(f, 'id', '')
        if name.endswith('.zip') and fid:
            result[name.replace('.zip', '')] = fid

    if result:
        print(f"[gdrive] Found {len(result)} .zip file(s) via gdown")
    return result


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
            resp = requests.get(base_url, params=params, timeout=timeout, verify=False)
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
        resp = requests.get(url, headers=headers, timeout=timeout, verify=False)
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


def download_folder_all(folder_id, data_dir, quiet=False, max_workers=5):
    """
    Download ALL .zip files from a public Google Drive folder.

    Downloads up to max_workers files in parallel (default: 5).
    Press Ctrl+C to stop cleanly — in-progress downloads finish, queued ones cancel.
    Already-present zips are always skipped.
    Returns list of downloaded zip paths.
    """
    try:
        import gdown
    except ImportError:
        raise ImportError("gdown is required.\nInstall with: pip install gdown")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    os.makedirs(data_dir, exist_ok=True)

    # Scan folder to get file IDs (no download yet)
    print("[gdrive] Scanning Drive folder...")
    id_map = _scan_folder_gdown(folder_id)
    if not id_map:
        id_map = _scan_folder_html(folder_id)
    if not id_map:
        print("[gdrive] ERROR: could not list folder contents.")
        print("         Make sure the folder is publicly shared.")
        return []

    # Filter out zips already on disk
    seq_order = sorted(id_map.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    to_download = {}
    n_skipped = 0
    for seq_id in seq_order:
        if os.path.isfile(os.path.join(data_dir, f"{seq_id}.zip")):
            n_skipped += 1
        else:
            to_download[seq_id] = id_map[seq_id]

    if n_skipped:
        print(f"[gdrive] {n_skipped} already present — skipping")
    if not to_download:
        print("[gdrive] Nothing to download.")
        return []

    print(f"[gdrive] Downloading {len(to_download)} file(s) → {data_dir}/")
    print(f"[gdrive] Parallel workers : {max_workers}  |  Press Ctrl+C to stop cleanly")
    print()

    downloaded = []
    stop_event = threading.Event()

    def _download_one(seq_id, file_id):
        if stop_event.is_set():
            return None
        out_path = os.path.join(data_dir, f"{seq_id}.zip")
        url      = f"https://drive.google.com/uc?id={file_id}"
        try:
            result = gdown.download(url, out_path, quiet=quiet, resume=True, fuzzy=True)
            if result and os.path.isfile(out_path):
                size_mb = os.path.getsize(out_path) // (1024 * 1024)
                print(f"  [done] {seq_id}.zip  ({size_mb} MB)")
                return out_path
            else:
                print(f"  [skip] {seq_id}.zip  (rate-limited — re-run later)")
                return None
        except Exception as e:
            if not stop_event.is_set():
                print(f"  [fail] {seq_id}.zip  — {e}")
            return None

    interrupted = False
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {
        executor.submit(_download_one, seq_id, fid): seq_id
        for seq_id, fid in to_download.items()
    }
    try:
        for future in as_completed(futures):
            result = future.result()
            if result:
                downloaded.append(result)
    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
        print("\n[gdrive] Ctrl+C — cancelling queued downloads...")
        for f in futures:
            f.cancel()
    finally:
        executor.shutdown(wait=False)

    total = len(to_download)
    done  = len(downloaded)
    print()
    print(f"[gdrive] Done — {done}/{total} downloaded")
    if interrupted:
        print(f"[gdrive] {total - done} remaining. Re-run to continue — existing zips are skipped.")
    elif done < total:
        print(f"[gdrive] {total - done} rate-limited. Re-run later — existing zips are skipped.")

    return downloaded


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
