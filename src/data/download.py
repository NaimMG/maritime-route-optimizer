"""
AIS Data Downloader - Marine Cadastre (NOAA)
Downloads real vessel tracking data in GeoParquet format from Azure.
Source: https://marinecadastre.gov/downloads/ais2024/
"""

import os
import requests
from tqdm import tqdm

RAW_DIR = "data/raw"
AZURE_BASE = "https://marinecadastre.gov/downloads/ais2024"


def download_file(url: str, dest_path: str) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=120)
        if response.status_code != 200:
            print(f"  ✗ Not found (HTTP {response.status_code}): {url}")
            return False

        total = int(response.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total,
            unit="B",
            unit_scale=True,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def download_ais_parquet(dates: list[str]) -> None:
    """
    Download AIS GeoParquet files for given dates.

    Args:
        dates: List of dates in format 'YYYY-MM-DD'
                e.g. ['2024-01-01', '2024-01-02']
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    success, failed = 0, 0

    print(f"\n📡 Downloading AIS GeoParquet data — {len(dates)} day(s)\n")

    for date in dates:
        filename = f"ais-{date}.parquet"
        url = f"{AZURE_BASE}/{filename}"
        dest = os.path.join(RAW_DIR, filename)

        if os.path.exists(dest):
            print(f"  ✓ Already exists: {filename}")
            success += 1
            continue

        print(f"  ↓ Downloading: {filename}")
        if download_file(url, dest):
            success += 1
            size_mb = os.path.getsize(dest) / 1e6
            print(f"  ✓ Saved ({size_mb:.1f} MB)")
        else:
            failed += 1

    print(f"\n✅ Done — {success} downloaded, {failed} failed")
    print(f"📁 Files saved to: {RAW_DIR}/")


if __name__ == "__main__":
    # Start with 3 days to validate pipeline
    download_ais_parquet(dates=[
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ])