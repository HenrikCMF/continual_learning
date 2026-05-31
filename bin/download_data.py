"""
Download the large dataset CSVs for the ACORD testbed.

The datasets are too large for normal git, so they are hosted as GitHub
release assets on the `datasets-v1` release. This script fetches them into
the local `datasets/` folder.

Usage:
    python bin/download_data.py            # download any missing datasets
    python bin/download_data.py --force    # re-download even if present
"""
import argparse
import os
import sys
import urllib.request

# Public release-asset URLs (no auth required for a public repo).
RELEASE = "https://github.com/HenrikCMF/ACORD/releases/download/datasets-v1"
DATASETS = {
    "HAI.csv": f"{RELEASE}/HAI.csv",        # HAI ICS security dataset (used by configs.json)
    "sensor.csv": f"{RELEASE}/sensor.csv",  # Kaggle pump-sensor dataset (used by configs_pump.json)
}

DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")


def _report(name):
    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            sys.stdout.write(f"\r  {name}: {pct:5.1f}%  ({downloaded // (1024*1024)} MiB)")
        else:
            sys.stdout.write(f"\r  {name}: {downloaded // (1024*1024)} MiB")
        sys.stdout.flush()
    return hook


def main():
    parser = argparse.ArgumentParser(description="Download ACORD datasets.")
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()

    os.makedirs(DEST_DIR, exist_ok=True)
    for name, url in DATASETS.items():
        dest = os.path.join(DEST_DIR, name)
        if os.path.exists(dest) and not args.force:
            print(f"  {name}: already present, skipping (use --force to re-download)")
            continue
        print(f"Downloading {name} from {url}")
        urllib.request.urlretrieve(url, dest, _report(name))
        print(f"\n  saved to {dest}")
    print("Done.")


if __name__ == "__main__":
    main()
