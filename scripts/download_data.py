"""Download AESO hourly pool price and Alberta Internal Load (AIL) data.

Run this on your own computer:
    python scripts/download_data.py

Files are saved to data/raw/. They are large (the asset level generation
columns make them wide), so the download can take a few minutes.
"""
from pathlib import Path
import urllib.request

BASE = "https://www.aeso.ca/assets/Uploads/data-requests/"
FILES = [
    "Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2010-2019.csv",
    "Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv",
    # Older history, uncomment if you want it:
    # "Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2001-2009.csv",
]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = RAW_DIR / name
        if target.exists():
            print(f"Already have {name}, skipping")
            continue
        print(f"Downloading {name} ...")
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "alberta-power-agent"})
        with urllib.request.urlopen(req) as resp, open(target, "wb") as out:
            out.write(resp.read())
        print(f"  saved {target.stat().st_size / 1e6:.1f} MB")
    print("Done. Next: python scripts/build_dataset.py")


if __name__ == "__main__":
    main()
