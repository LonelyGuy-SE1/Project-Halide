"""
Download FilmDamageSimulator and BlueNeg datasets.
FilmDamageSimulator: GitHub + Figshare
BlueNeg: HuggingFace
"""
import json
import os
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_film_damage_simulator():
    """Download FilmDamageSimulator scans and annotations from GitHub."""
    print("Downloading FilmDamageSimulator from GitHub...")

    local_dir = DATA_DIR / "FilmDamageSimulator"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Clone the repo (shallow clone to save space)
    repo_url = "https://github.com/daniela997/FilmDamageSimulator.git"
    clone_dir = local_dir / "FilmDamageSimulator"

    if clone_dir.exists():
        print(f"  Already cloned at {clone_dir}")
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
            check=True,
        )
        print(f"  Cloned to {clone_dir}")

    # List what we got
    scans_dir = clone_dir / "scans"
    if scans_dir.exists():
        scans = list(scans_dir.glob("*.jpg"))
        jsons = list(scans_dir.glob("*.json"))
        print(f"  Found {len(scans)} scans, {len(jsons)} annotation files")
    else:
        print(f"  Warning: scans/ directory not found in {clone_dir}")

    return local_dir


def download_blueneg():
    """Download BlueNeg dataset from HuggingFace."""
    print("Downloading BlueNeg from HuggingFace...")

    repo_id = "ur2good/blueneg-release"
    local_dir = DATA_DIR / "BlueNeg"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download meta.json first (small, contains all metadata)
    meta_path = hf_hub_download(
        repo_id=repo_id,
        filename="meta.json",
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    print(f"  Downloaded meta.json to {meta_path}")

    # Download preview images (8-bit PNGs, small files)
    print("  Downloading preview images...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=["negative-preview-8bit/*", "pseudogt-8bit/*"],
    )

    print(f"  Downloaded previews to {local_dir}")
    return local_dir


def main():
    print("=== Project Halide Dataset Download ===\n")

    fds_dir = download_film_damage_simulator()
    bn_dir = download_blueneg()

    print("\n=== Download Complete ===")
    print(f"FilmDamageSimulator: {fds_dir}")
    print(f"BlueNeg: {bn_dir}")

    # Print summary
    fds_scans_dir = fds_dir / "FilmDamageSimulator" / "scans"
    if fds_scans_dir.exists():
        fds_scans = list(fds_scans_dir.glob("*.jpg"))
        fds_jsons = list(fds_scans_dir.glob("*.json"))
        print(f"\nFilmDamageSimulator: {len(fds_scans)} scans, {len(fds_jsons)} annotation files")

    if (bn_dir / "meta.json").exists():
        with open(bn_dir / "meta.json") as f:
            meta = json.load(f)
        print(f"BlueNeg: {len(meta)} images")


if __name__ == "__main__":
    main()
