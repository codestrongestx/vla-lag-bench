"""Assemble the assets root that the DynamicVLA moving-box and moving-can studies read.

The official DOM assets provide ``objects/`` and ``scenes/``. The studies add two
environments under ``derived/`` and the invisible ribbed carrier the moving can
rides on. Environments are copied byte for byte, because their checksums enter
each episode's identity; the official folders are linked, not copied.

    python scripts/dynamicvla/prepare_assets.py /path/to/DynamicVLA-assets/extracted assets/dynamicvla

Point the placement's ``installation.assets_root`` at the output directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from vlalagbench.files import sha256

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "configs/experiments/dynamicvla"
# Published file -> (source in this repository, SHA-256 of the evaluated copy).
DERIVED = {
    "derived/fcan04-close-box.json": (
        EXPERIMENTS / "fcan04_compressed_path_20260928/fcan04-close-box.json",
        "62c3a6fdc3b8a682cce17fc75ec0d99b471aa33f209db309e28d86980ca4b20d",
    ),
    "derived/fcan04-moving-can-ribbed.json": (
        EXPERIMENTS / "fcan04_moving_can_20261006/fcan04-moving-can-ribbed.json",
        "9f0ec2185006b5c58618ce9cc2a486435dfd48a14d75153eb4cef9c004b98453",
    ),
    "objects/carrier/ribbed01.usda": (
        EXPERIMENTS / "fcan04_moving_can_20261006/carrier/ribbed01.usda",
        "181a16fb81b79257257de65f9f3ff6ddc922e7d8052a00656b3c0d8dc9197cc7",
    ),
}


def prepare(official: Path, output: Path) -> None:
    official = official.resolve()
    for folder in ("objects", "scenes"):
        if not (official / folder).is_dir():
            raise FileNotFoundError(f"{official / folder}: not an extracted DOM assets directory")
    output.mkdir(parents=True, exist_ok=True)
    scenes = output / "scenes"
    if not scenes.is_symlink():
        scenes.symlink_to(official / "scenes")
    objects = output / "objects"
    objects.mkdir(exist_ok=True)
    for entry in sorted((official / "objects").iterdir()):
        link = objects / entry.name
        if not link.is_symlink():
            link.symlink_to(entry)
    for relative, (source, digest) in DERIVED.items():
        if sha256(source) != digest:
            raise ValueError(f"{source}: differs from the evaluated file")
        target = output / relative
        if target.parent.is_symlink():
            raise ValueError(f"{target.parent}: links into the official assets, which stay untouched")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    print(f"assets root: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("official", type=Path, help="extracted official DOM assets (holds objects/ and scenes/)")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.official, args.output)


if __name__ == "__main__":
    main()
