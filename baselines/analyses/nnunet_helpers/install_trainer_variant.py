#!/usr/bin/env python3
"""Install the local 250-epoch trainer variant into the active nnU-Net package."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nnunetv2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("nnUNetTrainer_250epochs.py"),
        help="Local trainer variant source file.",
    )
    args = parser.parse_args()

    package_root = Path(nnunetv2.__file__).resolve().parent
    target_dir = package_root / "training" / "nnUNetTrainer" / "variants" / "training_length"
    target_dir.mkdir(parents=True, exist_ok=True)

    for init_dir in [
        package_root / "training",
        package_root / "training" / "nnUNetTrainer",
        package_root / "training" / "nnUNetTrainer" / "variants",
        target_dir,
    ]:
        init_file = init_dir / "__init__.py"
        if not init_file.exists():
            init_file.touch()

    target = target_dir / args.source.name
    source_bytes = args.source.read_bytes()
    if target.exists() and target.read_bytes() == source_bytes:
        print(f"[ok] trainer variant already installed: {target}")
        return 0

    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(source_bytes)
    os.replace(tmp, target)
    print(f"[done] installed trainer variant: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
