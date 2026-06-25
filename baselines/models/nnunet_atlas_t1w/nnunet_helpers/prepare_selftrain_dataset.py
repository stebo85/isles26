#!/usr/bin/env python3
"""Merge ATLAS labels with pseudo-labels into a self-training nnU-Net dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def case_id_from_image(path: Path) -> str:
    name = path.name
    if name.endswith("_0000.nii.gz"):
        return name[:-12]
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def symlink_force(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def validate_pair(image_path: Path, label_path: Path) -> None:
    image = nib.as_closest_canonical(nib.load(str(image_path)))
    label = nib.as_closest_canonical(nib.load(str(label_path)))
    if tuple(image.shape[:3]) != tuple(label.shape[:3]):
        raise ValueError(f"shape mismatch: {image_path} {image.shape} vs {label_path} {label.shape}")
    if not np.allclose(image.affine, label.affine, atol=1e-3):
        raise ValueError(f"affine mismatch: {image_path} vs {label_path}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", required=True, type=Path)
    parser.add_argument("--pseudo-images-dir", required=True, type=Path)
    parser.add_argument("--pseudo-labels-dir", required=True, type=Path)
    parser.add_argument("--dst-dir", required=True, type=Path)
    parser.add_argument("--src-splits", type=Path)
    parser.add_argument("--dst-splits", type=Path)
    parser.add_argument("--pseudo-prefix", default="PSEUDO")
    args = parser.parse_args()

    src_images = args.src_dir / "imagesTr"
    src_labels = args.src_dir / "labelsTr"
    dst_images = args.dst_dir / "imagesTr"
    dst_labels = args.dst_dir / "labelsTr"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    atlas_cases = []
    for image_path in sorted(src_images.glob("*_0000.nii.gz")):
        case_id = case_id_from_image(image_path)
        label_path = src_labels / f"{case_id}.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(f"missing source label for {case_id}: {label_path}")
        symlink_force(image_path, dst_images / image_path.name)
        symlink_force(label_path, dst_labels / label_path.name)
        atlas_cases.append(case_id)
        rows.append({"case_id": case_id, "source": "atlas", "image": str(image_path), "label": str(label_path)})

    pseudo_cases = []
    pseudo_images = sorted(args.pseudo_images_dir.glob("*_0000.nii.gz"))
    if not pseudo_images:
        pseudo_images = sorted(args.pseudo_images_dir.glob("*.nii.gz"))
    for i, image_path in enumerate(pseudo_images, start=1):
        input_case = case_id_from_image(image_path)
        label_path = args.pseudo_labels_dir / f"{input_case}.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(f"missing pseudo-label for {input_case}: {label_path}")
        validate_pair(image_path, label_path)
        pseudo_id = f"{args.pseudo_prefix}_{i:04d}"
        symlink_force(image_path, dst_images / f"{pseudo_id}_0000.nii.gz")
        symlink_force(label_path, dst_labels / f"{pseudo_id}.nii.gz")
        pseudo_cases.append(pseudo_id)
        rows.append({"case_id": pseudo_id, "source": "pseudo", "image": str(image_path), "label": str(label_path)})

    src_meta = json.loads((args.src_dir / "dataset.json").read_text())
    dataset = {
        "name": args.dst_dir.name,
        "description": "ATLAS R3.0 plus TopK10 pseudo-label self-training dataset",
        "channel_names": src_meta["channel_names"],
        "labels": src_meta["labels"],
        "numTraining": len(atlas_cases) + len(pseudo_cases),
        "file_ending": ".nii.gz",
    }
    write_json(args.dst_dir / "dataset.json", dataset)

    with (args.dst_dir / "selftrain_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "source", "image", "label"])
        writer.writeheader()
        writer.writerows(rows)

    if args.src_splits and args.dst_splits:
        splits = json.loads(args.src_splits.read_text())
        out_splits = []
        for split in splits:
            train = sorted(set(split["train"]) | set(pseudo_cases))
            val = sorted(split["val"])
            out_splits.append({"train": train, "val": val})
        write_json(args.dst_splits, out_splits)

    print(f"[done] atlas cases={len(atlas_cases)} pseudo cases={len(pseudo_cases)} -> {args.dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
