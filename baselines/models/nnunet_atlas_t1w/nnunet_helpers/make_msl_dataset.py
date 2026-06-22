#!/usr/bin/env python3
"""Build a Multi-Size-Labeling (MSL) dataset from a binary-lesion nnU-Net dataset.

Shang et al. 2024 (MSL/DBL) showed that splitting the single "lesion" class into
size-stratified sub-classes raises small-lesion recall, because the rare small
components get their own class signal instead of being drowned out by large
lesions under a single foreground label. We implement the MSL variant in a
framework-native way: relabel every connected lesion component (26-connectivity)
into a size class, write a new dataset, train nnU-Net multiclass, then merge the
size classes back to a binary lesion mask at inference.

Size classes (voxels, matching the eval framework's small-lesion bins):
    1 = small   (< 100 voxels)
    2 = medium  (100 - 999 voxels)
    3 = large   (>= 1000 voxels)

Images are symlinked from the source dataset (no copy); only labels are rewritten.

Usage:
    python make_msl_dataset.py --src-dir <raw/Dataset502_ATLASR30> \
        --dst-dir <raw/Dataset503_ATLASR30_MSL>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.int64)
# (upper-exclusive voxel bound, class id); last bound is +inf.
SIZE_CLASSES = [(100, 1), (1000, 2), (np.inf, 3)]
LABELS = {"background": 0, "small": 1, "medium": 2, "large": 3}


def size_relabel(mask: np.ndarray) -> np.ndarray:
    binary = mask > 0.5
    out = np.zeros(mask.shape, dtype=np.uint8)
    if not binary.any():
        return out
    lab, n = ndi.label(binary, structure=CONNECTIVITY_26)
    if n == 0:
        return out
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    for comp_id, sz in enumerate(sizes, start=1):
        cls = next(c for bound, c in SIZE_CLASSES if sz < bound)
        out[lab == comp_id] = cls
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, type=Path)
    ap.add_argument("--dst-dir", required=True, type=Path)
    args = ap.parse_args()

    src_images = args.src_dir / "imagesTr"
    src_labels = args.src_dir / "labelsTr"
    dst_images = args.dst_dir / "imagesTr"
    dst_labels = args.dst_dir / "labelsTr"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    # Symlink images (cheap; nnU-Net only reads them).
    for img in sorted(src_images.glob("*.nii.gz")):
        link = dst_images / img.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(img.resolve())

    label_files = sorted(src_labels.glob("*.nii.gz"))
    hist = {c: 0 for c in LABELS.values()}
    for lab_path in label_files:
        img = nib.load(str(lab_path))
        data = np.asanyarray(img.dataobj)
        relabeled = size_relabel(data)
        for c in LABELS.values():
            hist[c] += int((relabeled == c).sum())
        out = nib.Nifti1Image(relabeled, img.affine, img.header.copy())
        out.set_data_dtype(np.uint8)
        # nibabel infers the format from the suffix, so the temp must end .nii.gz.
        tmp = dst_labels / lab_path.name.replace(".nii.gz", ".tmp.nii.gz")
        nib.save(out, str(tmp))
        os.replace(tmp, dst_labels / lab_path.name)

    src_meta = json.loads((args.src_dir / "dataset.json").read_text())
    dataset = {
        "name": args.dst_dir.name,
        "description": "ATLAS R3.0 T1w MSL (size-stratified lesion classes) for small-lesion sensitivity",
        "channel_names": src_meta["channel_names"],
        "labels": LABELS,
        "numTraining": len(label_files),
        "file_ending": ".nii.gz",
    }
    (args.dst_dir / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")

    print(f"[done] wrote {len(label_files)} MSL labels to {dst_labels}")
    print(f"[info] voxel histogram by class: {hist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
