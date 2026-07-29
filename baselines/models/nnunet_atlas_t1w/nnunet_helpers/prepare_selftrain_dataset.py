#!/usr/bin/env python3
"""Merge ATLAS labels with pseudo-labels into a self-training nnU-Net dataset.

Pseudo-labelled cases are appended to the TRAIN side of *every* fold. That is
only sound while the pseudo cases are genuinely unlabelled data: if one of them
is really a labelled case under a different file name, it lands in the training
set of the exact fold that is supposed to hold it out, and every cross-validation
number reported afterwards is silently optimistic with nothing to flag it. A
prose warning cannot enforce that, so this script refuses to build the dataset
unless disjointness is proven, by checks that do not share a failure mode:

  * ``--pseudo-images-dir`` / ``--pseudo-labels-dir`` must not resolve inside an
    nnUNet_raw tree. Pointing SELFTRAIN_INPUT_DIR at a converted dataset is the
    easiest way to feed the labelled set straight back in.
  * filesystem identity: the realpath of each pseudo image must not equal a
    labelled image, a labelled label, or a ``t1_source_path`` recorded in the
    source dataset's ``conversion_manifest.csv``. Resolving to realpath also
    catches symlink aliases and ``..`` spellings that a string compare misses.
  * content identity: the SHA-256 of each pseudo image file must not equal that
    of any labelled image, and its 16-hex prefix must not equal a
    ``t1_source_sha256_prefix`` in the conversion manifest. Both are needed
    because conversion re-encodes the volume through nibabel, so raw ATLAS T1w
    bytes match the manifest prefix while a copied converted file matches the
    full hash.

The evidence (per-case source path and hash, for pseudo *and* labelled cases) is
written to ``selftrain_provenance.json`` next to the dataset, so a later reader
can re-confirm disjointness with a set intersection instead of re-hashing
several hundred volumes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

PROVENANCE_VERSION = "v1"

# convert_atlas_r21_to_nnunet.sha256_prefix() stores this many hex characters of
# the source-file SHA-256 in conversion_manifest.csv; compare like with like.
SOURCE_SHA_PREFIX_LEN = 16

CHECKS_PERFORMED = [
    "pseudo_dirs_outside_nnunet_raw",
    "realpath_vs_labelled_images_and_labels",
    "realpath_vs_conversion_manifest_sources",
    "sha256_vs_labelled_images",
    "sha256_prefix_vs_conversion_manifest_sources",
    "pseudo_case_ids_disjoint_from_labelled",
    "pseudo_case_ids_absent_from_source_splits",
]


class ContaminationError(RuntimeError):
    """A pseudo-labelled input could not be proven disjoint from the labelled set."""


@dataclass(frozen=True)
class CaseEntry:
    """One case on its way into the merged dataset."""

    case_id: str
    image_path: Path
    label_path: Path
    image_realpath: str
    image_sha256: str
    source: str
    input_case_id: str = ""


@dataclass
class LabelledIndex:
    """Every identity by which a labelled case can be recognised.

    Each dict maps the identity to ``(case_id, path_it_came_from)`` so a failure
    message can name the collision instead of just asserting one exists.
    """

    realpaths: dict[str, tuple[str, str]] = field(default_factory=dict)
    image_hashes: dict[str, tuple[str, str]] = field(default_factory=dict)
    source_hash_prefixes: dict[str, tuple[str, str]] = field(default_factory=dict)


def case_id_from_image(path: Path) -> str:
    name = path.name
    if name.endswith("_0000.nii.gz"):
        return name[:-12]
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def nnunet_raw_roots(src_dir: Path) -> list[Path]:
    """Raw-tree roots the pseudo inputs must stay out of.

    The env var is what the step-27 driver exports; the ancestor scan also works
    when this script is run by hand without the nnU-Net environment set.
    """
    roots: list[Path] = []
    env_raw = os.environ.get("nnUNet_raw", "").strip()
    if env_raw:
        roots.append(Path(env_raw).resolve())
    resolved_src = src_dir.resolve()
    for candidate in (resolved_src, *resolved_src.parents):
        if candidate.name == "nnUNet_raw":
            roots.append(candidate)
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def assert_outside_raw_tree(label: str, directory: Path, roots: list[Path]) -> None:
    resolved = directory.resolve()
    for root in roots:
        if resolved == root or resolved.is_relative_to(root):
            raise ContaminationError(
                f"{label} resolves inside the nnUNet_raw tree "
                f"(check: pseudo_dirs_outside_nnunet_raw): {directory} -> {resolved} "
                f"is under {root}. Pseudo-labelled inputs must be unlabelled data held "
                f"outside nnUNet_raw; feeding a converted dataset back in would put "
                f"held-out cases into the training side of their own fold."
            )


def build_labelled_index(src_dir: Path, labelled: list[CaseEntry]) -> LabelledIndex:
    index = LabelledIndex()
    for entry in labelled:
        index.realpaths.setdefault(entry.image_realpath, (entry.case_id, str(entry.image_path)))
        index.realpaths.setdefault(
            os.path.realpath(entry.label_path), (entry.case_id, str(entry.label_path))
        )
        index.image_hashes.setdefault(entry.image_sha256, (entry.case_id, str(entry.image_path)))

    # The converted images are re-encoded copies, so the pre-conversion sources
    # are a second, independent way to smuggle a labelled case back in.
    manifest_path = src_dir / "conversion_manifest.csv"
    if manifest_path.is_file():
        with manifest_path.open(newline="") as f:
            for row in csv.DictReader(f):
                case_id = row.get("nnunet_id", "") or "<unknown>"
                for field_name in ("t1_source_path", "lesion_source_path"):
                    raw_path = (row.get(field_name) or "").strip()
                    if raw_path:
                        index.realpaths.setdefault(
                            os.path.realpath(raw_path), (case_id, raw_path)
                        )
                prefix = (row.get("t1_source_sha256_prefix") or "").strip().lower()
                if prefix:
                    index.source_hash_prefixes.setdefault(
                        prefix, (case_id, row.get("t1_source_path", ""))
                    )
    return index


def assert_pseudo_disjoint(
    image_path: Path,
    image_realpath: str,
    image_sha256: str,
    index: LabelledIndex,
) -> None:
    """Fail loudly, naming both sides, if this pseudo image is a labelled case."""
    hit = index.realpaths.get(image_realpath)
    if hit is not None:
        case_id, labelled_path = hit
        raise ContaminationError(
            f"pseudo image is the labelled case {case_id} "
            f"(check: realpath): pseudo={image_path} -> {image_realpath}; "
            f"labelled={labelled_path}"
        )
    hit = index.image_hashes.get(image_sha256)
    if hit is not None:
        case_id, labelled_path = hit
        raise ContaminationError(
            f"pseudo image has byte-identical content to labelled case {case_id} "
            f"(check: sha256={image_sha256}): pseudo={image_path}; labelled={labelled_path}"
        )
    hit = index.source_hash_prefixes.get(image_sha256[:SOURCE_SHA_PREFIX_LEN])
    if hit is not None:
        case_id, source_path = hit
        raise ContaminationError(
            f"pseudo image has the content of the pre-conversion source of labelled "
            f"case {case_id} (check: sha256_prefix={image_sha256[:SOURCE_SHA_PREFIX_LEN]}): "
            f"pseudo={image_path}; labelled source={source_path}"
        )


def collect_labelled(src_images: Path, src_labels: Path) -> list[CaseEntry]:
    entries: list[CaseEntry] = []
    for image_path in sorted(src_images.glob("*_0000.nii.gz")):
        case_id = case_id_from_image(image_path)
        label_path = src_labels / f"{case_id}.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(f"missing source label for {case_id}: {label_path}")
        entries.append(
            CaseEntry(
                case_id=case_id,
                image_path=image_path,
                label_path=label_path,
                image_realpath=os.path.realpath(image_path),
                image_sha256=sha256_file(image_path),
                source="atlas",
                input_case_id=case_id,
            )
        )
    return entries


def collect_pseudo(
    images_dir: Path,
    labels_dir: Path,
    prefix: str,
    index: LabelledIndex,
) -> list[CaseEntry]:
    pseudo_images = sorted(images_dir.glob("*_0000.nii.gz"))
    if not pseudo_images:
        pseudo_images = sorted(images_dir.glob("*.nii.gz"))

    entries: list[CaseEntry] = []
    for i, image_path in enumerate(pseudo_images, start=1):
        input_case = case_id_from_image(image_path)
        label_path = labels_dir / f"{input_case}.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(f"missing pseudo-label for {input_case}: {label_path}")
        validate_pair(image_path, label_path)
        image_realpath = os.path.realpath(image_path)
        image_sha256 = sha256_file(image_path)
        assert_pseudo_disjoint(image_path, image_realpath, image_sha256, index)
        entries.append(
            CaseEntry(
                case_id=f"{prefix}_{i:04d}",
                image_path=image_path,
                label_path=label_path,
                image_realpath=image_realpath,
                image_sha256=image_sha256,
                source="pseudo",
                input_case_id=input_case,
            )
        )
    return entries


def assert_manifest_disjoint(
    labelled: list[CaseEntry], pseudo: list[CaseEntry], index: LabelledIndex
) -> None:
    """Re-check disjointness on the finished case lists.

    The per-case guards run while the lists are being built; this re-derives the
    same property from the data that is about to be written, so a future edit
    that skips or reorders a guard still cannot produce a contaminated manifest.
    Replaying the whole identity check is what makes that true for *every*
    vector: the labelled-label, manifest-source-path and manifest-source-hash
    aliases are only visible through the index, so a plain pseudo-image-versus-
    labelled-image compare would let them through on this second pass.
    """
    labelled_ids = {entry.case_id for entry in labelled}
    shared_ids = sorted(labelled_ids & {entry.case_id for entry in pseudo})
    if shared_ids:
        raise ContaminationError(
            "pseudo case ids collide with labelled case ids "
            f"(check: case_id): {', '.join(shared_ids[:5])}"
        )

    labelled_realpaths = {entry.image_realpath: entry.case_id for entry in labelled}
    labelled_hashes = {entry.image_sha256: entry.case_id for entry in labelled}
    for entry in pseudo:
        assert_pseudo_disjoint(entry.image_path, entry.image_realpath, entry.image_sha256, index)
        # The two dicts above are re-derived from the entries that are about to
        # be written, so they still hold if the index is ever built from a stale
        # or partial source.
        case_id = labelled_realpaths.get(entry.image_realpath)
        if case_id is not None:
            raise ContaminationError(
                f"manifest is contaminated: {entry.case_id} shares a realpath with "
                f"labelled case {case_id} (check: realpath): {entry.image_realpath}"
            )
        case_id = labelled_hashes.get(entry.image_sha256)
        if case_id is not None:
            raise ContaminationError(
                f"manifest is contaminated: {entry.case_id} shares content with "
                f"labelled case {case_id} (check: sha256): {entry.image_sha256}"
            )


def build_splits(
    src_splits: list[dict[str, list[str]]],
    labelled_ids: set[str],
    pseudo_ids: list[str],
) -> list[dict[str, list[str]]]:
    pseudo_set = set(pseudo_ids)
    out_splits: list[dict[str, list[str]]] = []
    for fold, split in enumerate(src_splits):
        unknown = sorted((set(split["train"]) | set(split["val"])) - labelled_ids)
        if unknown:
            raise ValueError(
                f"fold {fold} of the source splits references cases that are not in the "
                f"labelled dataset: {', '.join(unknown[:5])}"
            )
        leaked = sorted((set(split["train"]) | set(split["val"])) & pseudo_set)
        if leaked:
            raise ContaminationError(
                f"fold {fold} of the source splits already contains pseudo case ids "
                f"(check: pseudo_case_ids_absent_from_source_splits): {', '.join(leaked[:5])}"
            )
        out_splits.append(
            {"train": sorted(set(split["train"]) | pseudo_set), "val": sorted(split["val"])}
        )
    return out_splits


def provenance_payload(
    args: argparse.Namespace,
    raw_roots: list[Path],
    labelled: list[CaseEntry],
    pseudo: list[CaseEntry],
) -> dict[str, object]:
    return {
        "provenance_version": PROVENANCE_VERSION,
        "dataset": args.dst_dir.name,
        "src_dir": str(args.src_dir.resolve()),
        "pseudo_images_dir": str(args.pseudo_images_dir.resolve()),
        "pseudo_labels_dir": str(args.pseudo_labels_dir.resolve()),
        "nnunet_raw_roots_checked": [str(root) for root in raw_roots],
        "checks_performed": CHECKS_PERFORMED,
        "n_labelled": len(labelled),
        "n_pseudo": len(pseudo),
        "labelled": [
            {
                "case_id": entry.case_id,
                "image": str(entry.image_path),
                "image_realpath": entry.image_realpath,
                "image_sha256": entry.image_sha256,
            }
            for entry in labelled
        ],
        "pseudo": [
            {
                "case_id": entry.case_id,
                "input_case_id": entry.input_case_id,
                "image": str(entry.image_path),
                "image_realpath": entry.image_realpath,
                "image_sha256": entry.image_sha256,
                "label": str(entry.label_path),
                "label_realpath": os.path.realpath(entry.label_path),
            }
            for entry in pseudo
        ],
    }


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

    raw_roots = nnunet_raw_roots(args.src_dir)
    assert_outside_raw_tree(
        "--pseudo-images-dir (SELFTRAIN_INPUT_DIR)", args.pseudo_images_dir, raw_roots
    )
    assert_outside_raw_tree("--pseudo-labels-dir", args.pseudo_labels_dir, raw_roots)

    # Everything is validated before a single symlink is written, so a rejected
    # run leaves no half-built dataset that a later step could mistake for good.
    labelled = collect_labelled(src_images, src_labels)
    index = build_labelled_index(args.src_dir, labelled)
    pseudo = collect_pseudo(
        args.pseudo_images_dir, args.pseudo_labels_dir, args.pseudo_prefix, index
    )
    assert_manifest_disjoint(labelled, pseudo, index)

    out_splits: list[dict[str, list[str]]] | None = None
    if args.src_splits and args.dst_splits:
        out_splits = build_splits(
            json.loads(args.src_splits.read_text()),
            {entry.case_id for entry in labelled},
            [entry.case_id for entry in pseudo],
        )

    dst_images = args.dst_dir / "imagesTr"
    dst_labels = args.dst_dir / "labelsTr"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    for entry in labelled + pseudo:
        symlink_force(entry.image_path, dst_images / f"{entry.case_id}_0000.nii.gz")
        symlink_force(entry.label_path, dst_labels / f"{entry.case_id}.nii.gz")

    src_meta = json.loads((args.src_dir / "dataset.json").read_text())
    dataset = {
        "name": args.dst_dir.name,
        "description": "ATLAS R3.0 plus TopK10 pseudo-label self-training dataset",
        "channel_names": src_meta["channel_names"],
        "labels": src_meta["labels"],
        "numTraining": len(labelled) + len(pseudo),
        "file_ending": ".nii.gz",
    }
    write_json(args.dst_dir / "dataset.json", dataset)

    with (args.dst_dir / "selftrain_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_id", "source", "image", "label", "image_realpath", "image_sha256"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "case_id": entry.case_id,
                "source": entry.source,
                "image": str(entry.image_path),
                "label": str(entry.label_path),
                "image_realpath": entry.image_realpath,
                "image_sha256": entry.image_sha256,
            }
            for entry in labelled + pseudo
        )

    write_json(
        args.dst_dir / "selftrain_provenance.json",
        provenance_payload(args, raw_roots, labelled, pseudo),
    )

    if out_splits is not None:
        write_json(args.dst_splits, out_splits)

    print(f"[done] atlas cases={len(labelled)} pseudo cases={len(pseudo)} -> {args.dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
