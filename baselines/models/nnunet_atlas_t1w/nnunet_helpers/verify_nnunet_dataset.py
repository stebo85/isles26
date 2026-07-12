#!/usr/bin/env python3
"""Verify an immutable center-grouped nnU-Net dataset and emit its lineage.

The verifier intentionally avoids hashing the multi-gigabyte image payloads.
Instead, it proves exact case membership for every raw and preprocessed
collection and records deterministic filename/size inventory certificates.
Small control artifacts are content-hashed. Path-valued manifest fields and
the path-specific fields in the split summary are excluded from their semantic
hashes so a staged dataset and its atomically promoted copy have identical
lineage documents.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

try:
    from .make_center_grouped_splits import (
        ManifestError,
        SplitValidationError,
        assignment_input_sha256,
        read_manifest,
        validate_splits,
    )
except ImportError:  # Direct execution from the helpers directory.
    from make_center_grouped_splits import (  # type: ignore[no-redef]
        ManifestError,
        SplitValidationError,
        assignment_input_sha256,
        read_manifest,
        validate_splits,
    )


CONFIGURATION = "3d_fullres"
PLANS_DIRECTORY = "nnUNetPlans_3d_fullres"
EXPECTED_FOLDS = 5
LINEAGE_VERSION = "nnunet_center_grouped_lineage_v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# These fields vary when the converter writes the same dataset to a staging
# directory and then to its final directory. Source checksum prefixes and all
# non-path provenance remain covered by the semantic manifest hash.
MANIFEST_PATH_FIELDS = {
    "image",
    "label",
    "t1_source_path",
    "lesion_source_path",
    "t1_path",
    "lesion_path",
}
SUMMARY_PATH_FIELDS = {"manifest", "manifest_sha256"}


class VerificationError(ValueError):
    """Raised when a dataset violates an immutable-lineage invariant."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise VerificationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, description: str) -> object:
    if not path.is_file():
        raise VerificationError(f"missing {description}: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {description} {path}: {exc}") from exc


def _require_object(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise VerificationError(f"{description} must be a JSON object")
    return value


def _require_name_and_count(
    payload: Mapping[str, object],
    description: str,
    expected_dataset_name: str,
    expected_cases: int,
) -> None:
    if payload.get("name") != expected_dataset_name:
        raise VerificationError(
            f"{description} name is {payload.get('name')!r}; "
            f"expected {expected_dataset_name!r}"
        )
    count = payload.get("numTraining")
    if isinstance(count, bool) or not isinstance(count, int) or count != expected_cases:
        raise VerificationError(
            f"{description} numTraining is {count!r}; expected {expected_cases}"
        )


def _require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise VerificationError(f"missing {description}: {path}")


def _collect_nifti_cases(
    directory: Path, filename_suffix: str, description: str
) -> dict[str, Path]:
    _require_directory(directory, description)
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or not path.name.endswith(".nii.gz"):
            continue
        if not path.name.endswith(filename_suffix):
            raise VerificationError(
                f"unexpected {description} filename {path.name!r}; "
                f"expected suffix {filename_suffix!r}"
            )
        case_id = path.name[: -len(filename_suffix)]
        if not case_id:
            raise VerificationError(f"empty case ID in {description}: {path.name!r}")
        if case_id in result:
            raise VerificationError(f"duplicate case {case_id!r} in {description}")
        result[case_id] = path
    return result


def _collect_preprocessed_cases(
    directory: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    _require_directory(directory, f"preprocessed {CONFIGURATION} directory")
    properties: dict[str, Path] = {}
    data: dict[str, Path] = {}
    segmentations: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        if path.name.endswith(".pkl"):
            case_id = path.name[: -len(".pkl")]
            destination = properties
        elif path.name.endswith("_seg.b2nd"):
            case_id = path.name[: -len("_seg.b2nd")]
            destination = segmentations
        elif path.name.endswith(".b2nd"):
            case_id = path.name[: -len(".b2nd")]
            destination = data
        else:
            continue
        if not case_id:
            raise VerificationError(f"empty case ID in preprocessed file {path.name!r}")
        if case_id in destination:
            raise VerificationError(
                f"duplicate preprocessed artifact for case {case_id!r}: {path.name!r}"
            )
        destination[case_id] = path
    return properties, data, segmentations


def _assert_exact_cases(
    actual: Mapping[str, Path] | set[str], expected: set[str], description: str
) -> None:
    actual_cases = set(actual)
    missing = sorted(expected - actual_cases)
    unexpected = sorted(actual_cases - expected)
    if missing or unexpected:
        raise VerificationError(
            f"{description} case set differs from manifest: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def _case_ids_sha256(case_ids: set[str]) -> str:
    return _canonical_sha256(sorted(case_ids))


def _inventory_certificate(files: Mapping[str, Path]) -> dict[str, object]:
    entries = []
    total_bytes = 0
    for case_id in sorted(files):
        try:
            size = files[case_id].stat().st_size
        except OSError as exc:
            raise VerificationError(f"cannot stat {files[case_id]}: {exc}") from exc
        total_bytes += size
        entries.append([case_id, files[case_id].name, size])
    return {
        "count": len(entries),
        "total_bytes": total_bytes,
        "case_ids_sha256": _case_ids_sha256(set(files)),
        "names_and_sizes_sha256": _canonical_sha256(entries),
    }


def _semantic_manifest_sha256(path: Path) -> str:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise VerificationError(f"manifest has no CSV header: {path}")
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise VerificationError(f"manifest has duplicate CSV columns: {path}")
            retained_fields = sorted(
                field
                for field in reader.fieldnames
                if field not in MANIFEST_PATH_FIELDS and not field.lower().endswith("_path")
            )
            if "nnunet_id" not in retained_fields:
                raise VerificationError("manifest is missing nnunet_id")
            rows = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise VerificationError(
                        f"manifest row {row_number} has more values than its header"
                    )
                if any(row.get(field) is None for field in reader.fieldnames):
                    raise VerificationError(
                        f"manifest row {row_number} has fewer values than its header"
                    )
                rows.append({field: row[field] for field in retained_fields})
    except (OSError, UnicodeError, csv.Error) as exc:
        raise VerificationError(f"cannot read manifest {path}: {exc}") from exc
    rows.sort(key=lambda row: (row["nnunet_id"], _canonical_sha256(row)))
    return _canonical_sha256({"fields": retained_fields, "rows": rows})


def _semantic_summary_sha256(summary: Mapping[str, object]) -> str:
    semantic = {
        key: value for key, value in summary.items() if key not in SUMMARY_PATH_FIELDS
    }
    return _canonical_sha256(semantic)


def _load_and_validate_splits(
    path: Path, records: Sequence[object]
) -> tuple[list[Mapping[str, Sequence[str]]], Mapping[str, object]]:
    payload = _load_json(path, "split file")
    if not isinstance(payload, list):
        raise VerificationError("split file must contain a JSON list")
    if len(payload) != EXPECTED_FOLDS:
        raise VerificationError(
            f"split file has {len(payload)} folds; expected {EXPECTED_FOLDS}"
        )
    splits: list[Mapping[str, Sequence[str]]] = []
    for fold, split in enumerate(payload):
        if not isinstance(split, dict):
            raise VerificationError(f"split fold {fold} must be a JSON object")
        for subset in ("train", "val"):
            values = split.get(subset)
            if not isinstance(values, list) or not all(
                isinstance(case_id, str) for case_id in values
            ):
                raise VerificationError(
                    f"split fold {fold} {subset!r} must be a list of strings"
                )
        splits.append(split)
    try:
        certificate = validate_splits(splits, records)  # type: ignore[arg-type]
    except SplitValidationError as exc:
        raise VerificationError(f"invalid center-grouped splits: {exc}") from exc
    return splits, certificate


def _validate_split_summary(
    summary: Mapping[str, object],
    assignment_sha: str,
    expected_cases: int,
    expected_centers: int,
    splits: Sequence[Mapping[str, Sequence[str]]],
    case_to_center: Mapping[str, str],
) -> None:
    if summary.get("assignment_input_sha256") != assignment_sha:
        raise VerificationError(
            "split summary assignment_input_sha256 does not match the manifest: "
            f"{summary.get('assignment_input_sha256')!r} != {assignment_sha!r}"
        )
    if summary.get("folds") != len(splits):
        raise VerificationError(
            f"split summary folds is {summary.get('folds')!r}; expected {len(splits)}"
        )
    global_summary = _require_object(summary.get("global"), "split summary global")
    if global_summary.get("n_cases") != expected_cases:
        raise VerificationError(
            f"split summary global.n_cases is {global_summary.get('n_cases')!r}; "
            f"expected {expected_cases}"
        )
    if global_summary.get("n_centers") != expected_centers:
        raise VerificationError(
            f"split summary global.n_centers is {global_summary.get('n_centers')!r}; "
            f"expected {expected_centers}"
        )
    validation = _require_object(
        summary.get("validation"), "split summary validation"
    )
    if validation.get("passed") is not True:
        raise VerificationError("split summary does not report passed validation")
    for key, expected in (
        ("n_cases", expected_cases),
        ("n_centers", expected_centers),
        ("n_folds", len(splits)),
    ):
        if validation.get(key) != expected:
            raise VerificationError(
                f"split summary validation.{key} is {validation.get(key)!r}; "
                f"expected {expected}"
            )

    expected_center_folds: dict[str, int] = {}
    for fold, split in enumerate(splits):
        for case_id in split["val"]:
            expected_center_folds[case_to_center[case_id]] = fold
    assignments = summary.get("center_assignments")
    if not isinstance(assignments, list):
        raise VerificationError("split summary center_assignments must be a list")
    actual_center_folds: dict[str, int] = {}
    for index, item in enumerate(assignments):
        if not isinstance(item, dict):
            raise VerificationError(
                f"split summary center_assignments[{index}] must be an object"
            )
        center = item.get("center")
        fold = item.get("fold")
        if not isinstance(center, str) or isinstance(fold, bool) or not isinstance(fold, int):
            raise VerificationError(
                f"invalid center/fold in split summary center_assignments[{index}]"
            )
        if center in actual_center_folds:
            raise VerificationError(
                f"duplicate center {center!r} in split summary center_assignments"
            )
        actual_center_folds[center] = fold
    if actual_center_folds != expected_center_folds:
        missing = sorted(set(expected_center_folds) - set(actual_center_folds))
        unexpected = sorted(set(actual_center_folds) - set(expected_center_folds))
        mismatched = sorted(
            center
            for center in set(expected_center_folds) & set(actual_center_folds)
            if expected_center_folds[center] != actual_center_folds[center]
        )
        raise VerificationError(
            "split summary center assignments differ from splits: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"wrong_fold={mismatched[:5]}"
        )


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def verify_dataset(
    *,
    raw_dir: Path,
    preprocessed_dir: Path,
    manifest: Path,
    splits_path: Path,
    split_summary_path: Path,
    expected_dataset_name: str,
    expected_cases: int,
    expected_assignment_sha: str,
) -> dict[str, object]:
    """Validate a dataset and return a deterministic, path-free lineage."""

    if expected_cases <= 0:
        raise VerificationError("expected case count must be positive")
    if not SHA256_PATTERN.fullmatch(expected_assignment_sha):
        raise VerificationError(
            "expected assignment SHA must be 64 lowercase hexadecimal characters"
        )
    _require_directory(raw_dir, "raw dataset directory")
    _require_directory(preprocessed_dir, "preprocessed dataset directory")
    if not manifest.is_file():
        raise VerificationError(f"missing conversion manifest: {manifest}")

    try:
        records = read_manifest(manifest)
    except ManifestError as exc:
        raise VerificationError(f"invalid conversion manifest: {exc}") from exc
    manifest_cases = {record.case_id for record in records}
    if len(manifest_cases) != expected_cases:
        raise VerificationError(
            f"manifest contains {len(manifest_cases)} cases; expected {expected_cases}"
        )
    assignment_sha = assignment_input_sha256(records)
    if assignment_sha != expected_assignment_sha:
        raise VerificationError(
            "manifest assignment_input_sha256 mismatch: "
            f"{assignment_sha} != {expected_assignment_sha}"
        )

    raw_dataset_path = raw_dir / "dataset.json"
    preprocessed_dataset_path = preprocessed_dir / "dataset.json"
    fingerprint_path = preprocessed_dir / "dataset_fingerprint.json"
    plans_path = preprocessed_dir / "nnUNetPlans.json"
    raw_dataset = _require_object(
        _load_json(raw_dataset_path, "raw dataset.json"), "raw dataset.json"
    )
    preprocessed_dataset = _require_object(
        _load_json(preprocessed_dataset_path, "preprocessed dataset.json"),
        "preprocessed dataset.json",
    )
    _require_name_and_count(
        raw_dataset, "raw dataset.json", expected_dataset_name, expected_cases
    )
    _require_name_and_count(
        preprocessed_dataset,
        "preprocessed dataset.json",
        expected_dataset_name,
        expected_cases,
    )

    fingerprint = _require_object(
        _load_json(fingerprint_path, "dataset fingerprint"), "dataset fingerprint"
    )
    for field in ("spacings", "shapes_after_crop"):
        values = fingerprint.get(field)
        if not isinstance(values, list) or len(values) != expected_cases:
            length = len(values) if isinstance(values, list) else None
            raise VerificationError(
                f"dataset fingerprint {field} length is {length!r}; "
                f"expected {expected_cases}"
            )

    plans = _require_object(_load_json(plans_path, "nnUNet plans"), "nnUNet plans")
    if plans.get("dataset_name") != expected_dataset_name:
        raise VerificationError(
            f"nnUNet plans dataset_name is {plans.get('dataset_name')!r}; "
            f"expected {expected_dataset_name!r}"
        )
    configurations = plans.get("configurations")
    if not isinstance(configurations, dict) or CONFIGURATION not in configurations:
        raise VerificationError(
            f"nnUNet plans do not define required configuration {CONFIGURATION!r}"
        )

    raw_images = _collect_nifti_cases(
        raw_dir / "imagesTr", "_0000.nii.gz", "raw training images"
    )
    raw_labels = _collect_nifti_cases(
        raw_dir / "labelsTr", ".nii.gz", "raw training labels"
    )
    ground_truth = _collect_nifti_cases(
        preprocessed_dir / "gt_segmentations",
        ".nii.gz",
        "preprocessed ground-truth segmentations",
    )
    properties, data, preprocessed_segmentations = _collect_preprocessed_cases(
        preprocessed_dir / PLANS_DIRECTORY
    )
    collections = {
        "raw_images": raw_images,
        "raw_labels": raw_labels,
        "ground_truth": ground_truth,
        "preprocessed_properties": properties,
        "preprocessed_data": data,
        "preprocessed_segmentations": preprocessed_segmentations,
    }
    for description, files in collections.items():
        _assert_exact_cases(files, manifest_cases, description.replace("_", " "))

    splits, split_validation = _load_and_validate_splits(splits_path, records)
    split_summary = _require_object(
        _load_json(split_summary_path, "split summary"), "split summary"
    )
    case_to_center = {record.case_id: record.center for record in records}
    n_centers = len(set(case_to_center.values()))
    _validate_split_summary(
        split_summary,
        assignment_sha,
        expected_cases,
        n_centers,
        splits,
        case_to_center,
    )

    case_ids_sha = _case_ids_sha256(manifest_cases)
    return {
        "schema_version": 1,
        "lineage_version": LINEAGE_VERSION,
        "dataset": {
            "name": expected_dataset_name,
            "num_training": expected_cases,
            "configuration": CONFIGURATION,
            "case_ids_sha256": case_ids_sha,
        },
        "splits": {
            "folds": len(splits),
            "centers": n_centers,
            "assignment_input_sha256": assignment_sha,
            "case_ids_sha256": case_ids_sha,
            "validation": dict(split_validation),
        },
        "control_artifacts": {
            "manifest_semantic_sha256": _semantic_manifest_sha256(manifest),
            "raw_dataset_json_sha256": _sha256(raw_dataset_path),
            "preprocessed_dataset_json_sha256": _sha256(preprocessed_dataset_path),
            "dataset_fingerprint_sha256": _sha256(fingerprint_path),
            "nnunet_plans_sha256": _sha256(plans_path),
            "splits_sha256": _sha256(splits_path),
            "split_summary_semantic_sha256": _semantic_summary_sha256(split_summary),
        },
        "payload_inventories": {
            description: _inventory_certificate(files)
            for description, files in sorted(collections.items())
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--preprocessed-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--split-summary", required=True, type=Path)
    parser.add_argument("--expected-dataset-name", required=True)
    parser.add_argument("--expected-cases", required=True, type=int)
    parser.add_argument("--expected-assignment-sha", required=True)
    parser.add_argument("--out-lineage", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        lineage = verify_dataset(
            raw_dir=args.raw_dir,
            preprocessed_dir=args.preprocessed_dir,
            manifest=args.manifest,
            splits_path=args.splits,
            split_summary_path=args.split_summary,
            expected_dataset_name=args.expected_dataset_name,
            expected_cases=args.expected_cases,
            expected_assignment_sha=args.expected_assignment_sha,
        )
        _write_json_atomic(args.out_lineage, lineage)
    except (VerificationError, OSError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    print(
        f"[done] verified {args.expected_dataset_name}: {args.expected_cases} cases, "
        f"{lineage['splits']['centers']} centers, {lineage['splits']['folds']} folds"
    )
    print(f"[done] lineage: {args.out_lineage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
