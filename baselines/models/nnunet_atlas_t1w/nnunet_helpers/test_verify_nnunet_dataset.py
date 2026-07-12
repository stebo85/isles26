#!/usr/bin/env python3
"""Direct lightweight tests for verify_nnunet_dataset.py."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from baselines.models.nnunet_atlas_t1w.nnunet_helpers.make_center_grouped_splits import (
    assignment_input_sha256,
    build_splits,
    records_from_rows,
    validate_splits,
)
from baselines.models.nnunet_atlas_t1w.nnunet_helpers.verify_nnunet_dataset import (
    VerificationError,
    main as verifier_main,
    verify_dataset,
)


DATASET_NAME = "Dataset507_ATLASR30_CORRECTED"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture_rows(raw_dir: Path) -> list[dict[str, object]]:
    rows = []
    for center_index in range(5):
        for case_index in range(2):
            case_id = f"CASE_{center_index}_{case_index}"
            rows.append(
                {
                    "nnunet_id": case_id,
                    "conversion_version": "test-v1",
                    "image": str(raw_dir / "imagesTr" / f"{case_id}_0000.nii.gz"),
                    "label": str(raw_dir / "labelsTr" / f"{case_id}.nii.gz"),
                    "t1_source_path": f"/source/{case_id}_T1w.nii.gz",
                    "lesion_source_path": f"/source/{case_id}_lesion.nii.gz",
                    "t1_source_sha256_prefix": hashlib.sha256(
                        f"t1-{case_id}".encode()
                    ).hexdigest()[:16],
                    "lesion_source_sha256_prefix": hashlib.sha256(
                        f"seg-{case_id}".encode()
                    ).hexdigest()[:16],
                    "site": f"CENTER_{center_index}",
                    "chronicity_bin": ["lt180d", "ge180d", "unknown"][
                        (center_index + case_index) % 3
                    ],
                    "size_bin": ["small", "medium", "large"][
                        (2 * center_index + case_index) % 3
                    ],
                    "lesion_volume_ml": f"{0.25 + center_index + case_index:.3f}",
                }
            )
    return rows


def _create_fixture(root: Path) -> dict[str, object]:
    raw_dir = root / "nnUNet_raw" / DATASET_NAME
    preprocessed_dir = root / "nnUNet_preprocessed" / DATASET_NAME
    images = raw_dir / "imagesTr"
    labels = raw_dir / "labelsTr"
    gt = preprocessed_dir / "gt_segmentations"
    configuration = preprocessed_dir / "nnUNetPlans_3d_fullres"
    for directory in (images, labels, gt, configuration):
        directory.mkdir(parents=True, exist_ok=True)

    rows = _fixture_rows(raw_dir)
    manifest = raw_dir / "conversion_manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    records = records_from_rows(rows)
    expected_sha = assignment_input_sha256(records)
    cases = [record.case_id for record in records]
    for index, case_id in enumerate(cases, start=1):
        (images / f"{case_id}_0000.nii.gz").write_bytes(b"image" * index)
        (labels / f"{case_id}.nii.gz").write_bytes(b"label" * index)
        (gt / f"{case_id}.nii.gz").write_bytes(b"ground-truth" * index)
        (configuration / f"{case_id}.pkl").write_bytes(b"properties" * index)
        (configuration / f"{case_id}.b2nd").write_bytes(b"data" * index)
        (configuration / f"{case_id}_seg.b2nd").write_bytes(b"seg" * index)

    dataset_json = {
        "name": DATASET_NAME,
        "channel_names": {"0": "T1w"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
    }
    _write_json(raw_dir / "dataset.json", dataset_json)
    _write_json(preprocessed_dir / "dataset.json", dataset_json)
    _write_json(
        preprocessed_dir / "dataset_fingerprint.json",
        {
            "spacings": [[1.0, 1.0, 1.0] for _ in cases],
            "shapes_after_crop": [[32, 32, 32] for _ in cases],
            "median_relative_size_after_cropping": 0.5,
        },
    )
    _write_json(
        preprocessed_dir / "nnUNetPlans.json",
        {
            "dataset_name": DATASET_NAME,
            "plans_name": "nnUNetPlans",
            "configurations": {"3d_fullres": {"batch_size": 2}},
        },
    )

    center_to_fold = {f"CENTER_{fold}": fold for fold in range(5)}
    splits = build_splits(records, center_to_fold, folds=5)
    validation = validate_splits(splits, records)
    splits_path = preprocessed_dir / "splits_final.json"
    summary_path = preprocessed_dir / "splits_summary.json"
    _write_json(splits_path, splits)
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "manifest": str(manifest.resolve()),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "assignment_input_sha256": expected_sha,
            "folds": 5,
            "global": {"n_cases": len(cases), "n_centers": 5},
            "validation": validation,
            "center_assignments": [
                {"center": f"CENTER_{fold}", "fold": fold, "n_cases": 2}
                for fold in range(5)
            ],
        },
    )
    return {
        "raw_dir": raw_dir,
        "preprocessed_dir": preprocessed_dir,
        "manifest": manifest,
        "splits_path": splits_path,
        "summary_path": summary_path,
        "expected_sha": expected_sha,
        "expected_cases": len(cases),
        "cases": cases,
    }


def _verify(fixture: dict[str, object]) -> dict[str, object]:
    return verify_dataset(
        raw_dir=fixture["raw_dir"],  # type: ignore[arg-type]
        preprocessed_dir=fixture["preprocessed_dir"],  # type: ignore[arg-type]
        manifest=fixture["manifest"],  # type: ignore[arg-type]
        splits_path=fixture["splits_path"],  # type: ignore[arg-type]
        split_summary_path=fixture["summary_path"],  # type: ignore[arg-type]
        expected_dataset_name=DATASET_NAME,
        expected_cases=fixture["expected_cases"],  # type: ignore[arg-type]
        expected_assignment_sha=fixture["expected_sha"],  # type: ignore[arg-type]
    )


def _expect_verification_error(function: object, message_fragment: str) -> None:
    try:
        function()  # type: ignore[operator]
    except VerificationError as exc:
        assert message_fragment in str(exc), str(exc)
    else:
        raise AssertionError("invalid dataset was accepted")


def test_valid_fixture_and_cli_output() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        fixture = _create_fixture(root / "fixture")
        lineage = _verify(fixture)
        assert lineage["dataset"]["num_training"] == 10  # type: ignore[index]
        assert lineage["splits"]["folds"] == 5  # type: ignore[index]
        assert all(
            inventory["count"] == 10
            for inventory in lineage["payload_inventories"].values()  # type: ignore[union-attr]
        )

        output = root / "reports" / "lineage.json"
        result = verifier_main(
            [
                "--raw-dir",
                str(fixture["raw_dir"]),
                "--preprocessed-dir",
                str(fixture["preprocessed_dir"]),
                "--manifest",
                str(fixture["manifest"]),
                "--splits",
                str(fixture["splits_path"]),
                "--split-summary",
                str(fixture["summary_path"]),
                "--expected-dataset-name",
                DATASET_NAME,
                "--expected-cases",
                str(fixture["expected_cases"]),
                "--expected-assignment-sha",
                str(fixture["expected_sha"]),
                "--out-lineage",
                str(output),
            ]
        )
        assert result == 0
        assert json.loads(output.read_text()) == lineage


def test_lineage_is_path_independent() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        staged = _create_fixture(root / "staging" / "job-123")
        promoted = _create_fixture(root / "final")
        staged_lineage = _verify(staged)
        promoted_lineage = _verify(promoted)
        assert staged_lineage == promoted_lineage
        serialized = json.dumps(staged_lineage, sort_keys=True)
        assert str(root) not in serialized


def test_rejects_stale_preprocessing_and_fingerprint() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        fixture = _create_fixture(Path(temporary_directory))
        first_case = fixture["cases"][0]  # type: ignore[index]
        data_file = (
            fixture["preprocessed_dir"]  # type: ignore[operator]
            / "nnUNetPlans_3d_fullres"
            / f"{first_case}.b2nd"
        )
        data_file.unlink()
        _expect_verification_error(
            lambda: _verify(fixture), "preprocessed data case set differs"
        )
        data_file.write_bytes(b"restored")

        fingerprint_path = (
            fixture["preprocessed_dir"] / "dataset_fingerprint.json"  # type: ignore[operator]
        )
        fingerprint = json.loads(fingerprint_path.read_text())
        fingerprint["spacings"].pop()
        _write_json(fingerprint_path, fingerprint)
        _expect_verification_error(lambda: _verify(fixture), "spacings length")


def test_rejects_split_center_and_stale_summary() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        fixture = _create_fixture(Path(temporary_directory))
        splits_path = fixture["splits_path"]  # type: ignore[assignment]
        original = json.loads(splits_path.read_text())
        corrupted = copy.deepcopy(original)
        moved_case = corrupted[0]["val"].pop()
        corrupted[0]["train"].append(moved_case)
        _write_json(splits_path, corrupted)
        _expect_verification_error(lambda: _verify(fixture), "splits center")

        _write_json(splits_path, original)
        summary_path = fixture["summary_path"]  # type: ignore[assignment]
        summary = json.loads(summary_path.read_text())
        summary["center_assignments"][0]["fold"] = 4
        _write_json(summary_path, summary)
        _expect_verification_error(
            lambda: _verify(fixture), "center assignments differ"
        )


def test_rejects_assignment_hash_and_dataset_count_mismatches() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        fixture = _create_fixture(Path(temporary_directory))
        bad_sha = "0" * 64
        original_sha = fixture["expected_sha"]
        fixture["expected_sha"] = bad_sha
        _expect_verification_error(
            lambda: _verify(fixture), "assignment_input_sha256 mismatch"
        )
        fixture["expected_sha"] = original_sha

        dataset_path = fixture["preprocessed_dir"] / "dataset.json"  # type: ignore[operator]
        dataset = json.loads(dataset_path.read_text())
        dataset["numTraining"] = 9
        _write_json(dataset_path, dataset)
        _expect_verification_error(lambda: _verify(fixture), "numTraining")


def main() -> int:
    tests = [
        test_valid_fixture_and_cli_output,
        test_lineage_is_path_independent,
        test_rejects_stale_preprocessing_and_fingerprint,
        test_rejects_split_center_and_stale_summary,
        test_rejects_assignment_hash_and_dataset_count_mismatches,
    ]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
