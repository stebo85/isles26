#!/usr/bin/env python3
"""Direct lightweight tests for make_center_grouped_splits.py.

Run from the repository root:
  python3 baselines/models/nnunet_atlas_t1w/nnunet_helpers/test_make_center_grouped_splits.py
"""

from __future__ import annotations

import copy
import csv
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from baselines.models.nnunet_atlas_t1w.nnunet_helpers.make_center_grouped_splits import (
    ManifestError,
    SplitValidationError,
    allocate_centers,
    assignment_input_sha256,
    build_splits,
    main as split_main,
    records_from_rows,
    validate_splits,
)


def _synthetic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    size_names = ["vsmall", "small", "medium", "large", "vlarge"]
    # Twelve indivisible centers have unequal sizes and deliberately different
    # phase/size mixes. This is large enough to exercise moves and swaps.
    center_sizes = [13, 12, 11, 10, 9, 8, 7, 7, 6, 6, 5, 5]
    for center_index, n_cases in enumerate(center_sizes):
        for case_index in range(n_cases):
            rows.append(
                {
                    "nnunet_id": f"CASE_{center_index:02d}_{case_index:03d}",
                    "site": f"CENTER_{center_index:02d}",
                    "chronicity_bin": ["lt180d", "ge180d", "unknown"][
                        (case_index + center_index) % 3
                    ],
                    "size_bin": size_names[(2 * case_index + center_index) % len(size_names)],
                    "lesion_volume_ml": 0.05 * (1 + ((case_index * 7 + center_index) % 100)),
                }
            )
    return rows


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "nnunet_id",
        "site",
        "chronicity_bin",
        "size_bin",
        "lesion_volume_ml",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_exact_coverage_and_center_exclusivity() -> None:
    records = records_from_rows(_synthetic_rows())
    allocation = allocate_centers(records, folds=5, seed=2026, restarts=12)
    splits = build_splits(records, allocation.center_to_fold, folds=5)
    certificate = validate_splits(splits, records)
    assert certificate["passed"] is True

    all_cases = {record.case_id for record in records}
    seen_val: set[str] = set()
    center_folds: dict[str, set[int]] = {}
    case_to_center = {record.case_id: record.center for record in records}
    for fold, split in enumerate(splits):
        train = set(split["train"])
        val = set(split["val"])
        assert train == all_cases - val
        assert not (train & val)
        assert not (seen_val & val)
        seen_val |= val
        for case_id in val:
            center_folds.setdefault(case_to_center[case_id], set()).add(fold)
    assert seen_val == all_cases
    assert all(len(folds) == 1 for folds in center_folds.values())


def test_deterministic_and_independent_of_manifest_order() -> None:
    rows = _synthetic_rows()
    forward = records_from_rows(rows)
    reverse = records_from_rows(reversed(rows))
    first = allocate_centers(forward, folds=5, seed=19, restarts=10)
    second = allocate_centers(reverse, folds=5, seed=19, restarts=10)
    third = allocate_centers(forward, folds=5, seed=19, restarts=10)
    assert first.center_to_fold == second.center_to_fold == third.center_to_fold
    assert first.objective == second.objective == third.objective
    assert assignment_input_sha256(forward) == assignment_input_sha256(reverse)


def test_assignment_hash_ignores_paths_and_numeric_formatting() -> None:
    left_rows = _synthetic_rows()
    right_rows = list(reversed(copy.deepcopy(left_rows)))
    for row in left_rows:
        row["image"] = f"/old/Dataset502/{row['nnunet_id']}.nii.gz"
    for row in right_rows:
        row["image"] = f"/immutable/Dataset507/{row['nnunet_id']}.nii.gz"
        row["lesion_volume_ml"] = f"{float(row['lesion_volume_ml']):.16e}"
    left = records_from_rows(left_rows)
    right = records_from_rows(right_rows)
    assert assignment_input_sha256(left) == assignment_input_sha256(right)

    changed_rows = copy.deepcopy(right_rows)
    changed_rows[0]["chronicity_bin"] = "changed_phase"
    changed = records_from_rows(changed_rows)
    assert assignment_input_sha256(left) != assignment_input_sha256(changed)


def test_balance_is_reasonable_for_synthetic_centers() -> None:
    records = records_from_rows(_synthetic_rows())
    allocation = allocate_centers(records, folds=5, seed=7, restarts=16)
    splits = build_splits(records, allocation.center_to_fold, folds=5)
    fold_sizes = [len(split["val"]) for split in splits]
    largest_center = max(
        sum(record.center == center for record in records)
        for center in {record.center for record in records}
    )
    # Grouping makes exact equality impossible in general. No fold should be
    # farther apart than the largest indivisible group in this fixture.
    assert max(fold_sizes) - min(fold_sizes) <= largest_center, fold_sizes
    assert all(split["val"] for split in splits)


def test_cli_writes_nnunet_and_audit_outputs() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        manifest = root / "conversion_manifest.csv"
        splits_path = root / "splits_final.json"
        summary_path = root / "center_grouped_splits_summary.json"
        markdown_path = root / "center_grouped_splits_summary.md"
        _write_manifest(manifest, _synthetic_rows())

        result = split_main(
            [
                "--manifest",
                str(manifest),
                "--out-splits",
                str(splits_path),
                "--out-summary-json",
                str(summary_path),
                "--out-summary-md",
                str(markdown_path),
                "--folds",
                "5",
                "--seed",
                "31",
                "--restarts",
                "8",
            ]
        )
        assert result == 0
        splits = json.loads(splits_path.read_text())
        summary = json.loads(summary_path.read_text())
        markdown = markdown_path.read_text()
        assert len(splits) == 5
        assert summary["validation"]["passed"] is True
        assert summary["global"]["n_cases"] == len(_synthetic_rows())
        assert len(summary["center_assignments"]) == 12
        assert all("train_centers" in fold for fold in summary["fold_summary"])
        assert all("deviation_from_target" in fold for fold in summary["fold_summary"])
        assert "Center assignments" in markdown
        assert summary["manifest_sha256"]
        assert summary["assignment_input_sha256"]


def test_rejects_duplicate_cases_and_too_few_centers() -> None:
    duplicate = _synthetic_rows()[:2]
    duplicate[1]["nnunet_id"] = duplicate[0]["nnunet_id"]
    try:
        records_from_rows(duplicate)
    except ManifestError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate case was accepted")

    rows = [
        {
            "nnunet_id": f"CASE_{index}",
            "site": f"CENTER_{index % 2}",
            "chronicity_bin": "unknown",
            "size_bin": "small",
            "lesion_volume_ml": "",
        }
        for index in range(6)
    ]
    records = records_from_rows(rows)
    try:
        allocate_centers(records, folds=5, seed=1, restarts=1)
    except ManifestError as exc:
        assert "at least 5 centers" in str(exc)
    else:
        raise AssertionError("allocation with too few centers was accepted")


def test_validator_rejects_a_split_center() -> None:
    records = records_from_rows(_synthetic_rows())
    allocation = allocate_centers(records, folds=5, seed=3, restarts=4)
    splits = build_splits(records, allocation.center_to_fold, folds=5)
    corrupted = copy.deepcopy(splits)
    case_to_center = {record.case_id: record.center for record in records}
    val = corrupted[0]["val"]
    center = case_to_center[val[0]]
    moved_case = next(case_id for case_id in val if case_to_center[case_id] == center)
    corrupted[0]["val"].remove(moved_case)
    corrupted[0]["train"].append(moved_case)
    try:
        validate_splits(corrupted, records)
    except SplitValidationError as exc:
        assert "splits center" in str(exc)
    else:
        raise AssertionError("validator accepted a center split across train and val")


def main() -> int:
    tests = [
        test_exact_coverage_and_center_exclusivity,
        test_deterministic_and_independent_of_manifest_order,
        test_assignment_hash_ignores_paths_and_numeric_formatting,
        test_balance_is_reasonable_for_synthetic_centers,
        test_cli_writes_nnunet_and_audit_outputs,
        test_rejects_duplicate_cases_and_too_few_centers,
        test_validator_rejects_a_split_center,
    ]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
