#!/usr/bin/env python3
"""Build deterministic center-grouped, metadata-balanced nnU-Net folds.

Every center is assigned to exactly one validation fold. The allocator balances
case count, chronicity, lesion-size strata, and lesion-volume summaries without
ever splitting a center. It uses seeded greedy restarts followed by deterministic
single-center move and center-swap refinement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


CASE_FIELD = "nnunet_id"
CENTER_FIELD = "site"
CHRONICITY_FIELD = "chronicity_bin"
SIZE_FIELD = "size_bin"
VOLUME_FIELD = "lesion_volume_ml"
ALGORITHM_VERSION = "center_grouped_v1"
EPSILON = 1e-12

# Each metadata family contributes as one objective term, independent of how
# many bins it contains. Raw and log-volume terms keep both total lesion burden
# and the lesion-size distribution visible to the optimizer.
OBJECTIVE_WEIGHTS = {
    "case_count": 4.0,
    "chronicity": 3.0,
    "size_bin": 3.0,
    "volume_known_count": 0.5,
    "volume_sum_ml": 0.5,
    "volume_log1p_sum": 1.0,
}


class ManifestError(ValueError):
    """Raised when a conversion manifest cannot define valid grouped folds."""


class SplitValidationError(ValueError):
    """Raised when generated splits violate a coverage or grouping invariant."""


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    center: str
    chronicity: str
    size_bin: str
    lesion_volume_ml: float | None


@dataclass(frozen=True)
class CenterProfile:
    center: str
    case_ids: tuple[str, ...]
    vector: tuple[float, ...]

    @property
    def n_cases(self) -> int:
        return len(self.case_ids)


@dataclass(frozen=True)
class BalanceContext:
    folds: int
    chronicity_bins: tuple[str, ...]
    size_bins: tuple[str, ...]
    target: tuple[float, ...]

    @property
    def chronicity_slice(self) -> slice:
        return slice(1, 1 + len(self.chronicity_bins))

    @property
    def size_slice(self) -> slice:
        start = 1 + len(self.chronicity_bins)
        return slice(start, start + len(self.size_bins))

    @property
    def volume_start(self) -> int:
        return 1 + len(self.chronicity_bins) + len(self.size_bins)


@dataclass(frozen=True)
class Allocation:
    center_to_fold: dict[str, int]
    objective: float
    objective_breakdown: dict[str, float]
    restarts: int


def _category(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return text or "unknown"


def _parse_volume(value: object, case_id: str) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"na", "n/a", "nan", "none", "null"}:
        return None
    try:
        volume = float(text)
    except ValueError as exc:
        raise ManifestError(
            f"case {case_id!r} has invalid {VOLUME_FIELD}: {text!r}"
        ) from exc
    if not math.isfinite(volume) or volume < 0:
        raise ManifestError(
            f"case {case_id!r} has invalid {VOLUME_FIELD}: {text!r}"
        )
    return volume


def records_from_rows(rows: Iterable[Mapping[str, object]]) -> list[CaseRecord]:
    records: list[CaseRecord] = []
    seen: set[str] = set()
    for row_index, row in enumerate(rows, start=2):
        case_id = str(row.get(CASE_FIELD) or "").strip()
        if not case_id:
            raise ManifestError(f"manifest row {row_index} is missing {CASE_FIELD}")
        if case_id in seen:
            raise ManifestError(f"duplicate {CASE_FIELD}: {case_id!r}")
        seen.add(case_id)

        center = str(row.get(CENTER_FIELD) or "").strip()
        if not center:
            raise ManifestError(
                f"case {case_id!r} is missing {CENTER_FIELD}; blank centers cannot be grouped"
            )
        records.append(
            CaseRecord(
                case_id=case_id,
                center=center,
                chronicity=_category(row.get(CHRONICITY_FIELD)),
                size_bin=_category(row.get(SIZE_FIELD)),
                lesion_volume_ml=_parse_volume(row.get(VOLUME_FIELD), case_id),
            )
        )
    if not records:
        raise ManifestError("manifest contains no cases")
    return sorted(records, key=lambda record: record.case_id)


def read_manifest(path: Path) -> list[CaseRecord]:
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ManifestError(f"manifest has no CSV header: {path}")
            missing = {CASE_FIELD, CENTER_FIELD} - set(reader.fieldnames)
            if missing:
                raise ManifestError(
                    f"manifest is missing required column(s): {', '.join(sorted(missing))}"
                )
            return records_from_rows(reader)
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc


def _build_profiles(
    records: Sequence[CaseRecord], folds: int
) -> tuple[list[CenterProfile], BalanceContext]:
    chronicity_bins = tuple(sorted({record.chronicity for record in records}))
    size_bins = tuple(sorted({record.size_bin for record in records}))
    chronicity_index = {name: index for index, name in enumerate(chronicity_bins)}
    size_index = {name: index for index, name in enumerate(size_bins)}

    by_center: dict[str, list[CaseRecord]] = defaultdict(list)
    for record in records:
        by_center[record.center].append(record)
    if len(by_center) < folds:
        raise ManifestError(
            f"need at least {folds} centers for {folds} non-empty folds; "
            f"manifest has {len(by_center)}"
        )

    profiles: list[CenterProfile] = []
    vector_length = 1 + len(chronicity_bins) + len(size_bins) + 3
    for center in sorted(by_center):
        center_records = sorted(by_center[center], key=lambda record: record.case_id)
        vector = [0.0] * vector_length
        vector[0] = float(len(center_records))
        size_offset = 1 + len(chronicity_bins)
        volume_offset = size_offset + len(size_bins)
        for record in center_records:
            vector[1 + chronicity_index[record.chronicity]] += 1.0
            vector[size_offset + size_index[record.size_bin]] += 1.0
            if record.lesion_volume_ml is not None:
                vector[volume_offset] += 1.0
                vector[volume_offset + 1] += record.lesion_volume_ml
                vector[volume_offset + 2] += math.log1p(record.lesion_volume_ml)
        profiles.append(
            CenterProfile(
                center=center,
                case_ids=tuple(record.case_id for record in center_records),
                vector=tuple(vector),
            )
        )

    totals = [math.fsum(profile.vector[i] for profile in profiles) for i in range(vector_length)]
    context = BalanceContext(
        folds=folds,
        chronicity_bins=chronicity_bins,
        size_bins=size_bins,
        target=tuple(value / folds for value in totals),
    )
    return profiles, context


def _normalized_mse(values: Sequence[float], target: float) -> float:
    if target <= EPSILON:
        return 0.0
    scale = max(target, 1.0)
    return math.fsum(((value - target) / scale) ** 2 for value in values) / len(values)


def _objective_breakdown(
    loads: Sequence[Sequence[float]], context: BalanceContext
) -> dict[str, float]:
    chronicity_terms = [
        _normalized_mse([load[index] for load in loads], context.target[index])
        for index in range(context.chronicity_slice.start, context.chronicity_slice.stop)
    ]
    size_terms = [
        _normalized_mse([load[index] for load in loads], context.target[index])
        for index in range(context.size_slice.start, context.size_slice.stop)
    ]
    volume_start = context.volume_start
    return {
        "case_count": _normalized_mse([load[0] for load in loads], context.target[0]),
        "chronicity": math.fsum(chronicity_terms) / max(len(chronicity_terms), 1),
        "size_bin": math.fsum(size_terms) / max(len(size_terms), 1),
        "volume_known_count": _normalized_mse(
            [load[volume_start] for load in loads], context.target[volume_start]
        ),
        "volume_sum_ml": _normalized_mse(
            [load[volume_start + 1] for load in loads], context.target[volume_start + 1]
        ),
        "volume_log1p_sum": _normalized_mse(
            [load[volume_start + 2] for load in loads], context.target[volume_start + 2]
        ),
    }


def _objective(loads: Sequence[Sequence[float]], context: BalanceContext) -> float:
    breakdown = _objective_breakdown(loads, context)
    return math.fsum(OBJECTIVE_WEIGHTS[name] * value for name, value in breakdown.items())


def _update_load(load: list[float], vector: Sequence[float], sign: float) -> None:
    for index, value in enumerate(vector):
        load[index] += sign * value


def _profile_difficulty(profile: CenterProfile, context: BalanceContext) -> float:
    # The norm prioritizes large centers and centers concentrated in rare strata.
    terms: list[float] = []
    for index, value in enumerate(profile.vector):
        target = max(context.target[index], 1.0)
        terms.append((value / target) ** 2)
    return math.sqrt(math.fsum(terms))


def _greedy_assignment(
    profiles: Sequence[CenterProfile], context: BalanceContext, rng: random.Random
) -> tuple[dict[str, int], list[list[float]], list[int]]:
    jitter = {profile.center: 0.90 + 0.20 * rng.random() for profile in profiles}
    ordered = sorted(
        profiles,
        key=lambda profile: (
            -_profile_difficulty(profile, context) * jitter[profile.center],
            profile.center,
        ),
    )
    loads = [[0.0] * len(context.target) for _ in range(context.folds)]
    center_counts = [0] * context.folds
    assignment: dict[str, int] = {}

    fold_tie_order = list(range(context.folds))
    rng.shuffle(fold_tie_order)
    tie_rank = {fold: rank for rank, fold in enumerate(fold_tie_order)}
    for profile in ordered:
        empty_folds = [fold for fold, count in enumerate(center_counts) if count == 0]
        candidate_folds = empty_folds or list(range(context.folds))
        candidates: list[tuple[float, int, int]] = []
        for fold in candidate_folds:
            _update_load(loads[fold], profile.vector, 1.0)
            score = _objective(loads, context)
            _update_load(loads[fold], profile.vector, -1.0)
            candidates.append((score, tie_rank[fold], fold))
        fold = min(candidates)[2]
        assignment[profile.center] = fold
        center_counts[fold] += 1
        _update_load(loads[fold], profile.vector, 1.0)
    return assignment, loads, center_counts


def _refine_assignment(
    profiles: Sequence[CenterProfile],
    context: BalanceContext,
    assignment: dict[str, int],
    loads: list[list[float]],
    center_counts: list[int],
    rng: random.Random,
) -> float:
    profile_by_center = {profile.center: profile for profile in profiles}
    centers = sorted(profile_by_center)
    tie_order = centers[:]
    rng.shuffle(tie_order)
    tie_rank = {center: rank for rank, center in enumerate(tie_order)}
    current_score = _objective(loads, context)

    # Strictly improving moves and swaps guarantee termination. A generous cap
    # protects CLI runs if objective changes make a very long improvement path.
    for _ in range(max(20, len(centers) * 6)):
        best_move: tuple[float, int, str, int] | None = None
        for center in centers:
            source = assignment[center]
            if center_counts[source] <= 1:
                continue
            profile = profile_by_center[center]
            for destination in range(context.folds):
                if destination == source:
                    continue
                _update_load(loads[source], profile.vector, -1.0)
                _update_load(loads[destination], profile.vector, 1.0)
                score = _objective(loads, context)
                _update_load(loads[destination], profile.vector, -1.0)
                _update_load(loads[source], profile.vector, 1.0)
                candidate = (score, tie_rank[center], center, destination)
                if best_move is None or candidate < best_move:
                    best_move = candidate

        if best_move is not None and best_move[0] < current_score - EPSILON:
            score, _, center, destination = best_move
            source = assignment[center]
            profile = profile_by_center[center]
            _update_load(loads[source], profile.vector, -1.0)
            _update_load(loads[destination], profile.vector, 1.0)
            center_counts[source] -= 1
            center_counts[destination] += 1
            assignment[center] = destination
            current_score = score
            continue

        best_swap: tuple[float, int, int, str, str] | None = None
        for left_index, left_center in enumerate(centers):
            left_fold = assignment[left_center]
            left_profile = profile_by_center[left_center]
            for right_center in centers[left_index + 1 :]:
                right_fold = assignment[right_center]
                if left_fold == right_fold:
                    continue
                right_profile = profile_by_center[right_center]
                _update_load(loads[left_fold], left_profile.vector, -1.0)
                _update_load(loads[left_fold], right_profile.vector, 1.0)
                _update_load(loads[right_fold], right_profile.vector, -1.0)
                _update_load(loads[right_fold], left_profile.vector, 1.0)
                score = _objective(loads, context)
                _update_load(loads[right_fold], left_profile.vector, -1.0)
                _update_load(loads[right_fold], right_profile.vector, 1.0)
                _update_load(loads[left_fold], right_profile.vector, -1.0)
                _update_load(loads[left_fold], left_profile.vector, 1.0)
                candidate = (
                    score,
                    tie_rank[left_center],
                    tie_rank[right_center],
                    left_center,
                    right_center,
                )
                if best_swap is None or candidate < best_swap:
                    best_swap = candidate

        if best_swap is None or best_swap[0] >= current_score - EPSILON:
            break
        score, _, _, left_center, right_center = best_swap
        left_fold = assignment[left_center]
        right_fold = assignment[right_center]
        left_profile = profile_by_center[left_center]
        right_profile = profile_by_center[right_center]
        _update_load(loads[left_fold], left_profile.vector, -1.0)
        _update_load(loads[left_fold], right_profile.vector, 1.0)
        _update_load(loads[right_fold], right_profile.vector, -1.0)
        _update_load(loads[right_fold], left_profile.vector, 1.0)
        assignment[left_center] = right_fold
        assignment[right_center] = left_fold
        current_score = score
    return current_score


def allocate_centers(
    records: Sequence[CaseRecord], folds: int = 5, seed: int = 42, restarts: int = 64
) -> Allocation:
    if folds < 2:
        raise ManifestError(f"folds must be at least 2, got {folds}")
    if restarts < 1:
        raise ManifestError(f"restarts must be at least 1, got {restarts}")
    profiles, context = _build_profiles(records, folds)
    master_rng = random.Random(seed)
    best: tuple[float, tuple[tuple[str, int], ...], dict[str, int], dict[str, float]] | None = None

    for _ in range(restarts):
        restart_rng = random.Random(master_rng.getrandbits(64))
        assignment, loads, center_counts = _greedy_assignment(profiles, context, restart_rng)
        score = _refine_assignment(
            profiles, context, assignment, loads, center_counts, restart_rng
        )
        signature = tuple(sorted(assignment.items()))
        breakdown = _objective_breakdown(loads, context)
        candidate = (score, signature, dict(assignment), breakdown)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    assert best is not None
    return Allocation(
        center_to_fold=best[2],
        objective=best[0],
        objective_breakdown=best[3],
        restarts=restarts,
    )


def build_splits(
    records: Sequence[CaseRecord], center_to_fold: Mapping[str, int], folds: int
) -> list[dict[str, list[str]]]:
    all_cases = {record.case_id for record in records}
    val_by_fold: list[list[str]] = [[] for _ in range(folds)]
    for record in records:
        if record.center not in center_to_fold:
            raise SplitValidationError(f"center missing from assignment: {record.center!r}")
        fold = center_to_fold[record.center]
        if not 0 <= fold < folds:
            raise SplitValidationError(
                f"center {record.center!r} has invalid fold {fold}; expected 0..{folds - 1}"
            )
        val_by_fold[fold].append(record.case_id)
    return [
        {
            "train": sorted(all_cases - set(val_cases)),
            "val": sorted(val_cases),
        }
        for val_cases in val_by_fold
    ]


def validate_splits(
    splits: Sequence[Mapping[str, Sequence[str]]], records: Sequence[CaseRecord]
) -> dict[str, object]:
    expected_cases = {record.case_id for record in records}
    case_to_center = {record.case_id: record.center for record in records}
    expected_centers = set(case_to_center.values())
    val_counts: Counter[str] = Counter()
    center_to_val_fold: dict[str, int] = {}

    if len(splits) < 2:
        raise SplitValidationError("at least two folds are required")
    for fold, split in enumerate(splits):
        train_list = list(split.get("train", []))
        val_list = list(split.get("val", []))
        train = set(train_list)
        val = set(val_list)
        if len(train) != len(train_list):
            raise SplitValidationError(f"fold {fold} contains duplicate training cases")
        if len(val) != len(val_list):
            raise SplitValidationError(f"fold {fold} contains duplicate validation cases")
        unknown = (train | val) - expected_cases
        if unknown:
            raise SplitValidationError(
                f"fold {fold} contains unknown cases: {sorted(unknown)[:5]}"
            )
        if train & val:
            raise SplitValidationError(
                f"fold {fold} train/val overlap: {sorted(train & val)[:5]}"
            )
        if train != expected_cases - val:
            raise SplitValidationError(f"fold {fold} training set is not the validation complement")
        if not train or not val:
            raise SplitValidationError(f"fold {fold} has an empty train or validation set")

        train_centers = {case_to_center[case_id] for case_id in train}
        val_centers = {case_to_center[case_id] for case_id in val}
        if train_centers & val_centers:
            raise SplitValidationError(
                f"fold {fold} splits center(s) between train and val: "
                f"{sorted(train_centers & val_centers)}"
            )
        for center in val_centers:
            previous = center_to_val_fold.setdefault(center, fold)
            if previous != fold:
                raise SplitValidationError(
                    f"center {center!r} appears in validation folds {previous} and {fold}"
                )
        val_counts.update(val)

    missing = expected_cases - set(val_counts)
    repeated = sorted(case_id for case_id, count in val_counts.items() if count != 1)
    if missing or repeated:
        raise SplitValidationError(
            f"validation coverage is not exactly once: missing={sorted(missing)[:5]}, "
            f"non_unique={repeated[:5]}"
        )
    if set(center_to_val_fold) != expected_centers:
        missing_centers = sorted(expected_centers - set(center_to_val_fold))
        raise SplitValidationError(
            f"centers missing from validation coverage: {missing_centers}"
        )
    return {
        "passed": True,
        "n_cases": len(expected_cases),
        "n_centers": len(expected_centers),
        "n_folds": len(splits),
        "each_case_in_validation_exactly_once": True,
        "train_val_disjoint_in_every_fold": True,
        "train_is_validation_complement_in_every_fold": True,
        "each_center_in_one_validation_fold": True,
        "center_exclusive_train_val_in_every_fold": True,
    }


def _counts(records: Iterable[CaseRecord], attribute: str) -> dict[str, int]:
    return dict(sorted(Counter(getattr(record, attribute) for record in records).items()))


def _volume_summary(records: Sequence[CaseRecord]) -> dict[str, int | float | None]:
    volumes = [
        record.lesion_volume_ml
        for record in records
        if record.lesion_volume_ml is not None
    ]
    return {
        "n_known": len(volumes),
        "n_missing": len(records) - len(volumes),
        "sum_ml": math.fsum(volumes),
        "mean_ml": math.fsum(volumes) / len(volumes) if volumes else None,
        "median_ml": statistics.median(volumes) if volumes else None,
        "log1p_sum": math.fsum(math.log1p(volume) for volume in volumes),
    }


def assignment_input_sha256(records: Sequence[CaseRecord]) -> str:
    """Hash only the normalized fields that can influence fold assignment.

    This intentionally excludes manifest location, source paths, checksums, and
    row order. Equivalent Dataset502 and immutable Dataset507 manifests can
    therefore prove that they supplied the same assignment inputs.
    """

    canonical_rows = [
        [
            record.case_id,
            record.center,
            record.chronicity,
            record.size_bin,
            None
            if record.lesion_volume_ml is None
            else format(record.lesion_volume_ml, ".17g"),
        ]
        for record in sorted(records, key=lambda item: item.case_id)
    ]
    payload = json.dumps(
        canonical_rows, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _distribution(values: Sequence[float]) -> dict[str, object]:
    mean = math.fsum(values) / len(values)
    return {
        "values_by_fold": list(values),
        "target": mean,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "coefficient_of_variation": statistics.pstdev(values) / mean if mean else 0.0,
    }


def _deviation(actual: float, target: float) -> dict[str, float | None]:
    return {
        "actual": actual,
        "target": target,
        "difference": actual - target,
        "relative_difference": (actual - target) / target if target else None,
    }


def _imbalance_summary(
    fold_records: Sequence[Sequence[CaseRecord]],
    chronicity_bins: Sequence[str],
    size_bins: Sequence[str],
) -> dict[str, object]:
    chronicity_counts = [_counts(records, "chronicity") for records in fold_records]
    size_counts = [_counts(records, "size_bin") for records in fold_records]
    volumes = [_volume_summary(records) for records in fold_records]
    return {
        "case_count": _distribution([len(records) for records in fold_records]),
        "chronicity": {
            name: _distribution([counts.get(name, 0) for counts in chronicity_counts])
            for name in chronicity_bins
        },
        "size_bin": {
            name: _distribution([counts.get(name, 0) for counts in size_counts])
            for name in size_bins
        },
        "volume_known_count": _distribution([volume["n_known"] for volume in volumes]),
        "volume_sum_ml": _distribution([volume["sum_ml"] for volume in volumes]),
        "volume_log1p_sum": _distribution([volume["log1p_sum"] for volume in volumes]),
    }


def build_summary(
    records: Sequence[CaseRecord],
    splits: Sequence[Mapping[str, Sequence[str]]],
    allocation: Allocation,
    manifest_path: Path,
    manifest_sha256: str,
    seed: int,
) -> dict[str, object]:
    row_by_case = {record.case_id: record for record in records}
    folds = len(splits)
    fold_records = [
        [row_by_case[case_id] for case_id in split["val"]] for split in splits
    ]
    all_centers = sorted({record.center for record in records})
    chronicity_bins = sorted({record.chronicity for record in records})
    size_bins = sorted({record.size_bin for record in records})
    global_chronicity = _counts(records, "chronicity")
    global_size = _counts(records, "size_bin")
    global_volume = _volume_summary(records)
    target_case_count = len(records) / folds
    target_chronicity = {
        name: global_chronicity.get(name, 0) / folds for name in chronicity_bins
    }
    target_size = {name: global_size.get(name, 0) / folds for name in size_bins}
    target_volume_known = global_volume["n_known"] / folds
    target_volume_sum = global_volume["sum_ml"] / folds
    target_volume_log_sum = global_volume["log1p_sum"] / folds

    fold_summary: list[dict[str, object]] = []
    for fold, split in enumerate(splits):
        train_records = [row_by_case[case_id] for case_id in split["train"]]
        val_records = fold_records[fold]
        val_centers = sorted({record.center for record in val_records})
        train_centers = sorted(set(all_centers) - set(val_centers))
        val_chronicity = _counts(val_records, "chronicity")
        val_size = _counts(val_records, "size_bin")
        val_volume = _volume_summary(val_records)
        fold_summary.append(
            {
                "fold": fold,
                "n_train": len(train_records),
                "n_val": len(val_records),
                "n_train_centers": len(train_centers),
                "n_val_centers": len(val_centers),
                "train_centers": train_centers,
                "val_centers": val_centers,
                "train_by_chronicity": _counts(train_records, "chronicity"),
                "val_by_chronicity": val_chronicity,
                "train_by_size_bin": _counts(train_records, "size_bin"),
                "val_by_size_bin": val_size,
                "train_lesion_volume": _volume_summary(train_records),
                "val_lesion_volume": val_volume,
                "deviation_from_target": {
                    "case_count": _deviation(len(val_records), target_case_count),
                    "chronicity": {
                        name: _deviation(
                            val_chronicity.get(name, 0), target_chronicity[name]
                        )
                        for name in chronicity_bins
                    },
                    "size_bin": {
                        name: _deviation(val_size.get(name, 0), target_size[name])
                        for name in size_bins
                    },
                    "volume_known_count": _deviation(
                        val_volume["n_known"], target_volume_known
                    ),
                    "volume_sum_ml": _deviation(
                        val_volume["sum_ml"], target_volume_sum
                    ),
                    "volume_log1p_sum": _deviation(
                        val_volume["log1p_sum"], target_volume_log_sum
                    ),
                },
            }
        )

    by_center: dict[str, list[CaseRecord]] = defaultdict(list)
    for record in records:
        by_center[record.center].append(record)
    center_assignments = []
    for center in sorted(by_center, key=lambda name: (allocation.center_to_fold[name], name)):
        center_records = sorted(by_center[center], key=lambda record: record.case_id)
        center_assignments.append(
            {
                "center": center,
                "fold": allocation.center_to_fold[center],
                "n_cases": len(center_records),
                "by_chronicity": _counts(center_records, "chronicity"),
                "by_size_bin": _counts(center_records, "size_bin"),
                "lesion_volume": _volume_summary(center_records),
            }
        )

    return {
        "schema_version": 1,
        "algorithm": ALGORITHM_VERSION,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "assignment_input_sha256": assignment_input_sha256(records),
        "assignment_input_fields": [
            CASE_FIELD,
            CENTER_FIELD,
            CHRONICITY_FIELD,
            SIZE_FIELD,
            VOLUME_FIELD,
        ],
        "seed": seed,
        "folds": folds,
        "restarts": allocation.restarts,
        "group_field": CENTER_FIELD,
        "balance_fields": [
            CASE_FIELD,
            CHRONICITY_FIELD,
            SIZE_FIELD,
            VOLUME_FIELD,
        ],
        "objective": {
            "value": allocation.objective,
            "weights": dict(OBJECTIVE_WEIGHTS),
            "unweighted_breakdown": allocation.objective_breakdown,
        },
        "validation": validate_splits(splits, records),
        "global": {
            "n_cases": len(records),
            "n_centers": len(all_centers),
            "by_chronicity": global_chronicity,
            "by_size_bin": global_size,
            "lesion_volume": global_volume,
        },
        "target_per_fold": {
            "case_count": target_case_count,
            "by_chronicity": target_chronicity,
            "by_size_bin": target_size,
            "volume_known_count": target_volume_known,
            "volume_sum_ml": target_volume_sum,
            "volume_log1p_sum": target_volume_log_sum,
        },
        "imbalance": _imbalance_summary(fold_records, chronicity_bins, size_bins),
        "fold_summary": fold_summary,
        "center_assignments": center_assignments,
    }


def _compact_counts(counts: Mapping[str, object]) -> str:
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def _markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def summary_markdown(summary: Mapping[str, object]) -> str:
    validation = summary["validation"]
    global_summary = summary["global"]
    objective = summary["objective"]
    lines = [
        f"# Center-grouped {summary['folds']}-fold split plan",
        "",
        f"- Manifest: `{_markdown_escape(summary['manifest'])}`",
        f"- Manifest SHA-256: `{summary['manifest_sha256']}`",
        f"- Assignment-input SHA-256: `{summary['assignment_input_sha256']}`",
        f"- Assignment-input fields: `{', '.join(summary['assignment_input_fields'])}`",
        f"- Algorithm: `{summary['algorithm']}`; seed `{summary['seed']}`; "
        f"restarts `{summary['restarts']}`",
        f"- Cases: {global_summary['n_cases']}; centers: {global_summary['n_centers']}",
        f"- Optimizer objective: {objective['value']:.8f}",
        "",
        "## Validation",
        "",
        f"All invariants passed: **{str(validation['passed']).lower()}**. Each case appears "
        "in validation exactly once, every training set is its validation complement, "
        "and every center belongs to exactly one validation fold.",
        "",
        "## Fold balance",
        "",
        "| fold | val centers | val cases | case delta | chronicity | lesion size | volume known | volume sum (mL) | volume delta (mL) |",
        "|---:|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for fold in summary["fold_summary"]:
        volume = fold["val_lesion_volume"]
        deviation = fold["deviation_from_target"]
        lines.append(
            f"| {fold['fold']} | {fold['n_val_centers']} | {fold['n_val']} | "
            f"{deviation['case_count']['difference']:+.1f} | "
            f"{_markdown_escape(_compact_counts(fold['val_by_chronicity']))} | "
            f"{_markdown_escape(_compact_counts(fold['val_by_size_bin']))} | "
            f"{volume['n_known']} | {volume['sum_ml']:.3f} | "
            f"{deviation['volume_sum_ml']['difference']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Center assignments",
            "",
            "| fold | center | cases | chronicity | lesion size | volume sum (mL) |",
            "|---:|---|---:|---|---|---:|",
        ]
    )
    for center in summary["center_assignments"]:
        lines.append(
            f"| {center['fold']} | {_markdown_escape(center['center'])} | {center['n_cases']} | "
            f"{_markdown_escape(_compact_counts(center['by_chronicity']))} | "
            f"{_markdown_escape(_compact_counts(center['by_size_bin']))} | "
            f"{center['lesion_volume']['sum_ml']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-splits", required=True, type=Path)
    parser.add_argument("--out-summary-json", required=True, type=Path)
    parser.add_argument("--out-summary-md", required=True, type=Path)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--restarts",
        default=64,
        type=int,
        help="seeded greedy/local-search restarts (default: 64)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_paths = [args.out_splits, args.out_summary_json, args.out_summary_md]
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        print("[error] output paths must be distinct", file=sys.stderr)
        return 2
    try:
        records = read_manifest(args.manifest)
        allocation = allocate_centers(records, args.folds, args.seed, args.restarts)
        splits = build_splits(records, allocation.center_to_fold, args.folds)
        validate_splits(splits, records)
        summary = build_summary(
            records=records,
            splits=splits,
            allocation=allocation,
            manifest_path=args.manifest,
            manifest_sha256=_sha256(args.manifest),
            seed=args.seed,
        )
    except (ManifestError, SplitValidationError, OSError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    _write_text_atomic(args.out_splits, json.dumps(splits, indent=2) + "\n")
    _write_text_atomic(args.out_summary_json, json.dumps(summary, indent=2) + "\n")
    _write_text_atomic(args.out_summary_md, summary_markdown(summary))
    print(
        f"[done] wrote {args.folds} center-grouped folds for {len(records)} cases / "
        f"{summary['global']['n_centers']} centers; objective={allocation.objective:.8f}"
    )
    print(f"[done] splits: {args.out_splits}")
    print(f"[done] summary: {args.out_summary_json} and {args.out_summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
