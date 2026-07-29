#!/usr/bin/env python3
"""Direct lightweight tests for characterize_training_data.py chronicity parsing.

The chronicity derivation is load-bearing: it decides the `chronicity_bin`
stratum for ~1450 training cases, and the CHRONICITY fallback alone rescues 270
of them out of `unknown`. It had no other automated coverage, so the encoding
assumption (CHRONICITY==1 means CHRONIC, >=180 days) and the audit tripwire that
guards it are pinned here.

Run from the repository root:
  python3 scripts/analysis/test_characterize_training_data.py
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis.characterize_training_data import (  # noqa: E402
    FIELDS,
    _parse_float,
    chronicity_bin,
    new_chronicity_audit,
    parse_metadata,
    update_chronicity_audit,
)


def _write_metadata(directory: Path, **cells: str) -> Path:
    path = directory / "meta.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells))
        writer.writeheader()
        writer.writerow(cells)
    return path


def test_days_post_stroke_is_authoritative() -> None:
    assert chronicity_bin(10.0, None) == ("lt180d", "days_post_stroke")
    assert chronicity_bin(179.9, None) == ("lt180d", "days_post_stroke")
    # 180 is the chronic boundary and belongs to ge180d.
    assert chronicity_bin(180.0, None) == ("ge180d", "days_post_stroke")
    # A usable DPS wins even when CHRONICITY claims otherwise.
    assert chronicity_bin(10.0, 1.0) == ("lt180d", "days_post_stroke")


def test_non_positive_days_post_stroke_counts_as_missing() -> None:
    # R049/R050 encode missing DAYS_POST_STROKE as 0; one R049 session is -4.
    assert chronicity_bin(0.0, None) == ("unknown", "")
    assert chronicity_bin(-4.0, None) == ("unknown", "")
    assert chronicity_bin(None, None) == ("unknown", "")


def test_chronicity_flag_only_rescues_into_ge180d() -> None:
    assert chronicity_bin(None, 1.0) == ("ge180d", "chronicity_flag")
    assert chronicity_bin(0.0, 1.0) == ("ge180d", "chronicity_flag")
    # Any code other than 1 is outside the established encoding: stay unknown
    # rather than guess a bin.
    assert chronicity_bin(None, 0.0) == ("unknown", "")
    assert chronicity_bin(None, 2.0) == ("unknown", "")


def test_parse_float_missing_tokens_and_non_finite() -> None:
    for token in ("", "  ", "NA", "n/a", "NaN", "Nan", "none", "NULL"):
        assert _parse_float(token) is None, token
    assert _parse_float("junk") is None
    assert _parse_float(None) is None
    # Non-finite must not leak: NaN compares False against every threshold and a
    # naive `dps < 180` reader would silently bin it as chronic.
    assert _parse_float("inf") is None
    assert _parse_float(" 182.5 ") == 182.5


def test_parse_metadata_returns_raw_and_value() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        meta = _write_metadata(tmp_path, DAYS_POST_STROKE="471", SITE="R001",
                               CHRONICITY="")
        assert parse_metadata(meta) == (471.0, "R001", "", None)

        meta = _write_metadata(tmp_path, DAYS_POST_STROKE="", SITE="SOOP",
                               CHRONICITY="1.0")
        assert parse_metadata(meta) == (None, "SOOP", "1.0", 1.0)

        # A release without the CHRONICITY column must not crash.
        meta = _write_metadata(tmp_path, DAYS_POST_STROKE="30", SITE="R001")
        assert parse_metadata(meta) == (30.0, "R001", "", None)

        assert parse_metadata(tmp_path / "absent.csv") == (None, None, "", None)


def test_audit_counters_and_tripwire() -> None:
    audit = new_chronicity_audit()
    update_chronicity_audit(audit, 471.0, "", None, "ge180d")    # DPS only
    update_chronicity_audit(audit, None, "1.0", 1.0, "ge180d")   # rescued by flag
    update_chronicity_audit(audit, 182.5, "1.0", 1.0, "ge180d")  # both, agreeing
    assert audit["n_sessions"] == 3
    assert audit["dps_usable"] == 2
    assert audit["chronicity_present"] == 2
    assert audit["both_present"] == 1
    assert audit["both_agree_ge180d"] == 1
    assert audit["rescued_by_chronicity"] == 1
    assert audit["both_disagree"] == 0
    assert audit["unexpected_chronicity_values"] == {}
    assert audit["bins"] == {"lt180d": 0, "ge180d": 3, "unknown": 0}

    # Tripwire: a release that flips the encoding (CHRONICITY==1 on an acute
    # case) or introduces a second code must show up in the audit.
    flipped = new_chronicity_audit()
    update_chronicity_audit(flipped, 30.0, "1", 1.0, "lt180d")
    assert flipped["both_disagree"] == 1
    update_chronicity_audit(flipped, None, "2", 2.0, "unknown")
    update_chronicity_audit(flipped, None, "chronic", None, "unknown")
    assert flipped["unexpected_chronicity_values"] == {"2": 1, "chronic": 1}
    assert flipped["chronicity_present"] == 3


def test_new_columns_are_declared_in_fields() -> None:
    for name in ("chronicity_lt180d", "chronicity_raw", "chronicity_bin",
                 "chronicity_source"):
        assert name in FIELDS, name
    assert len(FIELDS) == len(set(FIELDS))


def main() -> int:
    tests = [
        test_days_post_stroke_is_authoritative,
        test_non_positive_days_post_stroke_counts_as_missing,
        test_chronicity_flag_only_rescues_into_ge180d,
        test_parse_float_missing_tokens_and_non_finite,
        test_parse_metadata_returns_raw_and_value,
        test_audit_counters_and_tripwire,
        test_new_columns_are_declared_in_fields,
    ]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
