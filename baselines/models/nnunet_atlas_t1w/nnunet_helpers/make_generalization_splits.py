#!/usr/bin/env python3
"""Build leave-center-out and leave-chronicity-out split plans."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter((row.get(field) or "unknown") for row in rows).items()))


def split_summary(
    split_id: str,
    kind: str,
    holdout: str,
    train: list[str],
    val: list[str],
    row_by_case: dict[str, dict[str, str]],
) -> dict[str, object]:
    train_rows = [row_by_case[c] for c in train]
    val_rows = [row_by_case[c] for c in val]
    return {
        "split_id": split_id,
        "kind": kind,
        "holdout": holdout,
        "n_train": len(train),
        "n_val": len(val),
        "val_by_site": counts(val_rows, "site"),
        "val_by_chronicity": counts(val_rows, "chronicity_bin"),
        "val_by_size_bin": counts(val_rows, "size_bin"),
        "train_by_chronicity": counts(train_rows, "chronicity_bin"),
    }


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# Hidden-generalization split plan",
        "",
        "Splits are leave-group-out plans for retraining/evaluating hidden-generalization risk.",
        "The companion JSON is nnU-Net-compatible: copy one split into `splits_final.json` as index 0 for an isolated training run.",
        "",
        "| index | split_id | kind | holdout | train n | val n | val chronicity |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    for i, row in enumerate(summaries):
        chronicity = ", ".join(f"{k}:{v}" for k, v in row["val_by_chronicity"].items())
        lines.append(
            f"| {i} | {row['split_id']} | {row['kind']} | {row['holdout']} | "
            f"{row['n_train']} | {row['n_val']} | {chronicity} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-summary-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--min-center-cases", default=20, type=int)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    row_by_case = {row["nnunet_id"]: row for row in rows}
    all_cases = sorted(row_by_case)
    all_case_set = set(all_cases)

    splits: list[dict[str, list[str]]] = []
    summaries: list[dict[str, object]] = []

    by_site: dict[str, list[str]] = {}
    for row in rows:
        by_site.setdefault(row.get("site") or "unknown", []).append(row["nnunet_id"])
    for site, cases in sorted(by_site.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(cases) < args.min_center_cases:
            continue
        val = sorted(cases)
        train = sorted(all_case_set - set(val))
        split_id = f"center_{site}"
        splits.append({"train": train, "val": val})
        summaries.append(split_summary(split_id, "center_holdout", site, train, val, row_by_case))

    by_chronicity: dict[str, list[str]] = {}
    for row in rows:
        by_chronicity.setdefault(row.get("chronicity_bin") or "unknown", []).append(row["nnunet_id"])
    for chronicity, cases in sorted(by_chronicity.items()):
        val = sorted(cases)
        train = sorted(all_case_set - set(val))
        split_id = f"chronicity_{chronicity}"
        splits.append({"train": train, "val": val})
        summaries.append(split_summary(split_id, "chronicity_holdout", chronicity, train, val, row_by_case))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(splits, indent=2) + "\n")
    args.out_summary_json.write_text(json.dumps({"n_cases": len(all_cases), "splits": summaries}, indent=2) + "\n")
    write_markdown(args.out_md, summaries)
    print(f"[done] wrote {len(splits)} split(s) to {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
