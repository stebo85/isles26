#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a model-comparison leaderboard from per-method reports.

Two properties of this script are deliberate and load-bearing:

1. A report directory that holds a report.json but no meta.json is a hard error,
   not a skip. Silently dropping such a directory is how the held-center
   cross-validation -- the only evaluation here whose folds hold out whole
   centers, and therefore the closest proxy we have for the 60+ center hidden
   test set -- stayed invisible while a center-leaking row was presented as the
   current best. An unreadable results directory must stop the build.
2. Every row carries its evaluator provenance. Rows produced by
   nnunet_summary_to_report.py drop cases that are empty in both prediction and
   reference, while rows produced by the repo harness score those cases as 1.0
   (the ISLES'26 convention). The two conventions differ by a couple of cases
   out of ~1450 and are not comparable at the third decimal, so the evaluator
   and the number of cases actually scored are printed next to every number.
"""

from __future__ import print_function, unicode_literals

import argparse
import io
import json
import math
import os
from collections import defaultdict
from datetime import datetime

try:
    string_types = (basestring,)
except NameError:
    string_types = (str,)

try:
    number_types = (int, long, float)
except NameError:
    number_types = (int, float)


CSV_COLUMNS = [
    "eval_set",
    "method",
    "display_name",
    "model_family",
    "input_modalities",
    "evaluator",
    "empty_case_convention",
    "n_cases",
    "n_scored",
    "dice_mean",
    "dice_median",
    "lesion_f1_mean",
    "lesion_recall_mean",
    "surface_dice_3mm_mean",
    "hd95_mm_mean",
    "abs_volume_diff_ml_mean",
    "status",
    "notes",
]

LEADERBOARD_METRICS = {
    "dice_mean": ("dice", "mean"),
    "dice_median": ("dice", "median"),
    "lesion_f1_mean": ("lesion_f1", "mean"),
    "lesion_recall_mean": ("lesion_recall", "mean"),
    "surface_dice_3mm_mean": ("surface_dice_3mm", "mean"),
    "hd95_mm_mean": ("hd95_mm", "mean"),
    "abs_volume_diff_ml_mean": ("abs_volume_diff_ml", "mean"),
}

MD_COLUMNS = [
    "Method",
    "Modalities",
    "n",
    "Evaluator (n scored)",
    "Dice mean",
    "Dice median",
    "Lesion F1",
    "Lesion recall",
    "Surface Dice 3mm",
    "HD95 mm",
    "AVD mL",
    "Notes",
]

MISSING = "—"

# Rows whose meta.json predates the provenance key. They are not silently
# treated as equivalent to a declared evaluator: they read as "unspecified".
EVALUATOR_UNSPECIFIED = "unspecified"


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_path(path_text, root):
    path = os.path.expanduser(path_text)
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(root, path))


def read_json(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, number_types):
        return False
    return not math.isnan(float(value)) and not math.isinf(float(value))


def optional_number(value):
    if is_finite_number(value):
        return value
    return None


def metric_index(report):
    indexed = {}
    overall = report.get("overall", [])
    if not isinstance(overall, list):
        return indexed

    for item in overall:
        if not isinstance(item, dict):
            continue
        metric = item.get("metric")
        if isinstance(metric, string_types):
            indexed[metric] = item
    return indexed


def metric_value(metrics, metric, field):
    return optional_number(metrics.get(metric, {}).get(field))


def joined_modalities(value):
    if isinstance(value, list):
        return "+".join(text_value(item) for item in value)
    if value is None:
        return ""
    return text_value(value)


def text_value(value, fallback=""):
    if value is None:
        return fallback
    if isinstance(value, string_types):
        return value
    return str(value)


def integer_or_blank(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return ""


def scored_case_count(metrics, report):
    """Cases that actually contributed to the headline Dice number.

    This is where the two empty-case conventions become visible: the nnU-Net
    summary adapter emits n_valid 1448 on the same 1450-case cross-validation
    where the repo harness emits 1450, because it drops cases that are empty in
    both prediction and reference instead of scoring them as 1.0.
    """
    n_valid = integer_or_blank(metrics.get("dice", {}).get("n_valid"))
    if n_valid != "":
        return n_valid
    return integer_or_blank(report.get("n_cases"))


def provenance_cell(row):
    n_scored = row.get("n_scored")
    if n_scored == "" or n_scored is None:
        return md_cell(row.get("evaluator"))
    return "{0} (n={1})".format(md_cell(row.get("evaluator")), n_scored)


def build_row(report_dir, report_dir_name, meta, report):
    metrics = metric_index(report)
    method = text_value(meta.get("method"), report_dir_name)

    row = {
        "eval_set": text_value(meta.get("eval_set"), "unknown"),
        "method": method,
        "display_name": text_value(meta.get("display_name"), method),
        "model_family": text_value(meta.get("model_family")),
        "input_modalities": joined_modalities(meta.get("input_modalities")),
        "evaluator": text_value(meta.get("evaluator"), EVALUATOR_UNSPECIFIED),
        "empty_case_convention": text_value(meta.get("empty_case_convention"), EVALUATOR_UNSPECIFIED),
        "n_cases": integer_or_blank(report.get("n_cases")),
        "n_scored": scored_case_count(metrics, report),
        "status": text_value(meta.get("status")),
        "notes": text_value(meta.get("notes")),
        "_report_dir": report_dir_name,
    }

    for column, metric_spec in LEADERBOARD_METRICS.items():
        metric, field = metric_spec
        row[column] = metric_value(metrics, metric, field)

    return row


def discover_rows(reports_dir):
    """Return (rows, undeclared) where undeclared lists scored-but-invisible dirs.

    A directory carrying a report.json but no meta.json is collected rather than
    raised on immediately, so that one run names every offender instead of
    stopping at the first.
    """
    rows = []
    undeclared = []
    if not os.path.exists(reports_dir):
        raise IOError("reports directory does not exist: {0}".format(reports_dir))
    if not os.path.isdir(reports_dir):
        raise IOError("reports path is not a directory: {0}".format(reports_dir))

    report_dir_names = sorted(
        name for name in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, name))
    )
    for report_dir_name in report_dir_names:
        report_dir = os.path.join(reports_dir, report_dir_name)
        report_path = os.path.join(report_dir, "report.json")
        meta_path = os.path.join(report_dir, "meta.json")

        if not os.path.exists(report_path):
            # Nothing the leaderboard schema can read. Note this guard is
            # narrower than it looks: dbl_fold0/, msl_fold0/ and
            # msl_tversky_fold0/ do hold scored results, but in a bespoke
            # summary.json rather than a report.json, so they are still
            # invisible here and are NOT caught by the undeclared check below.
            # Closing that gap needs a report.json adapter for those runs, not
            # a filename heuristic (the *_splits/ dirs also carry a report.md
            # and are genuinely auxiliary).
            print("[skip] {0}: no report.json".format(report_dir_name))
            continue
        if not os.path.exists(meta_path):
            undeclared.append(report_dir_name)
            print(
                "[error] {0}: has report.json but no meta.json, so its result "
                "cannot appear on the leaderboard".format(report_dir_name)
            )
            continue

        meta = read_json(meta_path)
        report = read_json(report_path)
        if not isinstance(meta, dict):
            print("[skip] {0}: meta.json is not an object".format(report_dir_name))
            continue
        if not isinstance(report, dict):
            print("[skip] {0}: report.json is not an object".format(report_dir_name))
            continue

        row = build_row(report_dir, report_dir_name, meta, report)
        rows.append(row)
        if row["evaluator"] == EVALUATOR_UNSPECIFIED:
            print(
                "[warn] {0}: meta.json declares no evaluator; its row is not "
                "comparable like-for-like".format(report_dir_name)
            )
        print(
            "[ok] {0}: method={1} eval_set={2} n_cases={3} dice_mean={4} evaluator={5}".format(
                report_dir_name,
                row["method"],
                row["eval_set"],
                row["n_cases"],
                format_md_float(row.get("dice_mean")),
                row["evaluator"],
            )
        )

    rows.sort(key=lambda row: (text_value(row["eval_set"]), text_value(row["method"]), text_value(row["_report_dir"])))
    return rows, undeclared


def format_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and optional_number(value) is None:
        return ""
    return value


def csv_cell(value):
    value = format_csv_value(value)
    if value is None:
        text = ""
    elif isinstance(value, float):
        text = repr(value)
    else:
        text = text_value(value)

    if any(char in text for char in [",", '"', "\n", "\r"]):
        text = '"' + text.replace('"', '""') + '"'
    return text


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def write_csv(rows, path, undeclared=()):
    ensure_dir(os.path.dirname(path))
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        if undeclared:
            # The markdown banner is invisible to anything that reads the CSV,
            # which is the machine-readable half of the artefact pair. Emitting
            # the marker here too means an incomplete leaderboard cannot be
            # loaded as if it were complete: a reader that honours '#' comments
            # sees the warning, and a reader that does not takes this line as
            # the header and loses every column name instead of quietly
            # returning a table that looks whole.
            f.write(
                "# INCOMPLETE: generated with --allow-missing-meta; "
                "scored report directories omitted for lack of meta.json: {0}\n".format(
                    " ".join(undeclared)
                )
            )
        f.write(",".join(CSV_COLUMNS) + "\n")
        for row in rows:
            f.write(",".join(csv_cell(row.get(column, "")) for column in CSV_COLUMNS) + "\n")


def format_md_float(value):
    number = optional_number(value)
    if number is None:
        return MISSING
    return "{0:.3f}".format(number)


def md_cell(value):
    text = text_value(value, MISSING)
    if text == "":
        text = MISSING
    return text.replace("|", "\\|").replace("\n", " ")


def eval_set_sort_key(eval_set):
    return (0 if eval_set == "soop_bench" else 1, eval_set)


def row_sort_key(row):
    dice = optional_number(row.get("dice_mean"))
    if dice is None:
        return (1, 0.0, text_value(row["display_name"]), text_value(row["_report_dir"]))
    return (0, -float(dice), text_value(row["display_name"]), text_value(row["_report_dir"]))


def markdown_table(rows):
    lines = [
        "| " + " | ".join(MD_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in MD_COLUMNS) + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_cell(row.get("display_name")),
                    md_cell(row.get("input_modalities")),
                    md_cell(row.get("n_cases")),
                    provenance_cell(row),
                    format_md_float(row.get("dice_mean")),
                    format_md_float(row.get("dice_median")),
                    format_md_float(row.get("lesion_f1_mean")),
                    format_md_float(row.get("lesion_recall_mean")),
                    format_md_float(row.get("surface_dice_3mm_mean")),
                    format_md_float(row.get("hd95_mm_mean")),
                    format_md_float(row.get("abs_volume_diff_ml_mean")),
                    md_cell(row.get("notes")),
                ]
            )
            + " |"
        )
    return lines


def write_markdown(rows, path, undeclared=()):
    ensure_dir(os.path.dirname(path))
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    by_eval_set = defaultdict(list)
    for row in rows:
        by_eval_set[text_value(row["eval_set"])].append(row)

    lines = [
        "# Model leaderboard",
        "",
        "generated {0} by compare_models.py".format(generated_at),
        "",
    ]

    if undeclared:
        # Written into the artefact, not just the console, so a leaderboard
        # produced with --allow-missing-meta can never be mistaken for complete.
        lines.extend(
            [
                "> **INCOMPLETE.** Generated with `--allow-missing-meta`. These report "
                "directories hold a scored report.json but no meta.json, so their results "
                "are missing from every table below: {0}.".format(
                    ", ".join("`{0}`".format(name) for name in undeclared)
                ),
                "",
            ]
        )

    for eval_set in sorted(by_eval_set, key=eval_set_sort_key):
        lines.append("## eval set: {0}".format(eval_set))
        lines.append("")
        lines.extend(markdown_table(sorted(by_eval_set[eval_set], key=row_sort_key)))
        lines.append("")

    lines.append("Note: lower is better for HD95/AVD; higher is better for the remaining metrics.")
    lines.append("")
    lines.append(
        "Provenance: the `Evaluator (n scored)` column names the code that produced the row "
        "and how many cases entered the Dice mean. Rows from different evaluators use "
        "different empty-case conventions (`repo_harness` scores empty-on-both as 1.0, "
        "`nnunet_summary` drops those cases) and must not be compared at the third decimal."
    )
    lines.append("")

    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="baselines/reports")
    parser.add_argument("--out-dir", default="baselines/compare")
    parser.add_argument(
        "--allow-missing-meta",
        action="store_true",
        help=(
            "emit a leaderboard even though some report.json has no meta.json; "
            "the omitted directories are stamped into leaderboard.md so the "
            "artefact still advertises that it is incomplete"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = repo_root()
    reports_dir = resolve_path(args.reports_dir, root)
    out_dir = resolve_path(args.out_dir, root)

    rows, undeclared = discover_rows(reports_dir)
    if undeclared and not args.allow_missing_meta:
        raise SystemExit(
            "[error] {0} report director{1} scored but undeclared: {2}. Write a "
            "meta.json for each (or rerun with --allow-missing-meta to accept an "
            "explicitly incomplete leaderboard).".format(
                len(undeclared),
                "y is" if len(undeclared) == 1 else "ies are",
                ", ".join(undeclared),
            )
        )

    csv_path = os.path.join(out_dir, "leaderboard.csv")
    md_path = os.path.join(out_dir, "leaderboard.md")
    write_csv(rows, csv_path, undeclared)
    write_markdown(rows, md_path, undeclared)

    print("[write] {0}".format(csv_path))
    print("[write] {0}".format(md_path))


if __name__ == "__main__":
    main()
