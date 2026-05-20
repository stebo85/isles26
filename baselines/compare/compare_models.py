#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a model-comparison leaderboard from per-method reports."""

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
    "n_cases",
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


def build_row(report_dir, report_dir_name, meta, report):
    metrics = metric_index(report)
    method = text_value(meta.get("method"), report_dir_name)
    n_cases = report.get("n_cases")

    row = {
        "eval_set": text_value(meta.get("eval_set"), "unknown"),
        "method": method,
        "display_name": text_value(meta.get("display_name"), method),
        "model_family": text_value(meta.get("model_family")),
        "input_modalities": joined_modalities(meta.get("input_modalities")),
        "n_cases": n_cases if isinstance(n_cases, int) and not isinstance(n_cases, bool) else "",
        "status": text_value(meta.get("status")),
        "notes": text_value(meta.get("notes")),
        "_report_dir": report_dir_name,
    }

    for column, metric_spec in LEADERBOARD_METRICS.items():
        metric, field = metric_spec
        row[column] = metric_value(metrics, metric, field)

    return row


def discover_rows(reports_dir):
    rows = []
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

        if not os.path.exists(meta_path):
            print("[skip] {0}: missing meta.json".format(report_dir_name))
            continue
        if not os.path.exists(report_path):
            print("[skip] {0}: missing report.json".format(report_dir_name))
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
        print(
            "[ok] {0}: method={1} eval_set={2} n_cases={3} dice_mean={4}".format(
                report_dir_name,
                row["method"],
                row["eval_set"],
                row["n_cases"],
                format_md_float(row.get("dice_mean")),
            )
        )

    rows.sort(key=lambda row: (text_value(row["eval_set"]), text_value(row["method"]), text_value(row["_report_dir"])))
    return rows


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


def write_csv(rows, path):
    ensure_dir(os.path.dirname(path))
    with io.open(path, "w", encoding="utf-8", newline="") as f:
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


def write_markdown(rows, path):
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

    for eval_set in sorted(by_eval_set, key=eval_set_sort_key):
        lines.append("## eval set: {0}".format(eval_set))
        lines.append("")
        lines.extend(markdown_table(sorted(by_eval_set[eval_set], key=row_sort_key)))
        lines.append("")

    lines.append("Note: lower is better for HD95/AVD; higher is better for the remaining metrics.")
    lines.append("")

    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="baselines/reports")
    parser.add_argument("--out-dir", default="baselines/compare")
    return parser.parse_args()


def main():
    args = parse_args()
    root = repo_root()
    reports_dir = resolve_path(args.reports_dir, root)
    out_dir = resolve_path(args.out_dir, root)

    rows = discover_rows(reports_dir)

    csv_path = os.path.join(out_dir, "leaderboard.csv")
    md_path = os.path.join(out_dir, "leaderboard.md")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print("[write] {0}".format(csv_path))
    print("[write] {0}".format(md_path))


if __name__ == "__main__":
    main()
