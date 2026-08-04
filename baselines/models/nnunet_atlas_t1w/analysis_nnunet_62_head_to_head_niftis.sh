#!/bin/bash
#SBATCH --job-name=head_to_head_nii
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH -p normal
#
# Head-to-head NIfTIs: the SHIPPED model vs a stock-nnU-Net baseline vs the DA5
# arm we chose not to promote, on identical cases, in one directory that opens
# straight in fsleyes/ITK-SNAP.
#
# Models (all three predict each case with the fold that HELD IT OUT, so every
# number below is out-of-fold):
#   ours     Dataset507 (center-grouped folds), nnUNetTrainerDiceTopK10Loss,
#            1000 epochs -- the weights in the submission container.
#   baseline Dataset502 (random folds), nnUNetTrainer, 1000 epochs -- stock
#            nnU-Net v2 3d_fullres out of the box, the standard strong baseline
#            for this task. NOTE its folds are center-leaking (52 of 55 centers
#            appear on both sides of every fold), so a case that is genuinely
#            out-of-center for `ours` is merely out-of-subject for `baseline`.
#   ours_same_split
#            Dataset502, nnUNetTrainerDiceTopK10Loss -- our loss on the
#            BASELINE'S OWN split. This is the only arm that isolates the
#            network: same folds, same cases, same leakage. `ours` vs
#            `baseline` conflates the model with the difficulty of the split.
#   da5      Dataset507, nnUNetTrainerDA5TopK10 -- heavy augmentation; measured
#            +0.0111 Dice over ours but held back by the pre-registered gate.
#
# Every model is thresholded with the SAME shipped operating point (p >= 0.35,
# drop connected components < 20 voxels at 26-connectivity), so the only thing
# that differs between the masks is the network. Probability maps are emitted
# too, because PR-AUC over the soft map is one of the five ranked metrics.
#
# Case selection is fixed and NOT cherry-picked: the nine QC cases that span the
# whole ground-truth lesion-volume range (0.0 to 388 mL, from
# baselines/reports/qc_test_predictions/FINDING.md) plus a seeded stratified
# draw of two more per size bin.

set -euo pipefail

REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"
mkdir -p "$REPO/logs"

if command -v module >/dev/null 2>&1; then
  module purge
  if [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]]; then
    module use "$GROUP_HOME/modules"
  fi
  module load devel
  module load python/3.12.1
fi

export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

PY="$REPO/.venv-eval/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[error] missing .venv-eval" >&2
  exit 2
fi

OUT_DIR="$REPO/work/head_to_head"
REPORT_DIR="$REPO/baselines/reports/head_to_head"
mkdir -p "$OUT_DIR" "$REPORT_DIR"

THRESHOLD="${ISLES26_THRESHOLD:-0.35}"
MIN_CC="${ISLES26_MIN_CC_VOXELS:-20}"
EXTRA_PER_BIN="${EXTRA_PER_BIN:-2}"

"$PY" - "$REPO" "$OUT_DIR" "$REPORT_DIR" "$THRESHOLD" "$MIN_CC" "$EXTRA_PER_BIN" <<'PYEOF'
from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

repo, out_dir, report_dir, threshold, min_cc, extra_per_bin = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    Path(sys.argv[3]),
    float(sys.argv[4]),
    int(sys.argv[5]),
    int(sys.argv[6]),
)

RESULTS = repo / "work/nnunet/nnUNet_results"
RAW = repo / "work/nnunet/nnUNet_raw"
HELPER = repo / "baselines/models/nnunet_atlas_t1w/nnunet_helpers/npz_lesion_prob_to_nii.py"
PYBIN = repo / ".venv-eval/bin/python"

D507 = "Dataset507_ATLASR30_CORRECTED"
D502 = "Dataset502_ATLASR30"

MODELS = {
    "ours": RESULTS / D507 / "nnUNetTrainerDiceTopK10Loss__nnUNetPlans__3d_fullres",
    "baseline": RESULTS / D502 / "nnUNetTrainer__nnUNetPlans__3d_fullres",
    "ours_same_split": RESULTS / D502 / "nnUNetTrainerDiceTopK10Loss__nnUNetPlans__3d_fullres",
    "da5": RESULTS / D507 / "nnUNetTrainerDA5TopK10__nnUNetPlans__3d_fullres",
}

# The nine size-spanning QC cases, already characterised in
# baselines/reports/qc_test_predictions/FINDING.md.
QC_CASES = [
    "ATLAS_0809",  # lesion-free
    "ATLAS_0310",
    "ATLAS_0712",
    "ATLAS_0800",
    "ATLAS_1247",
    "ATLAS_0635",
    "ATLAS_1138",
    "ATLAS_1372",
    "ATLAS_0104",  # 388 mL
]


def load_manifest(dataset: str) -> dict[str, dict[str, str]]:
    path = RAW / dataset / "conversion_manifest.csv"
    with path.open(newline="") as handle:
        return {row["nnunet_id"]: row for row in csv.DictReader(handle)}


manifest507 = load_manifest(D507)
manifest502 = load_manifest(D502)

# The two conversions must agree on what each case id MEANS, or a per-case
# comparison silently pairs different subjects.
disagreeing = [
    case
    for case in set(manifest507) & set(manifest502)
    if manifest507[case]["t1_source_path"] != manifest502[case]["t1_source_path"]
]
if disagreeing:
    raise SystemExit(f"[error] {len(disagreeing)} case ids map to different subjects across datasets")


def fold_index(model_dir: Path) -> dict[str, int]:
    """case id -> the fold whose validation split held it out."""
    index: dict[str, int] = {}
    for fold in range(5):
        val = model_dir / f"fold_{fold}" / "validation"
        if not val.is_dir():
            raise SystemExit(f"[error] missing {val}")
        for npz in val.glob("*.npz"):
            index[npz.stem] = fold
    return index


folds = {name: fold_index(path) for name, path in MODELS.items()}

# Stratified extras, deterministic: sorted pool, fixed seed.
pool_by_bin: dict[str, list[str]] = {}
for case, row in sorted(manifest507.items()):
    if case in QC_CASES:
        continue
    if not all(case in folds[name] for name in MODELS):
        continue
    pool_by_bin.setdefault(row["size_bin"], []).append(case)

rng = random.Random(20260803)
cases = list(QC_CASES)
for size_bin in sorted(pool_by_bin):
    pool = pool_by_bin[size_bin]
    cases.extend(rng.sample(pool, min(extra_per_bin, len(pool))))

missing = [c for c in cases if not all(c in folds[name] for name in MODELS)]
if missing:
    raise SystemExit(f"[error] cases not out-of-fold in every model: {missing}")
print(f"[info] {len(cases)} cases: {', '.join(cases)}", flush=True)


def stage_probabilities(model_name: str, model_dir: Path, case_ids: list[str], dest: Path) -> None:
    """Convert each case's softmax NPZ to a lesion probability NIfTI.

    Reuses npz_lesion_prob_to_nii.py rather than re-deriving the axis
    permutation: that helper carries the hard-fail guards this project earned
    the hard way, and a silent transpose produces a plausible-looking mask.
    Cases are grouped by fold because the NPZ axis order is a per-fold property,
    and the helper's batch-consensus fallback is what rescues the lesion-free
    cases whose own segmentation cannot disambiguate.
    """
    dest.mkdir(parents=True, exist_ok=True)
    by_fold: dict[int, list[str]] = {}
    for case in case_ids:
        by_fold.setdefault(folds[model_name][case], []).append(case)

    for fold, fold_cases in sorted(by_fold.items()):
        val = model_dir / f"fold_{fold}" / "validation"
        with tempfile.TemporaryDirectory(dir=str(dest)) as tmp:
            tmp_dir = Path(tmp)
            for case in fold_cases:
                for suffix in (".npz", ".nii.gz"):
                    (tmp_dir / f"{case}{suffix}").symlink_to(val / f"{case}{suffix}")
            subprocess.run(
                [
                    str(PYBIN),
                    str(HELPER),
                    "--npz-dir",
                    str(tmp_dir),
                    "--seg-dir",
                    str(tmp_dir),
                    "--ref-dir",
                    str(tmp_dir),
                    "--out-dir",
                    str(tmp_dir / "_out"),
                    "--lesion-channel",
                    "1",
                ],
                check=True,
            )
            report = json.loads((tmp_dir / "_out" / "_convert_report.json").read_text())
            if report["failed"]:
                raise SystemExit(f"[error] {model_name} fold {fold} conversions failed: {report['failed']}")
            for case in fold_cases:
                shutil.move(str(tmp_dir / "_out" / f"{case}.nii.gz"), str(dest / f"{case}.nii.gz"))
        print(f"[ok] {model_name} fold {fold}: {len(fold_cases)} probability maps", flush=True)


staging = out_dir / "_staged_probs"
for name, path in MODELS.items():
    stage_probabilities(name, path, cases, staging / name)


def postprocess(prob: np.ndarray) -> np.ndarray:
    """Shipped operating point: threshold, then drop small components."""
    mask = prob >= threshold
    if min_cc > 0 and mask.any():
        labels, n = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=int))
        if n:
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            keep = np.flatnonzero(counts >= min_cc)
            mask = np.isin(labels, keep)
    return mask.astype(np.uint8)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Official empty convention: an empty GT scores 1.0 only for an empty pred."""
    denom = int(pred.sum()) + int(gt.sum())
    if denom == 0:
        return 1.0
    return 2.0 * float(np.logical_and(pred, gt).sum()) / denom


def n_components(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    return int(ndi.label(mask, structure=np.ones((3, 3, 3), dtype=int))[1])


rows: list[dict[str, object]] = []
panels: dict[str, dict[str, np.ndarray]] = {}

for case in cases:
    t1_src = RAW / D507 / "imagesTr" / f"{case}_0000.nii.gz"
    gt_src = RAW / D507 / "labelsTr" / f"{case}.nii.gz"
    t1_img = nib.load(str(t1_src))
    gt_img = nib.load(str(gt_src))
    gt = np.asanyarray(gt_img.dataobj).astype(np.uint8)
    voxel_ml = float(np.prod(t1_img.header.get_zooms()[:3])) / 1000.0

    shutil.copy2(t1_src, out_dir / f"{case}_t1w.nii.gz")
    shutil.copy2(gt_src, out_dir / f"{case}_gt.nii.gz")

    case_panel = {"t1": np.asanyarray(t1_img.dataobj).astype(np.float32), "gt": gt}
    row: dict[str, object] = {
        "case": case,
        "site": manifest507[case]["site"],
        "size_bin": manifest507[case]["size_bin"],
        "gt_ml": round(float(gt.sum()) * voxel_ml, 3),
        "gt_components": n_components(gt),
    }

    for name in MODELS:
        prob_img = nib.load(str(staging / name / f"{case}.nii.gz"))
        if prob_img.shape != t1_img.shape or not np.allclose(prob_img.affine, t1_img.affine, atol=1e-4):
            raise SystemExit(f"[error] {case}/{name}: geometry does not match the T1w")
        prob = np.asanyarray(prob_img.dataobj).astype(np.float32)
        pred = postprocess(prob)

        header = t1_img.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(pred, t1_img.affine, header=header), str(out_dir / f"{case}_{name}_pred.nii.gz"))
        header = t1_img.header.copy()
        header.set_data_dtype(np.float32)
        header.set_slope_inter(1.0, 0.0)
        nib.save(nib.Nifti1Image(prob, t1_img.affine, header=header), str(out_dir / f"{case}_{name}_prob.nii.gz"))

        row[f"{name}_fold"] = folds[name][case]
        row[f"{name}_dice"] = round(dice(pred, gt), 4)
        row[f"{name}_ml"] = round(float(pred.sum()) * voxel_ml, 3)
        row[f"{name}_components"] = n_components(pred)
        case_panel[name] = pred

    rows.append(row)
    panels[case] = case_panel
    print(
        f"[ok] {case} gt={row['gt_ml']}mL "
        + " ".join(f"{n}={row[f'{n}_dice']}" for n in MODELS),
        flush=True,
    )

shutil.rmtree(staging)

fields = list(rows[0].keys())
with (out_dir / "comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

summary = {
    name: {
        "mean_dice": round(float(np.mean([r[f"{name}_dice"] for r in rows])), 4),
        "median_dice": round(float(np.median([r[f"{name}_dice"] for r in rows])), 4),
        "wins_vs_baseline": int(sum(r[f"{name}_dice"] > r["baseline_dice"] for r in rows)),
    }
    for name in MODELS
}
(out_dir / "summary.json").write_text(json.dumps({"n": len(rows), "threshold": threshold,
                                                  "min_cc_voxels": min_cc, "models": summary}, indent=2) + "\n")

lines = [
    "# Head-to-head: shipped model vs stock nnU-Net baseline vs DA5",
    "",
    f"n = {len(rows)} cases, all out-of-fold for all three models. Operating point for",
    f"every model: p >= {threshold}, components < {min_cc} voxels dropped (26-connectivity).",
    "",
    "`ours` is evaluated on center-held-out folds and `baseline` on center-leaking",
    "folds, so those two columns are NOT a like-for-like model comparison.",
    "`ours_same_split` is: it is our loss trained on the baseline's own split.",
    "",
    "| case | site | GT mL | ours Dice | baseline Dice | ours-same-split Dice | DA5 Dice | GT mL | ours mL | baseline mL |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['case']} | {r['site']} | {r['gt_ml']:.2f} | {r['ours_dice']:.3f} | "
        f"{r['baseline_dice']:.3f} | {r['ours_same_split_dice']:.3f} | {r['da5_dice']:.3f} | "
        f"{r['gt_ml']:.2f} | {r['ours_ml']:.2f} | {r['baseline_ml']:.2f} |"
    )
lines += [
    "",
    "| model | mean Dice | median Dice | cases better than baseline |",
    "|---|---:|---:|---:|",
]
for name in MODELS:
    s = summary[name]
    lines.append(f"| {name} | {s['mean_dice']:.4f} | {s['median_dice']:.4f} | {s['wins_vs_baseline']}/{len(rows)} |")
lines += [
    "",
    "This n is far too small to rank models -- the corpus-level numbers in",
    "`work/official_eval/` do that. It exists so the masks can be looked at.",
    "",
    "## Viewing",
    "",
    "```",
    "fsleyes work/head_to_head/ATLAS_1138_t1w.nii.gz \\",
    "  work/head_to_head/ATLAS_1138_gt.nii.gz -cm green -a 40 \\",
    "  work/head_to_head/ATLAS_1138_ours_pred.nii.gz -cm red -a 40 \\",
    "  work/head_to_head/ATLAS_1138_baseline_pred.nii.gz -cm blue -a 40",
    "```",
]
(out_dir / "README.md").write_text("\n".join(lines) + "\n")
shutil.copy2(out_dir / "README.md", report_dir / "README.md")
shutil.copy2(out_dir / "comparison.csv", report_dir / "comparison.csv")
shutil.copy2(out_dir / "summary.json", report_dir / "summary.json")


def contour_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.rot90(np.take(volume, index, axis=axis))


# One row per case, at the slice through the ground-truth centre of mass, so the
# panel cannot be flattered by picking the most favourable slice.
fig, axes = plt.subplots(len(cases), 5, figsize=(16, 3.1 * len(cases)))
for r, case in enumerate(cases):
    panel = panels[case]
    ref = panel["gt"] if panel["gt"].any() else panel["ours"]
    if ref.any():
        com = ndi.center_of_mass(ref)
        areas = [np.take(ref, int(round(com[a])), axis=a).sum() for a in range(3)]
        axis = int(np.argmax(areas))
        index = int(round(com[axis]))
    else:
        axis, index = 2, ref.shape[2] // 2
    t1 = contour_slice(panel["t1"], axis, index)
    vmax = float(np.percentile(t1, 99.5)) or 1.0
    overlays = [
        ("gt", "#22c55e"),
        ("ours", "#ef4444"),
        ("baseline", "#3b82f6"),
        ("ours_same_split", "#a855f7"),
        ("da5", "#f59e0b"),
    ]
    for c, (key, color) in enumerate(overlays):
        ax = axes[r, c] if len(cases) > 1 else axes[c]
        ax.imshow(t1, cmap="gray", vmin=0, vmax=vmax)
        sl = contour_slice(panel[key], axis, index).astype(float)
        if sl.any():
            ax.contour(sl, levels=[0.5], colors=[color], linewidths=1.2)
        if key != "gt":
            gt_sl = contour_slice(panel["gt"], axis, index).astype(float)
            if gt_sl.any():
                ax.contour(gt_sl, levels=[0.5], colors=["#22c55e"], linewidths=0.9, alpha=0.8)
        row = next(x for x in rows if x["case"] == case)
        title = key if key == "gt" else f"{key}  Dice {row[f'{key}_dice']:.3f}"
        ax.set_title(f"{case} {title}" if c == 0 else title, fontsize=9)
        ax.axis("off")
fig.suptitle(
    f"green = ground truth, red = ours (shipped), blue = stock nnU-Net, "
    f"purple = ours on the baseline's split, orange = DA5 "
    f"| p>={threshold}, minCC={min_cc}",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.99))
fig.savefig(report_dir / "head_to_head_panel.png", dpi=110)
plt.close(fig)

print(f"[done] NIfTIs in {out_dir}; panel + tables in {report_dir}")
PYEOF
