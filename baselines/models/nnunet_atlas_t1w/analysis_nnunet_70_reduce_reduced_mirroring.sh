#!/bin/bash
#SBATCH --job-name=reduce_reduced_mirroring
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -p normal,owners
#
# Decide the mirroring question with one table.
#
# Six arms, all fold 0 of Dataset507, same 281 cases, same weights, same grid,
# scored with the organizers' own code. The ONLY variable is which spatial axes
# mirroring TTA is allowed to flip:
#
#   none          shipped -- 1 forward pass per fold
#   (0,)          left-right
#   (1,)          posterior-anterior
#   (2,)          inferior-superior
#   (0,1)         2 axes, 4 passes
#   (0,1,2)       full TTA, 8 passes -- over the CPU budget at 5 folds
#
# The decision rule, fixed BEFORE the numbers are read, because this project has
# been burned by post-hoc rules (see the threshold-0.54 retraction in
# docs/models/status.md):
#
#   Adopt the cheapest subset that (a) recovers at least HALF of full TTA's Dice
#   gain over no-TTA (i.e. >= +0.0071), (b) does not lose on any other of the
#   five ranked metrics beyond noise (paired |t| < 2), and (c) fits the CPU
#   budget with >= 2x headroom at 5 folds. Otherwise ship no-TTA unchanged.

set -euo pipefail
REPO=/scratch/users/sciget/isles26challenge
cd "$REPO"; mkdir -p "$REPO/logs"

if command -v module >/dev/null 2>&1; then
  module purge
  [[ -n "${GROUP_HOME:-}" && -d "$GROUP_HOME/modules" ]] && module use "$GROUP_HOME/modules"
  module load devel; module load python/3.12.1
fi
export LD_LIBRARY_PATH="/share/software/user/open/python/3.12.1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO"

REPORT_DIR="$REPO/baselines/reports/reduced_mirroring"
mkdir -p "$REPORT_DIR"

# Reduce each variant's shards into a per_case.csv first.
for TAG in ax0 ax1 ax2 ax01; do
  OUT_DIR="$REPO/work/official_eval/scored_mirror_${TAG}_fold0"
  N=$(find "$OUT_DIR" -maxdepth 1 -name 'per_case_shard_*.json' 2>/dev/null | wc -l)
  if [[ "$N" -ne 6 ]]; then
    echo "[error] expected 6 shards for $TAG, found $N" >&2; exit 2
  fi
  "$REPO/.venv-official/bin/python" "$REPO/eval/run_official_eval.py" reduce \
    --out-dir "$OUT_DIR" \
    --title "ISLES'26 official metrics -- TopK10 fold 0, mirror axes ${TAG}" \
    --reference-threshold 0.50
done

REPO="$REPO" REPORT_DIR="$REPORT_DIR" "$REPO/.venv-eval/bin/python" - <<'PYEOF'
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ["REPO"]); REPORT = Path(os.environ["REPORT_DIR"])
OP = "thr0.35_k20"          # the shipped operating point
METRICS = {"dice": "higher", "abs_volume_diff_ml": "lower",
           "abs_lesion_count_diff": "lower", "lesion_f1": "higher",
           "pr_auc_flatten_if_empty": "higher"}

ARMS = {
    "none (shipped)":  REPO / "work/official_eval/scored_noTTA_fold0/per_case.csv",
    "(0,) L-R":        REPO / "work/official_eval/scored_mirror_ax0_fold0/per_case.csv",
    "(1,) P-A":        REPO / "work/official_eval/scored_mirror_ax1_fold0/per_case.csv",
    "(2,) I-S":        REPO / "work/official_eval/scored_mirror_ax2_fold0/per_case.csv",
    "(0,1)":           REPO / "work/official_eval/scored_mirror_ax01_fold0/per_case.csv",
    "(0,1,2) full":    REPO / "baselines/reports/official_metrics_d507_topk10/per_case.csv",
}
PASSES = {"none (shipped)": 1, "(0,) L-R": 2, "(1,) P-A": 2, "(2,) I-S": 2,
          "(0,1)": 4, "(0,1,2) full": 8}

frames = {}
for name, path in ARMS.items():
    df = pd.read_csv(path)
    frames[name] = df[df.config == OP].set_index("subject_id")

cases = sorted(set.intersection(*(set(f.index) for f in frames.values())))
assert len(cases) == 281, f"expected 281 shared cases, got {len(cases)}"
print(f"[info] {len(cases)} cases shared by all {len(frames)} arms")


def col(name, metric):
    return frames[name][metric].reindex(cases).to_numpy(float)


def paired_t(a, b):
    d = a - b
    se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    return (float(np.mean(d)), float(np.mean(d) / se) if se > 0 else 0.0)


base, full = "none (shipped)", "(0,1,2) full"
full_dice_gain = float(np.mean(col(full, "dice")) - np.mean(col(base, "dice")))

lines = [
    "# Reduced mirroring TTA -- can a cheaper subset fit the CPU budget?",
    "",
    "Fold 0 of Dataset507 (center-grouped), n = 281, same weights, same grid,",
    f"scored at the shipped operating point (`{OP}`). The only variable is which",
    "spatial axes mirroring may flip. Axis 0 is left-right, 1 posterior-anterior,",
    "2 inferior-superior (volumes are RAS-canonical, transpose_forward [0,1,2]).",
    "",
    f"Full TTA is worth **{full_dice_gain:+.4f} Dice** over no-TTA here. Cost at 5",
    "folds scales with the pass count: ~2.0 min/case at 1 pass, ~12.9 min at 8,",
    "against a ~10 minute budget.",
    "",
    "| mirror axes | passes | est. min/case @5 folds | Dice | AVD mL | count diff | lesion-F1 | PR-AUC |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for name in ARMS:
    est = 2.0 * PASSES[name]
    lines.append(
        f"| {name} | {PASSES[name]} | {est:.1f} | "
        + " | ".join(f"{np.mean(col(name, m)):.4f}" for m in METRICS)
        + " |"
    )

lines += ["", f"## Paired deltas vs no-TTA (same 281 cases, at `{OP}`)", "",
          "| mirror axes | dDice | t | dAVD | t | dCount | t | dRQ | t | dPR-AUC | t |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
summary = {}
for name in ARMS:
    if name == base:
        continue
    cells, rec = [], {}
    for m in METRICS:
        d, t = paired_t(col(name, m), col(base, m))
        cells += [f"{d:+.4f}", f"{t:+.2f}"]
        rec[m] = {"delta": d, "t": t}
    summary[name] = rec
    lines.append(f"| {name} | " + " | ".join(cells) + " |")

# --- the pre-registered decision rule ---------------------------------------
lines += ["", "## Decision (rule fixed before the numbers were read)", "",
          "Adopt the cheapest subset that recovers >= half of full TTA's Dice gain",
          f"(>= {0.5 * full_dice_gain:+.4f}), loses nothing else beyond noise (paired |t| < 2",
          "against no-TTA on the other four metrics), and leaves >= 2x CPU headroom",
          "at 5 folds (<= 5.0 min/case). Otherwise keep no-TTA.", ""]

winner = None
for name in ["(0,) L-R", "(1,) P-A", "(2,) I-S", "(0,1)"]:
    rec = summary[name]
    est = 2.0 * PASSES[name]
    gain_ok = rec["dice"]["delta"] >= 0.5 * full_dice_gain
    budget_ok = est <= 5.0
    regress = [m for m, d in METRICS.items()
               if m != "dice" and abs(rec[m]["t"]) >= 2.0
               and ((rec[m]["delta"] < 0) if d == "higher" else (rec[m]["delta"] > 0))]
    verdict = ("ADOPT" if (gain_ok and budget_ok and not regress) else "reject")
    why = []
    if not gain_ok:
        why.append(f"Dice gain {rec['dice']['delta']:+.4f} < {0.5 * full_dice_gain:+.4f}")
    if not budget_ok:
        why.append(f"{est:.1f} min/case exceeds the 5.0 min ceiling")
    if regress:
        why.append("significant regression on " + ", ".join(regress))
    lines.append(f"- **{name}** ({est:.1f} min/case): {verdict}"
                 + (f" -- {'; '.join(why)}" if why else ""))
    if verdict == "ADOPT" and winner is None:
        winner = name

lines += [""]
if winner:
    lines += [f"**Result: adopt mirror axes {winner}.** Set `ISLES26_MIRROR_AXES` to the",
              "corresponding value and `ISLES26_DISABLE_TTA=0` in the Dockerfile/inference",
              "defaults, then re-verify the CPU budget on a real case before building."]
else:
    lines += ["**Result: no subset qualifies. The container ships unchanged, 5 folds",
              "with mirroring off.** Reduced mirroring is now measured and closed."]

(REPORT / "README.md").write_text("\n".join(lines) + "\n")
(REPORT / "summary.json").write_text(json.dumps(
    {"operating_point": OP, "n": len(cases), "full_tta_dice_gain": full_dice_gain,
     "passes": PASSES, "deltas_vs_no_tta": summary, "winner": winner}, indent=2))
print("\n".join(lines))
PYEOF

echo "[done] -> $REPORT_DIR/README.md"
