#!/bin/bash
#SBATCH --job-name=reduce_tta_ablation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -p normal
#
# Reduce the no-TTA shards and put the shipped configuration's real cost on
# paper: paired per-case deltas (no-TTA minus TTA) on the same 281 fold-0 cases,
# same weights, same grid.
#
# Also re-selects the operating point on the no-TTA probabilities. Thresholds
# fitted on TTA-smoothed maps do not automatically transfer: averaging over
# mirrors pulls probability mass off 0 and 1, so a no-TTA map is sharper and the
# same threshold sits at a different place on its own distribution.

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

OUT_DIR="$REPO/work/official_eval/scored_noTTA_fold0"
REPORT_DIR="$REPO/baselines/reports/tta_ablation"
mkdir -p "$REPORT_DIR"

N_SHARDS=$(find "$OUT_DIR" -maxdepth 1 -name 'per_case_shard_*.json' 2>/dev/null | wc -l)
if [[ "$N_SHARDS" -ne 6 ]]; then
  echo "[error] expected 6 shards in $OUT_DIR, found $N_SHARDS" >&2
  exit 2
fi

"$REPO/.venv-official/bin/python" "$REPO/eval/run_official_eval.py" reduce \
  --out-dir "$OUT_DIR" \
  --title "ISLES'26 official metrics -- TopK10 fold 0, mirroring TTA OFF" \
  --reference-threshold 0.50

# Paired comparison against the TTA control.
"$REPO/.venv-eval/bin/python" - "$OUT_DIR" "$REPO/baselines/reports/official_metrics_d507_topk10/per_case.csv" "$REPORT_DIR" <<'PYEOF'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out_dir, control_csv, report_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

no_tta = pd.read_csv(out_dir / "per_case.csv")
tta = pd.read_csv(control_csv)
tta = tta[tta.subject_id.isin(set(no_tta.subject_id))]

cases = sorted(set(no_tta.subject_id))
assert len(cases) == 281, len(cases)

METRICS = {
    "dice": "higher",
    "abs_volume_diff_ml": "lower",
    "abs_lesion_count_diff": "lower",
    "lesion_f1": "higher",
    "pr_auc_flatten_if_empty": "higher",
}

lines = [
    "# Mirroring TTA ablation -- what the shipped container actually gives up",
    "",
    f"Fold 0 of Dataset507 (center-grouped), n = {len(cases)} cases, same weights,",
    "same grid. TTA is the only variable. Positive delta = no-TTA minus TTA.",
    "",
    "## At the shipped operating point (p >= 0.35, minCC 20)",
    "",
    "| metric | TTA | no TTA | delta | paired t | better |",
    "|---|---:|---:|---:|---:|---|",
]

rows = []
for metric, direction in METRICS.items():
    a = no_tta[no_tta.config == "thr0.35_k20"].set_index("subject_id")[metric].reindex(cases).to_numpy(float)
    b = tta[tta.config == "thr0.35_k20"].set_index("subject_id")[metric].reindex(cases).to_numpy(float)
    d = a - b
    se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    t = float(np.mean(d) / se) if se > 0 else 0.0
    better = "no TTA" if ((np.mean(d) > 0) == (direction == "higher")) else "TTA"
    if abs(t) < 2.0:
        better = "tie"
    lines.append(
        f"| {metric} | {np.mean(b):.4f} | {np.mean(a):.4f} | {np.mean(d):+.4f} | {t:+.2f} | {better} |"
    )
    rows.append({"metric": metric, "tta": np.mean(b), "no_tta": np.mean(a), "delta": np.mean(d), "t": t})

# Does the operating point still sit where it was fitted?
grid = (
    no_tta.groupby("config")
    .agg(
        dice=("dice", "mean"),
        avd=("abs_volume_diff_ml", "mean"),
        count=("abs_lesion_count_diff", "mean"),
        rq=("lesion_f1", "mean"),
        pr=("pr_auc_flatten_if_empty", "mean"),
    )
    .reset_index()
)
ranks = (
    grid.rank(ascending=False)[["dice", "rq", "pr"]].sum(axis=1)
    + grid.rank(ascending=True)[["avd", "count"]].sum(axis=1)
) / 5.0
grid["mean_rank"] = ranks
grid = grid.sort_values("mean_rank")

lines += [
    "",
    "## Operating point re-selected on no-TTA probabilities (top 8 of 54)",
    "",
    "| config | Dice | AVD mL | count diff | lesion-F1 | PR-AUC | mean rank |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for _, r in grid.head(8).iterrows():
    lines.append(
        f"| {r.config} | {r.dice:.4f} | {r.avd:.4f} | {r['count']:.4f} | {r.rq:.4f} | {r.pr:.4f} | {r.mean_rank:.3f} |"
    )
shipped_rank = float(grid[grid.config == "thr0.35_k20"].mean_rank.iloc[0])
best = grid.iloc[0]
lines += [
    "",
    f"Shipped `thr0.35_k20` ranks {int((grid.mean_rank < shipped_rank).sum()) + 1} of {len(grid)} on no-TTA maps; "
    f"best is `{best.config}` (Dice {best.dice:.4f} vs "
    f"{float(grid[grid.config == 'thr0.35_k20'].dice.iloc[0]):.4f}).",
    "",
    "Change the shipped operating point only if the gap is larger than the",
    "single-fold noise this table is measured with.",
]

(report_dir / "README.md").write_text("\n".join(lines) + "\n")
pd.DataFrame(rows).to_csv(report_dir / "paired_deltas.csv", index=False)
grid.to_csv(report_dir / "no_tta_grid.csv", index=False)
print("\n".join(lines))
PYEOF

cp -f "$OUT_DIR/report.md" "$OUT_DIR/report.json" "$REPORT_DIR/" 2>/dev/null || true
echo "[done] TTA ablation reduced -> $REPORT_DIR"
