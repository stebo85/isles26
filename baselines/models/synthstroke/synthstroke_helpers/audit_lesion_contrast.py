#!/usr/bin/env python3
"""Audit how the cornucopia synth pipeline renders the LESION class.

Hypothesis under test (2026-06-30 strategic reframe): because
`cc.SynthFromLabelTransform()` assigns the lesion (label 33) an *independent
random* GMM intensity per sample - exactly like an anatomical tissue - a large
fraction of synthetic samples render the lesion at ~the same intensity as its
perilesional tissue. The lesion is then VISUALLY INVISIBLE yet still carries a
"lesion" label -> contradictory supervision. A stroke lesion (unlike anatomy)
has no consistent shape/location, so the only learnable cue is contrast; if the
contrast is randomly absent, the model cannot learn it. This likely explains
Run F (synth-only) collapsing to ATLAS Dice 0.139, and is the real (fixable)
root cause rather than "we need multi-dataset real data".

This script replicates the exact pre-synthesis crop pipeline from
our_train.synth_transform, runs the SAME `cc.SynthFromLabelTransform()`, and for
every lesion-containing synthetic patch measures the lesion-vs-perilesional
contrast-to-noise ratio (CNR):

    CNR = (mean(lesion) - mean(perilesional_tissue)) / std(tissue)

It reports the distribution of signed CNR, the fraction of samples that are
"invisible" (|CNR| < --invisible-thr), and the bright/dark polarity split.
A symmetric spread with a big spike near 0 confirms the hypothesis.

CPU-only. No GPU required.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
import monai as mn

REPO = Path(__file__).resolve().parents[4]


def build_pre_synth(patch, lesion_idx, num_samples):
    """Mirror our_train.synth_transform up to (but not including) synthesis,
    forcing lesion-centred crops (pos=1, neg=0) so every patch has lesion."""
    return mn.transforms.Compose([
        mn.transforms.LoadImageD(keys=["label"], image_only=True),
        mn.transforms.EnsureChannelFirstD(keys=["label"]),
        mn.transforms.SpacingD(keys=["label"], pixdim=1, mode="nearest"),
        mn.transforms.ToTensorD(keys=["label"], dtype=torch.float32),
        _LesionMaskD(lesion_idx),
        mn.transforms.RandCropByPosNegLabeld(
            keys=["label", "lesion"], label_key="lesion",
            spatial_size=(patch,) * 3, pos=1, neg=0,
            num_samples=num_samples, allow_smaller=True),
        mn.transforms.SpatialPadD(keys=["label", "lesion"], spatial_size=(patch,) * 3),
        mn.transforms.DeleteItemsd(keys=["lesion"]),
    ])


class _LesionMaskD(mn.transforms.MapTransform):
    def __init__(self, lesion_idx):
        super().__init__(["label"])
        self.lesion_idx = lesion_idx

    def __call__(self, data):
        d = dict(data)
        lab = d["label"]
        mask = (lab == self.lesion_idx).float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        d["lesion"] = mask
        return d


def measure(img, lab, lesion_idx, dilate_iters, min_peri):
    """Return (cnr, sign, mean_les, mean_peri, std_tissue) or None if unusable."""
    img = np.squeeze(np.asarray(img, dtype=np.float64))
    lab = np.squeeze(np.asarray(lab))
    les = lab == lesion_idx
    if les.sum() < 8:
        return None
    tissue = (lab > 0) & (~les)
    if tissue.sum() < 64:
        return None
    # Perilesional ring: match LesionContrastD's 3x3x3 max_pool3d dilation
    # (26-connectivity), then subtract lesion and keep only real tissue.
    dil = ndi.binary_dilation(
        les,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=dilate_iters,
    )
    peri = dil & (~les) & tissue
    if peri.sum() < min_peri:
        return None
    mean_les = float(img[les].mean())
    mean_peri = float(img[peri].mean())
    std_tissue = float(img[tissue].std())
    if not np.isfinite(std_tissue) or std_tissue <= 0:
        return None
    contrast = mean_les - mean_peri
    cnr = contrast / std_tissue
    return cnr, (1 if contrast >= 0 else -1), mean_les, mean_peri, std_tissue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelmaps-dir", default=str(REPO / "work/synthstroke/labelmaps"))
    ap.add_argument("--lesion-idx", type=int, default=33)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--n-cases", type=int, default=60)
    ap.add_argument("--samples-per-case", type=int, default=8)
    ap.add_argument("--dilate-iters", type=int, default=3)
    ap.add_argument("--min-peri", type=int, default=64)
    ap.add_argument("--invisible-thr", type=float, default=0.5,
                    help="|CNR| below this = lesion effectively invisible vs perilesional tissue")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    help="auto|cuda|cpu; synthesis runs on this device (training uses GPU, "
                         "so CPU is far slower). 'auto' = cuda if available.")
    ap.add_argument("--lesion-contrast-prob", type=float, default=0.0,
                    help="if >0, apply LesionContrastD (from our_train) to each synth patch "
                         "before measuring, to verify the fix collapses frac_invisible and "
                         "bounds |CNR| into [cnr-lo, cnr-hi]. Default 0 = audit raw synthesis.")
    ap.add_argument("--lesion-contrast-cnr-lo", type=float, default=0.75)
    ap.add_argument("--lesion-contrast-cnr-hi", type=float, default=2.0)
    ap.add_argument("--out-dir", default=str(REPO / "baselines/reports/synthstroke/lesion_contrast_audit"))
    args = ap.parse_args()

    import cornucopia as cc
    mn.utils.set_determinism(seed=args.seed)

    lc = None
    if args.lesion_contrast_prob and args.lesion_contrast_prob > 0:
        from baselines.models.synthstroke.synthstroke_helpers.our_train import LesionContrastD
        lc = LesionContrastD(
            args.lesion_idx, image_key="image", label_key="label",
            prob=args.lesion_contrast_prob,
            cnr_lo=args.lesion_contrast_cnr_lo, cnr_hi=args.lesion_contrast_cnr_hi,
            dilate_iters=args.dilate_iters,
        )
        lc.set_random_state(seed=args.seed)
        print(f"[info] APPLYING LesionContrastD prob={args.lesion_contrast_prob} "
              f"cnr=[{args.lesion_contrast_cnr_lo},{args.lesion_contrast_cnr_hi}] before measuring", flush=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[info] synthesis device={device}", flush=True)
    syn = cc.SynthFromLabelTransform()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(Path(args.labelmaps_dir) / "*.nii.gz")))
    rng = np.random.default_rng(args.seed)
    if len(files) > args.n_cases:
        files = list(rng.choice(files, size=args.n_cases, replace=False))
    print(f"[info] auditing {len(files)} cases x {args.samples_per_case} samples; "
          f"lesion_idx={args.lesion_idx} patch={args.patch}", flush=True)

    pre = build_pre_synth(args.patch, args.lesion_idx, args.samples_per_case)

    cnrs, signs = [], []
    n_patches = n_used = n_invisible = 0
    examples = []  # (cnr, img_slice, les_slice) for a few qualitative PNGs

    for fi, f in enumerate(files):
        try:
            crops = pre({"label": f})
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {Path(f).name}: pre-synth failed: {exc}", flush=True)
            continue
        if not isinstance(crops, list):
            crops = [crops]
        for crop in crops:
            n_patches += 1
            try:
                out = syn(crop["label"].long().to(device))
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {Path(f).name}: synth failed: {exc}", flush=True)
                continue
            if isinstance(out, (tuple, list)):
                img, new_lab = out
            else:
                img, new_lab = out["image"], out["label"]
            if lc is not None:
                # Apply the fix on the same device tensors the trainer would see,
                # then measure -> proves the transform on real cornucopia output.
                img_t = torch.as_tensor(img).float().to(device)
                lab_t = torch.as_tensor(new_lab).to(device)
                res = lc({"image": img_t, "label": lab_t})
                img, new_lab = res["image"], res["label"]
            if torch.is_tensor(img):
                img = img.detach().float().cpu()
            if torch.is_tensor(new_lab):
                new_lab = new_lab.detach().cpu()
            m = measure(img, new_lab, args.lesion_idx, args.dilate_iters, args.min_peri)
            if m is None:
                continue
            cnr, sign, ml, mp, st = m
            cnrs.append(cnr)
            signs.append(sign)
            n_used += 1
            if abs(cnr) < args.invisible_thr:
                n_invisible += 1
            if len(examples) < 9 and fi < 12:
                a = np.squeeze(np.asarray(img, dtype=np.float64))
                l = np.squeeze(np.asarray(new_lab)) == args.lesion_idx
                z = int(np.argmax(l.sum(axis=(0, 1)))) if l.any() else a.shape[2] // 2
                examples.append((cnr, a[:, :, z], l[:, :, z]))
        if (fi + 1) % 10 == 0:
            print(f"[info] {fi+1}/{len(files)} cases; used {n_used} patches so far", flush=True)

    cnrs = np.asarray(cnrs, dtype=np.float64)
    signs = np.asarray(signs, dtype=np.int64)
    stats = {
        "n_cases": len(files),
        "n_patches": n_patches,
        "n_used": n_used,
        "invisible_thr": args.invisible_thr,
        "dilation_connectivity": 26,
        "dilate_iters": args.dilate_iters,
        "frac_invisible": float(n_invisible / max(n_used, 1)),
        "frac_bright_lesion": float((signs > 0).mean()) if n_used else None,
        "frac_dark_lesion": float((signs < 0).mean()) if n_used else None,
        "abs_cnr_mean": float(np.abs(cnrs).mean()) if n_used else None,
        "abs_cnr_median": float(np.median(np.abs(cnrs))) if n_used else None,
        "cnr_mean": float(cnrs.mean()) if n_used else None,
        "cnr_std": float(cnrs.std()) if n_used else None,
        "cnr_pctiles": {p: float(np.percentile(cnrs, p)) for p in (1, 5, 25, 50, 75, 95, 99)} if n_used else None,
    }
    (out_dir / "lesion_contrast_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print("[result] " + json.dumps(stats, indent=2), flush=True)

    # Histogram + example panels
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.hist(np.clip(cnrs, -6, 6), bins=60, color="#4477aa", edgecolor="k", linewidth=0.3)
        ax.axvspan(-args.invisible_thr, args.invisible_thr, color="red", alpha=0.18,
                   label=f"|CNR|<{args.invisible_thr} (invisible: {stats['frac_invisible']*100:.0f}%)")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("signed lesion-vs-perilesional CNR")
        ax.set_ylabel("# synthetic patches")
        ax.set_title(f"Synth lesion contrast (n={n_used} patches)\n"
                     f"bright={stats['frac_bright_lesion']*100:.0f}% dark={stats['frac_dark_lesion']*100:.0f}% "
                     f"|CNR|med={stats['abs_cnr_median']:.2f}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "lesion_contrast_hist.png", dpi=130)
        plt.close(fig)

        if examples:
            k = len(examples)
            fig, axes = plt.subplots(1, k, figsize=(2.4 * k, 2.6))
            if k == 1:
                axes = [axes]
            for ax, (cnr, im, lm) in zip(axes, examples):
                ax.imshow(im.T, cmap="gray", origin="lower")
                ax.contour(lm.T, levels=[0.5], colors="r", linewidths=0.8)
                ax.set_title(f"CNR={cnr:+.2f}", fontsize=8)
                ax.axis("off")
            fig.suptitle("Synth patches: lesion contour (red) on synthesized image", fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / "lesion_contrast_examples.png", dpi=130)
            plt.close(fig)
        print(f"[ok] wrote PNGs to {out_dir}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] plotting failed (stats still saved): {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
