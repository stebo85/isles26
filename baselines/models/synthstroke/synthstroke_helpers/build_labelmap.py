#!/usr/bin/env python3
"""Build combined tissue+lesion label maps for SynthStroke-style training.

For each ATLAS case we have:
  - a FreeSurfer SynthSeg tissue segmentation (work/synthstroke/synthseg_labels/<CASE>.nii.gz)
  - the real binary lesion mask (nnU-Net labelsTr/<CASE>.nii.gz)
Both are on the identical native grid as the T1w (verified). We remap the
SynthSeg FreeSurfer aseg labels to contiguous tissue indices 0..K-1 and overlay
the lesion as the final class index K (lesion overwrites the tissue beneath it).
The resulting integer label map drives the cornucopia SynthFromLabelTransform
GMM image synthesis at training time.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import nibabel.processing as nibproc

# Canonical mri_synthseg (SynthSeg 2.0, non-robust) output label set — the fixed
# set of FreeSurfer aseg values it emits. Verified against produced outputs.
SYNTHSEG_LABELS = [
    0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28,
    41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60,
]
# Contiguous tissue indices 0..len-1; lesion is the next index.
TISSUE_INDEX = {lab: i for i, lab in enumerate(SYNTHSEG_LABELS)}
LESION_INDEX = len(SYNTHSEG_LABELS)          # 33
N_CLASSES = LESION_INDEX + 1                  # 34 (incl. background=0)

# --- compact6 scheme -------------------------------------------------------
# Group the FreeSurfer aseg labels into a small tissue set; lesion is the LAST
# class (index 5). Motivation: the 34-way softmax dilutes the lesion channel
# and busies the GMM synthesis; synth-plus (which beats our 34-class model)
# uses a 6-class scheme. 0=background, 1=CSF/ventricles, 2=gray matter
# (cortical+subcortical, L/R merged), 3=white matter, 4=cerebellum+brainstem
# (+ventral DC), 5=lesion.
COMPACT6_INDEX = {
    # CSF / ventricles -> 1
    4: 1, 5: 1, 14: 1, 15: 1, 24: 1, 43: 1, 44: 1,
    # gray matter (cortical + subcortical, left+right) -> 2
    3: 2, 10: 2, 11: 2, 12: 2, 13: 2, 17: 2, 18: 2, 26: 2,
    42: 2, 49: 2, 50: 2, 51: 2, 52: 2, 53: 2, 54: 2, 58: 2,
    # cerebral white matter -> 3
    2: 3, 41: 3,
    # cerebellum + brainstem (+ ventral DC) -> 4
    7: 4, 8: 4, 16: 4, 28: 4, 46: 4, 47: 4, 60: 4,
}
COMPACT6_LESION_INDEX = 5
COMPACT6_N_CLASSES = 6

SCHEMES = {
    "fs34": (TISSUE_INDEX, LESION_INDEX, N_CLASSES),
    "compact6": (COMPACT6_INDEX, COMPACT6_LESION_INDEX, COMPACT6_N_CLASSES),
}

REPO = Path(__file__).resolve().parents[4]


def build_one(synthseg_path: Path, lesion_path: Path, out_path: Path,
              tissue_map=TISSUE_INDEX, lesion_index=LESION_INDEX) -> int:
    ss_img = nib.load(str(synthseg_path))
    les_img = nib.load(str(lesion_path))

    # mri_synthseg outputs at 1mm, so for non-1mm native cases its grid differs
    # from the lesion/T1w native grid. Resample the SynthSeg labels onto the
    # lesion grid (nearest-neighbour) so tissue + lesion are voxel-aligned.
    if ss_img.shape != les_img.shape or not np.allclose(ss_img.affine, les_img.affine, atol=1e-3):
        ss_img = nibproc.resample_from_to(ss_img, (les_img.shape, les_img.affine), order=0)

    ss = np.rint(np.asanyarray(ss_img.dataobj)).astype(np.int32)
    les = np.asanyarray(les_img.dataobj)
    out = np.zeros(ss.shape, dtype=np.uint8)
    seen = set(np.unique(ss).tolist())
    unmapped = seen - set(tissue_map) - {0}
    if unmapped:
        # Leave unmapped values as background (0) but report them.
        print(f"[warn] {synthseg_path.name}: unmapped synthseg labels {sorted(unmapped)} -> background")
    for lab, idx in tissue_map.items():
        if idx == 0:
            continue
        out[ss == lab] = idx
    # Lesion overwrites tissue.
    out[(les > 0.5)] = lesion_index

    header = ss_img.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    out_img = nib.Nifti1Image(out, ss_img.affine, header=header)
    out_img.set_data_dtype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))
    return int((out == lesion_index).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthseg-dir", default=str(REPO / "work/synthstroke/synthseg_labels"))
    ap.add_argument("--lesion-dir", default=str(REPO / "work/nnunet/nnUNet_raw/Dataset501_ATLASR21/labelsTr"))
    ap.add_argument("--scheme", choices=list(SCHEMES), default="fs34",
                    help="fs34 = original 34-class FreeSurfer scheme; compact6 = 6-class grouping")
    ap.add_argument("--out-dir", default=None,
                    help="default: work/synthstroke/labelmaps (fs34) or labelmaps_compact6 (compact6)")
    ap.add_argument("--scheme-json", default=None,
                    help="default: <out-dir>/label_scheme.json (fs34) or label_scheme_compact6.json")
    args = ap.parse_args()

    tissue_map, lesion_index, n_classes = SCHEMES[args.scheme]
    default_out = "labelmaps" if args.scheme == "fs34" else "labelmaps_compact6"
    out_dir = Path(args.out_dir) if args.out_dir else (REPO / "work/synthstroke" / default_out)
    scheme_json = Path(args.scheme_json) if args.scheme_json else (
        out_dir / ("label_scheme.json" if args.scheme == "fs34" else "label_scheme_compact6.json"))

    ss_dir = Path(args.synthseg_dir)
    les_dir = Path(args.lesion_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Emit the label scheme once (reflects the active scheme).
    scheme_json.parent.mkdir(parents=True, exist_ok=True)
    scheme_json.write_text(json.dumps({
        "scheme": args.scheme,
        "synthseg_labels": SYNTHSEG_LABELS,
        "tissue_index": {str(k): v for k, v in tissue_map.items()},
        "lesion_index": lesion_index,
        "n_classes": n_classes,
    }, indent=2) + "\n")
    print(f"[info] scheme={args.scheme} n_classes={n_classes} lesion_index={lesion_index} out_dir={out_dir}")

    cases = sorted(p.name[:-len(".nii.gz")] for p in ss_dir.glob("*.nii.gz"))
    if not cases:
        raise SystemExit(f"no synthseg labels found in {ss_dir}")

    n_done = n_skip = n_missing_lesion = n_failed = 0
    for case in cases:
        out_path = out_dir / f"{case}.nii.gz"
        if out_path.exists() and out_path.stat().st_size > 0:
            n_skip += 1
            continue
        les_path = les_dir / f"{case}.nii.gz"
        if not les_path.exists():
            print(f"[warn] no lesion mask for {case}; skipping")
            n_missing_lesion += 1
            continue
        try:
            lvox = build_one(ss_dir / f"{case}.nii.gz", les_path, out_path,
                             tissue_map=tissue_map, lesion_index=lesion_index)
        except Exception as exc:  # noqa: BLE001 - log and continue, don't abort the batch
            print(f"[error] {case}: {exc}")
            n_failed += 1
            continue
        n_done += 1
        if n_done <= 3 or n_done % 200 == 0:
            print(f"[ok] {case}: lesion_voxels={lvox} -> {out_path.name}")

    print(f"[done] built={n_done} skipped={n_skip} missing_lesion={n_missing_lesion} "
          f"total_synthseg={len(cases)} scheme={args.scheme} n_classes={n_classes} lesion_index={lesion_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
