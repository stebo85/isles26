#!/usr/bin/env python3
"""Run SynthStroke inference on a single NIfTI image."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import monai as mn
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SYNTHSTROKE_REPO = REPO_ROOT / "work" / "synthstroke" / "repo"


def _resolve_existing_or_default(path_text: Optional[str], default: Path) -> Path:
    if not path_text:
        return default

    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path

    repo_path = (REPO_ROOT / path).resolve()
    if repo_path.exists():
        return repo_path

    return cwd_path


def _add_synthstroke_repo(repo_arg: Optional[str]) -> Path:
    repo_path = _resolve_existing_or_default(
        repo_arg or os.environ.get("SYNTHSTROKE_REPO"),
        DEFAULT_SYNTHSTROKE_REPO,
    ).resolve()
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SynthStroke lesion segmentation inference on one NIfTI image."
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Local weights directory with model.safetensors and config.json, or a HuggingFace Hub id with --from-hub.",
    )
    parser.add_argument("--from-hub", action="store_true", help="Load --weights as a HuggingFace Hub model id.")
    parser.add_argument("--image", required=True, help="Input .nii.gz image.")
    parser.add_argument("--out", required=True, help="Output .nii.gz lesion mask.")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation.")
    parser.add_argument("--patch", type=int, default=128, help="Sliding-window patch size.")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--repo", default=None, help="Path to the vendored SynthStroke repo.")
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_model(args: argparse.Namespace):
    from synthstroke_model import SynthStrokeModel

    if args.from_hub:
        return SynthStrokeModel.from_pretrained(args.weights)

    weights_dir = Path(args.weights).expanduser()
    checkpoint_path = weights_dir / "model.safetensors"
    config_path = weights_dir / "config.json"
    missing = [str(path) for path in (checkpoint_path, config_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Local weights directory must contain model.safetensors and config.json; "
            f"missing: {', '.join(missing)}"
        )
    return SynthStrokeModel.from_checkpoint(str(checkpoint_path))


def _run_inference(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    patch: int,
    use_tta: bool,
) -> np.ndarray:
    lesion_cls = 5 if model.config.out_channels == 6 else 1

    load = mn.transforms.Compose(
        [
            mn.transforms.LoadImageD(keys=["img"], image_only=True),
            mn.transforms.EnsureChannelFirstD(keys=["img"]),
            mn.transforms.ToTensorD(keys=["img"], device=device),
        ]
    )
    preproc = mn.transforms.Compose(
        [
            mn.transforms.OrientationD(keys=["img"], axcodes="RAS"),
            mn.transforms.SpacingD(keys=["img"], pixdim=1),
            mn.transforms.HistogramNormalizeD(keys="img"),
            mn.transforms.NormalizeIntensityD(keys="img", nonzero=False, channel_wise=True),
        ]
    )
    postproc = mn.transforms.Compose(
        [
            mn.transforms.Activations(softmax=True),
            mn.transforms.AsDiscrete(argmax=True),
        ]
    )
    window = mn.inferers.SlidingWindowInferer(
        [patch] * 3,
        sw_batch_size=1,
        overlap=0.5,
        mode="gaussian",
        sigma_scale=0.125,
        progress=False,
    )

    batch = load({"img": str(image_path)})
    batch = preproc(batch)
    img = batch["img"]

    with torch.no_grad():
        pred = window(img[None], model)[0]
        if use_tta:
            flips = [
                mn.transforms.Flip(spatial_axis=ax)
                for ax in [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
            ]
            for flip in flips:
                pred += flip(window(flip(img)[None], model)[0])
            pred /= len(flips) + 1

    pred.applied_operations = img.applied_operations
    with mn.transforms.utils.allow_missing_keys_mode(preproc):
        inv = preproc.inverse({"img": pred})["img"]
    disc = postproc(inv)

    label_map = np.asarray(disc.detach().cpu())[0]
    return (label_map == lesion_cls).astype(np.uint8)


def main() -> int:
    args = _parse_args()
    _add_synthstroke_repo(args.repo)

    device = _resolve_device(args.device)
    model = _load_model(args).to(device)
    model.eval()

    image_path = Path(args.image).expanduser()
    out_path = Path(args.out).expanduser()

    ref_img = nib.load(str(image_path))
    lesion_mask = np.squeeze(_run_inference(model, image_path, device, args.patch, args.tta))
    expected_shape = ref_img.shape[:3]
    if lesion_mask.shape != expected_shape:
        raise ValueError(
            f"Predicted mask shape {lesion_mask.shape} does not match input grid {expected_shape}."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ref_img.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    out_img = nib.Nifti1Image(lesion_mask.astype(np.uint8, copy=False), ref_img.affine, header=header)
    out_img.set_data_dtype(np.uint8)
    out_img.header.set_slope_inter(1, 0)
    nib.save(out_img, str(out_path))

    lesion_voxels = int(np.count_nonzero(lesion_mask))
    print(f"[done] lesion_voxels={lesion_voxels} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
