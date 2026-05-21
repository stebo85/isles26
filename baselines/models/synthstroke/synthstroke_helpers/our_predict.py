#!/usr/bin/env python3
"""Inference for our self-trained SynthStroke model (plain MONAI UNet checkpoint).

Mirrors the validated file-based MONAI pipeline used in infer_synthstroke.py
(LoadImageD -> Orientation RAS -> Spacing 1mm -> HistogramNormalize ->
NormalizeIntensity -> sliding window [+TTA] -> inverse -> argmax), but builds the
34-class UNet ourselves and loads our checkpoint's "net" state_dict. The lesion
is the last class (index n_classes-1).
"""

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import monai as mn
from monai.networks.nets import UNet


def build_model(n_classes, device):
    return UNet(spatial_dims=3, in_channels=1, out_channels=n_classes,
                channels=[32, 64, 128, 256, 320, 320], strides=[2, 2, 2, 2, 2],
                kernel_size=3, up_kernel_size=3, num_res_units=1, act="PRELU",
                norm="INSTANCE", dropout=0.0, bias=True, adn_ordering="NDA").to(device).eval()


def run_inference(model, image_path, device, patch, use_tta, lesion_cls):
    load = mn.transforms.Compose([
        mn.transforms.LoadImageD(keys=["img"], image_only=True),
        mn.transforms.EnsureChannelFirstD(keys=["img"]),
        mn.transforms.ToTensorD(keys=["img"], device=device),
    ])
    preproc = mn.transforms.Compose([
        mn.transforms.OrientationD(keys=["img"], axcodes="RAS"),
        mn.transforms.SpacingD(keys=["img"], pixdim=1),
        mn.transforms.HistogramNormalizeD(keys="img"),
        mn.transforms.NormalizeIntensityD(keys="img", nonzero=False, channel_wise=True),
    ])
    softmax = mn.transforms.Activations(softmax=True)
    # Keep the full-volume 34-class output buffer on CPU (only 128^3 patches run
    # on GPU) so inference fits on small (~11GB) GPUs.
    window = mn.inferers.SlidingWindowInferer([patch] * 3, sw_batch_size=1, overlap=0.5,
                                              mode="gaussian", sigma_scale=0.125, progress=False,
                                              sw_device=device, device=torch.device("cpu"))
    batch = preproc(load({"img": str(image_path)}))
    img = batch["img"]
    with torch.no_grad():
        pred = window(img[None], model)[0]
        if use_tta:
            flips = [mn.transforms.Flip(spatial_axis=ax)
                     for ax in [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]]
            for flip in flips:
                pred += flip(window(flip(img)[None], model)[0])
            pred /= len(flips) + 1
    pred.applied_operations = img.applied_operations
    with mn.transforms.utils.allow_missing_keys_mode(preproc):
        inv = preproc.inverse({"img": pred})["img"]
    # Binary lesion mask by thresholding the lesion-channel probability (matches
    # the checkpoint-selection metric used in training; argmax over 34 classes
    # over-segments for an undertrained model).
    probs = softmax(inv)
    les = np.asarray(probs[lesion_cls].detach().cpu()) > 0.5
    return les.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="our_train .pt checkpoint (has 'net' state_dict)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-classes", type=int, default=34)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    lesion_cls = args.n_classes - 1
    model = build_model(args.n_classes, device)
    ck = torch.load(args.checkpoint, map_location=device)
    state = ck["net"] if isinstance(ck, dict) and "net" in ck else ck
    model.load_state_dict(state)

    ref = nib.load(str(args.image))
    mask = np.squeeze(run_inference(model, args.image, device, args.patch, args.tta, lesion_cls))
    if mask.shape != ref.shape[:3]:
        raise ValueError(f"mask shape {mask.shape} != input {ref.shape[:3]}")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    hdr = ref.header.copy(); hdr.set_data_dtype(np.uint8); hdr.set_slope_inter(1, 0)
    img = nib.Nifti1Image(mask.astype(np.uint8), ref.affine, header=hdr)
    img.set_data_dtype(np.uint8); img.header.set_slope_inter(1, 0)
    nib.save(img, str(out))
    print(f"[done] lesion_voxels={int(mask.sum())} out={out}")


if __name__ == "__main__":
    raise SystemExit(main())
