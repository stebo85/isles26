#!/usr/bin/env python3
"""Train a SynthStroke-style model on our self-contained ATLAS label maps.

We reuse the vendored SynthStroke repo's tested transforms (CCSynthSeg cornucopia
GMM synthesis) and loss (DiceCEL2Loss) and mirror the repo's train.py loop, but:
  - drop the wandb coupling (offline cluster) -> plain logging,
  - load OUR 34-class label maps (work/synthstroke/labelmaps/<CASE>.nii.gz;
    tissue 0..32 + lesion 33) for the fold-0 split,
  - support --mix-real: 50:50 synthetic (GMM-from-label, contrast-randomized)
    and real ATLAS T1w batches (same 34-class label as target),
  - preempt-safe checkpointing for SLURM requeue.

The model is a MONAI UNet with out_channels = N_CLASSES (34); the lesion is the
last channel, so validation dice is computed on channel -1.
"""

import argparse
import glob
import json
import os
import signal
import sys
from pathlib import Path

import numpy as np
import torch
import monai as mn
from monai.networks.nets import UNet

REPO = Path(__file__).resolve().parents[4]


def _add_repo(repo_arg):
    repo = Path(repo_arg) if repo_arg else (REPO / "work/synthstroke/repo")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def fold0(splits_path):
    data = json.loads(Path(splits_path).read_text())
    folds = data if isinstance(data, list) else data["folds"]
    return list(folds[0]["train"]), list(folds[0]["val"])


class SynthFromLabelD(mn.transforms.MapTransform):
    """Wrap cornucopia's native SynthFromLabelTransform as a MONAI dict transform.

    Takes an integer label map (1,H,W,D) at d[label_key], synthesizes a
    contrast-randomized intensity image (deform + GMM + bias/gamma/noise/PV) and
    writes d[image_key]; the deformed integer label (classes preserved, incl. the
    lesion) replaces d[label_key]. Replaces the repo's vendored CCSynthSeg, which
    targets an unreleased cornucopia API.
    """

    def __init__(self, syn, label_key="label", image_key="image"):
        super().__init__([label_key])
        self.syn = syn
        self.lk = label_key
        self.ik = image_key

    def __call__(self, data):
        d = dict(data)
        lab = d[self.lk]
        lab = lab.long() if torch.is_tensor(lab) and lab.dtype.is_floating_point else lab
        out = self.syn(lab)
        img, new_lab = (out["image"], out["label"]) if not isinstance(out, (tuple, list)) else out
        d[self.ik] = torch.as_tensor(img).float()
        d[self.lk] = torch.as_tensor(new_lab).long()
        return d


def synth_transform(patch, n_classes, device, syn):
    # cornucopia SynthFromLabelTransform synthesizes an image from the integer
    # label map and returns the deformed integer label (lesion = class 33). We
    # crop to the patch first (cheap), synthesize, augment, then one-hot the
    # integer label as the training target.
    return mn.transforms.Compose([
        mn.transforms.LoadImageD(keys=["label"], image_only=True),
        mn.transforms.EnsureChannelFirstD(keys=["label"]),
        mn.transforms.SpacingD(keys=["label"], pixdim=1, mode="nearest"),
        mn.transforms.ToTensorD(keys=["label"], dtype=torch.float32, device=device),
        mn.transforms.RandSpatialCropD(keys=["label"], roi_size=(patch,) * 3, random_size=False),
        SynthFromLabelD(syn, label_key="label", image_key="image"),
        mn.transforms.RandAxisFlipd(keys=["image", "label"], prob=0.8),
        mn.transforms.RandAxisFlipd(keys=["image", "label"], prob=0.8),
        mn.transforms.RandAxisFlipd(keys=["image", "label"], prob=0.8),
        mn.transforms.NormalizeIntensityD(keys="image", nonzero=False, channel_wise=True),
        mn.transforms.AsDiscreteD(keys="label", to_onehot=n_classes),
        mn.transforms.ToTensorD(keys=["image", "label"], dtype=torch.float32),
    ])


def real_transform(patch, n_classes, device, train=True):
    keys = ["image", "label"]
    tfms = [
        mn.transforms.LoadImageD(keys=keys, image_only=True),
        mn.transforms.EnsureChannelFirstD(keys=keys),
        mn.transforms.OrientationD(keys=keys, axcodes="RAS"),
        mn.transforms.SpacingD(keys=["image"], pixdim=1, mode="bilinear"),
        mn.transforms.SpacingD(keys=["label"], pixdim=1, mode="nearest"),
        mn.transforms.ToTensorD(keys=keys, device=device),
    ]
    if train:
        tfms += [
            mn.transforms.RandSpatialCropD(keys=keys, roi_size=(patch,) * 3, random_size=False),
            mn.transforms.RandAxisFlipd(keys=keys, prob=0.8),
            mn.transforms.RandAxisFlipd(keys=keys, prob=0.8),
            mn.transforms.RandAxisFlipd(keys=keys, prob=0.8),
        ]
    tfms += [
        mn.transforms.HistogramNormalizeD(keys="image"),
        mn.transforms.NormalizeIntensityD(keys="image", nonzero=False, channel_wise=True),
        mn.transforms.AsDiscreteD(keys="label", to_onehot=n_classes),
        mn.transforms.ToTensorD(keys=keys, dtype=torch.float32),
    ]
    return mn.transforms.Compose(tfms)


def val_transform(device):
    # Full-volume validation: image on GPU, integer label kept on CPU (no one-hot,
    # no crop) to minimise GPU memory on small cards.
    return mn.transforms.Compose([
        mn.transforms.LoadImageD(keys=["image", "label"], image_only=True),
        mn.transforms.EnsureChannelFirstD(keys=["image", "label"]),
        mn.transforms.OrientationD(keys=["image", "label"], axcodes="RAS"),
        mn.transforms.SpacingD(keys=["image"], pixdim=1, mode="bilinear"),
        mn.transforms.SpacingD(keys=["label"], pixdim=1, mode="nearest"),
        mn.transforms.HistogramNormalizeD(keys="image"),
        mn.transforms.NormalizeIntensityD(keys="image", nonzero=False, channel_wise=True),
        mn.transforms.ToTensorD(keys=["image"], dtype=torch.float32, device=device),
        mn.transforms.ToTensorD(keys=["label"], dtype=torch.float32, device=torch.device("cpu")),
    ])


def make_loader(dicts, transform, shuffle):
    ds = mn.data.Dataset(dicts, transform=transform)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=0)


def cycle(loader):
    while True:
        for b in loader:
            yield b


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--logdir", default=str(REPO / "work/synthstroke/train"))
    ap.add_argument("--labelmaps-dir", default=str(REPO / "work/synthstroke/labelmaps"))
    ap.add_argument("--images-dir", default=str(REPO / "work/nnunet/nnUNet_raw/Dataset501_ATLASR21/imagesTr"))
    ap.add_argument("--splits", default=str(REPO / "work/nnunet/nnUNet_preprocessed/Dataset501_ATLASR21/splits_final.json"))
    ap.add_argument("--repo", default=None)
    ap.add_argument("--n-classes", type=int, default=34)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--epoch-length", type=int, default=200)
    ap.add_argument("--val-interval", type=int, default=5)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=int, default=50)
    ap.add_argument("--lesion-weight", type=float, default=2.0)
    ap.add_argument("--mix-real", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-val", type=int, default=32, help="cap in-training val cases for checkpoint selection")
    ap.add_argument("--smoke", action="store_true", help="few steps for a quick sanity run")
    args = ap.parse_args()

    repo = _add_repo(args.repo)
    import cornucopia as cc
    from custom import DiceCEL2Loss
    syn = cc.SynthFromLabelTransform()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    n_classes = args.n_classes
    lesion_idx = n_classes - 1
    outdir = Path(args.logdir) / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[info] device={device} n_classes={n_classes} mix_real={args.mix_real} epochs={args.epochs}")

    train_ids, val_ids = fold0(args.splits)
    lm = Path(args.labelmaps_dir)
    im = Path(args.images_dir)
    train_ids = [c for c in train_ids if (lm / f"{c}.nii.gz").exists()]
    val_ids = [c for c in val_ids if (lm / f"{c}.nii.gz").exists()]
    if args.smoke:
        train_ids, val_ids = train_ids[:8], val_ids[:4]
    elif args.max_val and len(val_ids) > args.max_val:
        # Cap in-training validation (checkpoint selection) for speed; the full
        # fold-0 val set is evaluated properly in Step 7. Deterministic stride.
        stride = len(val_ids) / args.max_val
        val_ids = [val_ids[int(i * stride)] for i in range(args.max_val)]
    print(f"[info] fold0 train={len(train_ids)} val={len(val_ids)} (with label maps)")
    if not train_ids:
        raise SystemExit("no training label maps found; run analysis_synth_05 first")

    synth_dicts = [{"label": str(lm / f"{c}.nii.gz")} for c in train_ids]
    real_dicts = [{"image": str(im / f"{c}_0000.nii.gz"), "label": str(lm / f"{c}.nii.gz")} for c in train_ids]
    val_dicts = [{"image": str(im / f"{c}_0000.nii.gz"), "label": str(lm / f"{c}.nii.gz")} for c in val_ids]

    synth_loader = make_loader(synth_dicts, synth_transform(args.patch, n_classes, device, syn), True)
    real_loader = make_loader(real_dicts, real_transform(args.patch, n_classes, device, True), True) if args.mix_real else None
    val_loader = make_loader(val_dicts, val_transform(device), False)

    model = UNet(spatial_dims=3, in_channels=1, out_channels=n_classes,
                 channels=[32, 64, 128, 256, 320, 320], strides=[2, 2, 2, 2, 2],
                 kernel_size=3, up_kernel_size=3, num_res_units=1, act="PRELU",
                 norm="INSTANCE", dropout=0.0, bias=True, adn_ordering="NDA").to(device)

    ce_weight = torch.ones(n_classes, device=device)
    ce_weight[lesion_idx] = args.lesion_weight
    crit = DiceCEL2Loss(include_background=False, to_onehot_y=False, softmax=True,
                        reduction="mean", smooth_nr=1e-5, smooth_dr=1e-5, batch=True,
                        lambda_dice=1.0, lambda_ce=1.0, ce_weight=ce_weight, l2_epochs=args.l2)
    opt = torch.optim.AdamW(model.parameters(), args.lr)

    # resume
    start_epoch, metric_best = 0, 0.0
    ckpt_last = outdir / "checkpoint.pt"
    if ckpt_last.exists():
        ck = torch.load(str(ckpt_last), map_location=device)
        model.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1; metric_best = ck["metric"]
        print(f"[info] resumed from epoch {ck['epoch']} (best={metric_best:.4f})")

    def lr_lambda(epoch):
        return (1 - epoch / args.epochs) ** 0.9
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lr_lambda], last_epoch=start_epoch - 1)

    # preempt-safe: on SIGUSR1, save and requeue
    def on_usr1(signum, frame):
        torch.save({"net": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": cur_epoch, "metric": metric_best}, str(ckpt_last))
        jid = os.environ.get("SLURM_JOB_ID")
        if jid:
            os.system(f"scontrol requeue {jid}")
        sys.exit(0)
    signal.signal(signal.SIGUSR1, on_usr1)

    synth_iter, real_iter = cycle(synth_loader), (cycle(real_loader) if real_loader else None)
    epoch_len = 6 if args.smoke else args.epoch_length
    epochs = (start_epoch + 1) if args.smoke else args.epochs
    cur_epoch = start_epoch
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for epoch in range(start_epoch, epochs):
        cur_epoch = epoch
        model.train()
        running = 0.0
        n_ok = 0
        n_nan = 0
        for step in range(epoch_len):
            # 50:50 synth/real interleave
            if real_iter is not None and step % 2 == 1:
                batch = next(real_iter)
            else:
                batch = next(synth_iter)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=args.amp):
                logits = model(images)
                loss = crit(logits, labels)
            # NaN/inf-resilient: skip the offending step rather than abort the run.
            if not torch.isfinite(loss):
                n_nan += 1
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(opt); scaler.update()
            running += float(loss.detach())
            n_ok += 1
        sched.step()
        print(f"[epoch {epoch}] train_loss={running/max(n_ok,1):.4f} lr={opt.param_groups[0]['lr']:.2e} "
              f"steps_ok={n_ok} nan_skipped={n_nan}", flush=True)

        if (epoch + 1) % args.val_interval == 0 or args.smoke or epoch == epochs - 1:
            model.eval()
            dices = []
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    # Keep the full-volume 34-class output buffer on CPU (only the
                    # 128^3 patches run on GPU) so validation fits on small GPUs.
                    logits = mn.inferers.sliding_window_inference(
                        images, (args.patch,) * 3, 1, model, overlap=0.5, mode="gaussian",
                        sw_device=device, device=torch.device("cpu"))
                    probs = torch.softmax(logits, dim=1)
                    pred = (probs[:, lesion_idx] > 0.5).float()
                    gt = (batch["label"][:, 0] == lesion_idx).float()
                    inter = (pred * gt).sum(); denom = pred.sum() + gt.sum()
                    dices.append(float((2 * inter / denom).cpu()) if denom > 0 else (1.0 if gt.sum() == 0 else 0.0))
            metric = float(np.nanmean(dices)) if dices else 0.0
            print(f"[epoch {epoch}] val_lesion_dice={metric:.4f} (n={len(dices)})", flush=True)
            ck = {"net": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "metric": max(metric, metric_best)}
            torch.save(ck, str(ckpt_last))
            if metric > metric_best:
                metric_best = metric
                torch.save(ck, str(outdir / "checkpoint_best.pt"))
                print(f"[epoch {epoch}] new best {metric_best:.4f}", flush=True)

    print(f"[done] best val lesion dice={metric_best:.4f} ckpt={outdir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
