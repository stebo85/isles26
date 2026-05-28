#!/usr/bin/env python3
"""Train a SynthStroke-style model on our self-contained ATLAS label maps.

We reuse the vendored SynthStroke repo's tested transforms (CCSynthSeg cornucopia
GMM synthesis) and loss (DiceCEL2Loss) and mirror the repo's train.py loop, but:
  - drop the wandb coupling (offline cluster) -> plain logging,
  - load OUR 34-class label maps (work/synthstroke/labelmaps/<CASE>.nii.gz;
    tissue 0..32 + lesion 33) for the fold-0 split,
  - support --mix-real: configurable synthetic:real mix (default 33% synth = 67%
    real, ATLAS-first) controlled by --synth-prob,
  - lesion-focused patch sampling via RandCropByPosNegLabeld,
  - preempt-safe checkpointing for SLURM requeue.

The model is a MONAI UNet with out_channels = N_CLASSES (34); the lesion is the
last channel, so validation dice is computed on channel -1.
"""

import argparse
import glob
import json
import os
import random
import signal
import sys
from pathlib import Path

import numpy as np
import torch
import monai as mn
from monai.data import list_data_collate
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


class LesionMaskD(mn.transforms.MapTransform):
    """Write a binary lesion mask key for use with RandCropByPosNegLabeld.

    Creates d[mask_key] = (d[label_key] == lesion_idx) as float32, shape
    (1, H, W, D), suitable for MONAI's positive/negative label sampling.
    """

    def __init__(self, lesion_idx, label_key="label", mask_key="lesion"):
        super().__init__([label_key])
        self.lesion_idx = lesion_idx
        self.lk = label_key
        self.mk = mask_key

    def __call__(self, data):
        d = dict(data)
        lab = d[self.lk]
        if torch.is_tensor(lab):
            mask = (lab == self.lesion_idx).float()
        else:
            mask = torch.as_tensor((np.asarray(lab) == self.lesion_idx).astype(np.float32))
        # Ensure channel-first shape (1, H, W, D)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        d[self.mk] = mask
        return d


class LesionFadeD(mn.transforms.MapTransform):
    """Lesion intensity 'fade' (synth path only): multiply the synthesized image
    inside the lesion region by a smooth low-frequency random field (~[lo, hi]),
    so the lesion is not a single flat GMM intensity. Mirrors the repo's --fade
    intent (penumbra / intensity inhomogeneity) for our integer-label cornucopia
    pipeline. Applied AFTER SynthFromLabelD (needs the synthesized image + the
    deformed integer label).
    """

    def __init__(self, lesion_idx, image_key="image", label_key="label",
                 lo=0.5, hi=1.5, ctrl=4):
        super().__init__([image_key])
        self.lesion_idx = lesion_idx
        self.ik = image_key
        self.lk = label_key
        self.lo, self.hi, self.ctrl = lo, hi, ctrl

    def __call__(self, data):
        d = dict(data)
        img = d[self.ik]
        lab = d[self.lk]
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        les = (lab == self.lesion_idx)
        if bool(les.any()):
            shp = tuple(int(s) for s in img.shape[-3:])
            coarse = torch.rand((1, 1, self.ctrl, self.ctrl, self.ctrl),
                                dtype=torch.float32, device=img.device)
            field = torch.nn.functional.interpolate(
                coarse, size=shp, mode="trilinear", align_corners=False)[0]
            field = self.lo + (self.hi - self.lo) * field          # (1,H,W,D) in [lo,hi]
            les_f = les.to(img.dtype)
            img = img * (1.0 - les_f) + img * field * les_f
            d[self.ik] = img
        return d


def synth_transform(patch, n_classes, device, syn, fade=False):
    """Synthetic transform: lesion-focused crop BEFORE synthesis for efficiency.

    Critical ordering:
    Load label -> EnsureChannelFirst -> Spacing(nearest) -> ToTensor ->
    LesionMaskD (build lesion binary mask) ->
    RandCropByPosNegLabeld (crop label + lesion mask) ->
    SpatialPadD (pad up to patch if crop was smaller) ->
    DeleteItemsd (drop lesion key) ->
    SynthFromLabelD (synthesize image from cropped label) ->
    [LesionFadeD if fade] -> flips -> NormalizeIntensity -> AsDiscrete -> ToTensor
    """
    lesion_idx = n_classes - 1
    tfms = [
        mn.transforms.LoadImageD(keys=["label"], image_only=True),
        mn.transforms.EnsureChannelFirstD(keys=["label"]),
        mn.transforms.SpacingD(keys=["label"], pixdim=1, mode="nearest"),
        mn.transforms.ToTensorD(keys=["label"], dtype=torch.float32, device=device),
        LesionMaskD(lesion_idx, label_key="label", mask_key="lesion"),
        mn.transforms.RandCropByPosNegLabeld(
            keys=["label", "lesion"],
            label_key="lesion",
            spatial_size=(patch,) * 3,
            pos=1,
            neg=1,
            num_samples=1,
            allow_smaller=True,
        ),
        mn.transforms.SpatialPadD(keys=["label", "lesion"], spatial_size=(patch,) * 3),
        mn.transforms.DeleteItemsd(keys=["lesion"]),
        SynthFromLabelD(syn, label_key="label", image_key="image"),
    ]
    if fade:
        tfms.append(LesionFadeD(lesion_idx, image_key="image", label_key="label"))
    tfms += [
        mn.transforms.RandAxisFlipd(keys=["image", "label"], prob=0.8),
        mn.transforms.RandAxisFlipd(keys=["image", "label"], prob=0.8),
        mn.transforms.RandAxisFlipd(keys=["image", "label"], prob=0.8),
        mn.transforms.NormalizeIntensityD(keys="image", nonzero=False, channel_wise=True),
        mn.transforms.AsDiscreteD(keys="label", to_onehot=n_classes),
        mn.transforms.ToTensorD(keys=["image", "label"], dtype=torch.float32),
    ]
    return mn.transforms.Compose(tfms)


def real_transform(patch, n_classes, device, train=True):
    """Real ATLAS T1w transform.

    Train branch uses lesion-focused RandCropByPosNegLabeld.
    Val/non-train branch keeps full-volume behavior (no crop).
    """
    lesion_idx = n_classes - 1
    keys = ["image", "label"]
    tfms = [
        mn.transforms.LoadImageD(keys=keys, image_only=True),
        mn.transforms.EnsureChannelFirstD(keys=keys),
        mn.transforms.OrientationD(keys=keys, axcodes="RAS"),
        mn.transforms.SpacingD(keys=["image"], pixdim=1, mode="bilinear"),
        mn.transforms.SpacingD(keys=["label"], pixdim=1, mode="nearest"),
        mn.transforms.ToTensorD(keys=keys, device=device),
        # Normalize the FULL volume before cropping so training intensity stats
        # match inference (our_predict normalizes the whole volume, then sliding
        # window). Previously normalization ran on the 128^3 crop -> train/test
        # distribution mismatch.
        mn.transforms.HistogramNormalizeD(keys="image"),
        mn.transforms.NormalizeIntensityD(keys="image", nonzero=False, channel_wise=True),
    ]
    if train:
        tfms += [
            LesionMaskD(lesion_idx, label_key="label", mask_key="lesion"),
            mn.transforms.RandCropByPosNegLabeld(
                keys=["image", "label", "lesion"],
                label_key="lesion",
                spatial_size=(patch,) * 3,
                pos=1,
                neg=1,
                num_samples=1,
                allow_smaller=True,
            ),
            mn.transforms.SpatialPadD(keys=["image", "label", "lesion"], spatial_size=(patch,) * 3),
            mn.transforms.DeleteItemsd(keys=["lesion"]),
            mn.transforms.RandAxisFlipd(keys=keys, prob=0.8),
            mn.transforms.RandAxisFlipd(keys=keys, prob=0.8),
            mn.transforms.RandAxisFlipd(keys=keys, prob=0.8),
        ]
    tfms += [
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
    # Use MONAI's DataLoader so that list outputs from RandCropByPosNegLabeld
    # (num_samples=1 -> list of length 1) are collated correctly via
    # list_data_collate. Keep batch_size=1, num_workers=0 for GPU memory safety.
    ds = mn.data.Dataset(dicts, transform=transform)
    return mn.data.DataLoader(
        ds, batch_size=1, shuffle=shuffle, num_workers=0,
        collate_fn=list_data_collate,
    )


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
    ap.add_argument("--fade", action="store_true",
                    help="apply smooth lesion intensity fade (penumbra/inhomogeneity) in the synth path")
    ap.add_argument("--synth-prob", type=float, default=0.33,
                    help="probability of drawing a synthetic batch each step "
                         "(default 0.33 = 67%% real, ATLAS-first); "
                         "ignored when --mix-real is not set")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-val", type=int, default=0,
                    help="cap in-training val cases for checkpoint selection; 0 = use full fold")
    ap.add_argument("--val-binarize", choices=("argmax", "threshold"), default="argmax",
                    help="how to binarize lesion predictions for validation checkpoint selection")
    ap.add_argument("--val-threshold", type=float, default=0.5,
                    help="lesion softmax threshold used when --val-binarize=threshold")
    ap.add_argument("--smoke", action="store_true", help="few steps for a quick sanity run")
    ap.add_argument("--requeue", action="store_true", help="(no-op flag; requeue is handled by SIGUSR1 handler)")
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
    print(f"[info] device={device} n_classes={n_classes} mix_real={args.mix_real} "
          f"synth_prob={args.synth_prob:.2f} epochs={args.epochs}")

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

    synth_loader = make_loader(synth_dicts, synth_transform(args.patch, n_classes, device, syn, fade=args.fade), True)
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

    # Seeded RNG for reproducible synth/real selection
    _mix_rng = random.Random(0)
    synth_prob = args.synth_prob

    for epoch in range(start_epoch, epochs):
        cur_epoch = epoch
        # CRITICAL: advance the loss's epoch so DiceCEL2Loss leaves the L2 warmup
        # (epochs < l2_epochs use the L2 objective; >= switch to Dice+CE). Without
        # this, crit.epoch stays 0 and the model trains on L2 forever.
        crit.epoch = epoch
        model.train()
        running = 0.0
        n_ok = 0
        n_nan = 0
        for step in range(epoch_len):
            # Configurable synth:real mix. When --mix-real is not set, always
            # use synth. Otherwise draw synth with probability synth_prob
            # (default 0.33 = 67% real / ATLAS-first).
            if real_iter is not None and _mix_rng.random() >= synth_prob:
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
                    if args.val_binarize == "threshold":
                        pred = (probs[:, lesion_idx] > args.val_threshold).float()
                    else:
                        pred = (probs.argmax(dim=1) == lesion_idx).float()
                    gt = (batch["label"][:, 0] == lesion_idx).float()
                    inter = (pred * gt).sum(); denom = pred.sum() + gt.sum()
                    dices.append(float((2 * inter / denom).cpu()) if denom > 0 else (1.0 if gt.sum() == 0 else 0.0))
            metric = float(np.nanmean(dices)) if dices else 0.0
            print(f"[epoch {epoch}] val_lesion_dice={metric:.4f} "
                  f"(n={len(dices)} binarize={args.val_binarize})", flush=True)
            ck = {"net": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "metric": max(metric, metric_best)}
            torch.save(ck, str(ckpt_last))
            if metric > metric_best:
                metric_best = metric
                torch.save(ck, str(outdir / "checkpoint_best.pt"))
                print(f"[epoch {epoch}] new best {metric_best:.4f}", flush=True)

    print(f"[done] best val lesion dice={metric_best:.4f} ckpt={outdir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
