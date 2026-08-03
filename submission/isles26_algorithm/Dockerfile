# ISLES'26 submission container.
#
# Structure follows the organizers' official template
# (github.com/ezequieldlrosa/isles26-docker-template): Grand Challenge runs the
# algorithm as an HTTP server on port 4743, polling /health until it returns 200
# and then calling POST /invoke once per case. app.py provides that server;
# inference.py provides init_model() and run().
#
# cuda12.6 is required for the T4 instances Grand Challenge uses. The 'runtime'
# image is sufficient -- we compile no custom kernels.
FROM --platform=linux/amd64 pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

# Required: Grand Challenge uses this label to detect the invoke API.
LABEL org.grand-challenge.api-method="invoke"

ENV PYTHONUNBUFFERED=1 \
    nnUNet_compile=f \
    nnUNet_raw=/tmp/nnunet_raw \
    nnUNet_preprocessed=/tmp/nnunet_preprocessed \
    nnUNet_results=/tmp/nnunet_results \
    MPLCONFIGDIR=/tmp/mpl

# nnUNet_compile=f is not optional: torch.compile has been unreliable for us and
# a compile failure at test time would take down the whole submission. The three
# nnUNet_* paths are read at import time even for pure inference; they point at
# scratch dirs that are never used.

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

# A venv inheriting system site-packages (torch, cuda libs) avoids PEP 668
# externally-managed-environment errors on recent PyTorch images.
RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt && \
    mkdir -p /tmp/nnunet_raw /tmp/nnunet_preprocessed /tmp/nnunet_results /tmp/mpl && \
    chown -R user:user /tmp/nnunet_raw /tmp/nnunet_preprocessed /tmp/nnunet_results /tmp/mpl

# nnU-Net's trainer discovery imports EVERY module under variants/ and does not
# catch ImportError, so primus_trainers.py (timm -> torchvision) can break all
# discovery. Neutralise it preventatively.
RUN python - <<'PY'
import pathlib, nnunetv2
p = pathlib.Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer" / "variants" / "primus" / "primus_trainers.py"
if p.is_file():
    p.write_text('"""Neutralised: pulls timm->torchvision, which nnU-Net trainer discovery imports unconditionally and cannot skip on ImportError."""\n')
    print(f"neutralised {p}")
PY

COPY --chown=user:user app.py inference.py geometry.py test_geometry.py /opt/app/

# Weights are baked in rather than mounted from Grand Challenge's separate
# model.tar.gz. The image grows ~600 MB, but the container is self-contained and
# there is no second upload to go stale against a hard deadline.
#
# `model/` is NOT in git. CI populates it from the public Hugging Face release
# and verifies every sha256 before this build; a local build must do the same or
# the COPY fails loudly rather than producing a weightless container.
COPY --chown=user:user model/ /opt/ml/model/

# Verify the weights are what we published. A truncated or stale download yields
# a container that starts, segments, and returns plausible nonsense -- the same
# silent-failure class as a mirror flip.
RUN python - <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path("/opt/ml/model")
mf = root / "hf_manifest.json"
if not mf.is_file():
    sys.exit("hf_manifest.json missing from the weights: refusing to build")
manifest = json.loads(mf.read_text())
bad = []
for rel, meta in manifest["files"].items():
    f = root / rel
    if not f.is_file():
        bad.append(f"{rel}: missing"); continue
    h = hashlib.sha256()
    with f.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != meta["sha256"]:
        bad.append(f"{rel}: sha256 mismatch")
if bad:
    sys.exit("weight verification FAILED:\n  " + "\n  ".join(bad))
print(f"verified {len(manifest['files'])} weight files, {manifest['total_bytes']/1e6:.1f} MB")
PY

# Fail the build if native-space orientation handling regresses. Real ATLAS cases
# are not in the build context, so this runs the synthetic all-orientation and
# NIfTI round-trip checks only.
RUN python test_geometry.py --per-session-csv /nonexistent

USER user
EXPOSE 4743
ENTRYPOINT ["python", "app.py"]
