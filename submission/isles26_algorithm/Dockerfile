# ISLES'26 Grand Challenge algorithm container.
#
# The model weights are NOT baked in. Grand Challenge uploads them separately as
# a model.tar.gz which is mounted at /opt/ml/model at run time, which keeps this
# image small and means a weight change does not require a rebuild.
#
# NOTE ON WHERE THIS CAN BE BUILT: not on Sherlock. The cluster has no docker,
# podman, buildah or skopeo, and its apptainer cannot emit the `docker save`
# tarball Grand Challenge requires. Build on a workstation or a cheap cloud VM;
# no GPU is needed to build or to run the local test.
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# Grand Challenge invokes the container through this label.
LABEL org.grand-challenge.api-method="invoke"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    nnUNet_compile=f \
    nnUNet_raw=/tmp/nnunet_raw \
    nnUNet_preprocessed=/tmp/nnunet_preprocessed \
    nnUNet_results=/tmp/nnunet_results \
    MPLCONFIGDIR=/tmp/mpl \
    OMP_NUM_THREADS=2

# `nnUNet_compile=f` is not optional: torch.compile is disabled deliberately
# because nnU-Net's inductor path has been unreliable for us, and a compile
# failure at test time would take down the whole submission.

# nnU-Net reads these three env vars at import time and raises if unset, even
# for pure inference. They point at scratch paths that are never used.

RUN groupadd -r algorithm && useradd -m -r -g algorithm algorithm

WORKDIR /opt/algorithm

COPY --chown=algorithm:algorithm requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    mkdir -p /tmp/nnunet_raw /tmp/nnunet_preprocessed /tmp/nnunet_results /tmp/mpl && \
    chown -R algorithm:algorithm /tmp/nnunet_raw /tmp/nnunet_preprocessed /tmp/nnunet_results /tmp/mpl

# nnU-Net's trainer discovery imports every module under
# training/nnUNetTrainer/variants/, and primus_trainers.py pulls timm ->
# torchvision. If that import fails the discovery itself dies rather than
# skipping the module, so neutralise it. This is the same fix analysis_nnunet_01
# applies on the cluster; here it is preventative.
RUN python - <<'PY'
import pathlib, nnunetv2
p = pathlib.Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer" / "variants" / "primus" / "primus_trainers.py"
if p.is_file():
    p.write_text('"""Neutralised: pulls timm->torchvision, which nnU-Net trainer discovery imports unconditionally and cannot skip on ImportError."""\n')
    print(f"neutralised {p}")
PY

COPY --chown=algorithm:algorithm geometry.py inference.py ./

USER algorithm

ENTRYPOINT ["python", "/opt/algorithm/inference.py"]
