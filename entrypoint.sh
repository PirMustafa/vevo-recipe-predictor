#!/usr/bin/env bash
# Container entrypoint. Two jobs: refuse to start without a GPU, and fetch
# the models before serving.
set -euo pipefail

echo "== Vevo Recipe Predictor =="

# 1. Fail loudly on a missing GPU.
#
# This check exists because the failure is otherwise INVISIBLE. Without CUDA,
# TabICL still works -- it just takes ~26 minutes per prediction instead of
# ~90 seconds. The app starts, answers /api/meta, passes a naive health check,
# and then every real request times out. Better to refuse to boot.
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit(
        "FATAL: no CUDA device.\n"
        "  This image REQUIRES a GPU. On CPU a single prediction takes ~26\n"
        "  minutes instead of ~90 seconds, so the service would appear to\n"
        "  start and then time out on every request.\n"
        "  Run on g5.xlarge (A10G) or g4dn.xlarge (T4), with the NVIDIA\n"
        "  container runtime enabled (--gpus all)."
    )
name = torch.cuda.get_device_name(0)
cap  = "sm_%d%d" % torch.cuda.get_device_capability(0)
archs = torch.cuda.get_arch_list()
print(f"  GPU: {name} ({cap})")
if cap not in archs:
    sys.exit(
        f"FATAL: this torch build has no kernels for {cap}.\n"
        f"  Supported: {', '.join(a for a in archs if a.startswith('sm_'))}\n"
        f"  {cap} is most likely an L4 (g6 instances). Use g5 or g4dn, or\n"
        f"  rebuild torch for this architecture."
    )
PY

# 2. Fetch models if they are not already present.
#
# Models live in S3 rather than in the image: they are 576 MB, and they change
# on a different cadence from the code. Baking them in would mean a full image
# rebuild and push for every retrain.
MODELS_DIR="${MODELS_DIR:-/app/models}"
if [ ! -f "${MODELS_DIR}/stage1_classifier.joblib" ]; then
  if [ -z "${MODELS_S3_URI:-}" ]; then
    echo "FATAL: models not found at ${MODELS_DIR} and MODELS_S3_URI is unset." >&2
    echo "  Set MODELS_S3_URI=s3://your-bucket/vevo-models/ or mount a volume." >&2
    exit 1
  fi
  echo "  fetching models from ${MODELS_S3_URI} ..."
  mkdir -p "${MODELS_DIR}"
  aws s3 sync "${MODELS_S3_URI}" "${MODELS_DIR}" --only-show-errors
fi

# The light-mode model is the production path per the client brief (light
# spectral data only, M0 only). src/backing_modes.py pins it to the
# wolves_v1 variant; if that tree is missing, light mode silently falls back
# to the superseded 2%-threshold model, which scores 59% instead of 88% on
# the client's actual question. Check for it explicitly.
if [ ! -f "${MODELS_DIR}/wolves_v1/backing_modes/light/stage1.joblib" ]; then
  echo "WARNING: wolves_v1 light model missing. Light mode will fall back to" >&2
  echo "  the superseded 2%-threshold model (59% vs 88% on 0.1% labels)." >&2
fi

echo "  models ready: $(du -sh "${MODELS_DIR}" 2>/dev/null | cut -f1)"

# app.py defaults to 127.0.0.1, which inside a container means "reachable by
# nothing". Bind to all interfaces here so the port can be published.
#
# This is safe ONLY because of how the instance is reached: no inbound port is
# open on the security group, and access is via an SSM Session Manager tunnel
# authenticated by IAM. The app itself has NO authentication of any kind, so
# if this container is ever placed behind a public load balancer or given a
# public IP, that must be paired with an authenticating proxy first.
export HOST=0.0.0.0
export PORT="${PORT:-5000}"

echo "  binding ${HOST}:${PORT} (reachable only through the SSM tunnel)"
echo "  starting server (first load takes ~60s for 213 MB of models)"
exec python app.py
