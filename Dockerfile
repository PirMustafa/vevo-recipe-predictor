# Vevo Recipe Predictor -- GPU container
#
# WHY A GPU IMAGE AT ALL: Stage 1 (TabICL) and Stage 2 both hard-require CUDA
# (src/stage1_classifier.py:79, src/stage2_regressor.py:125,131). TabICL is an
# in-context learner -- it re-processes the whole 2051-row training set on
# every predict() call. On GPU that is ~60-90s; on CPU it is ~26 MINUTES. A
# CPU-only container will appear to work and then time out in production.
#
# TARGET HARDWARE: the installed torch is 2.11.0+cu128, whose kernels cover
# sm_75, sm_80, sm_86, sm_90, sm_100, sm_120. That means:
#   g5.xlarge   (A10G, sm_86)  -- RECOMMENDED, closest to the dev GPU
#   g4dn.xlarge (T4,   sm_75)  -- works, roughly half the speed
#   g6.xlarge   (L4,   sm_89)  -- NOT SUPPORTED by this build. Do not use
#                                 without rebuilding torch.
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch MUST come from the CUDA index first. A plain `pip install torch`
# pulls a CPU-only or mismatched-CUDA wheel, and the app silently degrades
# from ~90s to ~26min per prediction. See requirements.txt.
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Source only. Models are NOT baked in -- they are 576 MB and change
# independently of the code. They are fetched at start (see entrypoint.sh)
# so a model swap does not require an image rebuild.
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY app.py predict.py train_backing_modes.py ./
COPY sample_spectrum.json ./
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE 5000

# Fails if CUDA is missing, rather than silently falling back to a 26-minute
# CPU path. Long start period: loading 213 MB of models takes ~60s.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import torch,sys,urllib.request; \
sys.exit(0 if torch.cuda.is_available() and \
urllib.request.urlopen('http://127.0.0.1:5000/api/meta',timeout=5).status==200 else 1)"

ENTRYPOINT ["./entrypoint.sh"]
