#!/bin/bash
# One-time environment setup on USC CARC (Discovery).
#
#   git clone https://github.com/Soorya2201/school-facilities-cv.git
#   cd school-facilities-cv && bash slurm/setup_carc.sh
#
# Work out of /scratch1 rather than $HOME: the home quota is small and the
# HuggingFace cache alone is tens of gigabytes.
set -euo pipefail

ENV_NAME="${ENV_NAME:-schoolcv}"
SCRATCH_DIR="/scratch1/$USER/schoolcv"

echo "=== modules ==="
module purge
module load conda
# Only needed for the GPU stage; harmless if absent on a login node.
module load cuda/12.4 2>/dev/null || echo "  (cuda module not available here — fine for CPU stages)"

echo "=== conda env: $ENV_NAME ==="
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  conda create -y -n "$ENV_NAME" python=3.11
fi
# `conda activate` needs the shell hook in a non-interactive script.
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=== CPU pipeline dependencies ==="
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "=== scratch layout ==="
mkdir -p "$SCRATCH_DIR/hf" "$SCRATCH_DIR/cache" logs
cat > .env.carc <<ENVEOF
# source this before running anything on CARC
module purge && module load conda && eval "\$(conda shell.bash hook)" && conda activate $ENV_NAME
export HF_HOME="$SCRATCH_DIR/hf"
export SCHOOLCV_CACHE="$SCRATCH_DIR/cache"
export TOKENIZERS_PARALLELISM=false
ENVEOF

echo "=== verifying the offline path (no network, no keys, no GPU) ==="
python -m src.run --offline stage-c
python tests/test_pipeline.py

cat <<'DONE'

Setup complete.

  source .env.carc                 # every new shell

CPU only (works on a login node):
  python -m src.run --offline stage-c

GPU stage — install the heavy dependencies first, on a node with internet:
  pip install torch transformers huggingface_hub accelerate
  pip install vllm                 # needed for token logprobs
  sbatch slurm/stage_b.sh

Whole pipeline as one chained submission:
  bash slurm/submit_all.sh
DONE
