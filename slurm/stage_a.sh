#!/bin/bash
#SBATCH --job-name=schoolcv-stageA
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/stageA_%j.out
#
# Stage A -- CPU node WITH internet. No GPU is requested on purpose.
#
# This stage does every network-dependent thing: imagery, Overpass, the
# free-tier APIs, and pre-staging the HuggingFace weights. CARC GPU nodes have
# restricted outbound network, and holding an A100 while waiting on a
# rate-limited HTTP call would waste the scarce resource to save the abundant
# one.
set -euo pipefail

module purge
module load conda
source activate schoolcv

export SCRATCH_DIR="/scratch1/$USER/schoolcv"
export HF_HOME="$SCRATCH_DIR/hf"
export SCHOOLCV_CACHE="$SCRATCH_DIR/cache"
mkdir -p "$HF_HOME" "$SCHOOLCV_CACHE" logs

# --- pre-stage weights while we still have network -------------------------
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
# Solar segmenter: DeepLabV3-ResNet101 fine-tuned on BDAPPV.
for f in ("model.py", "deeplab_google_best.pth", "deeplab_ign_best.pth"):
    hf_hub_download("gabrielkasmi/bdappv-models", f)
# Small specialists: together well under 10 GB.
for repo in ("google/owlv2-large-patch14-ensemble",
             "IDEA-Research/grounding-dino-base",
             "facebook/sam2.1-hiera-large"):
    snapshot_download(repo)
print("weights staged")
PY

# --- fetch imagery + OSM, run the deterministic extractors -----------------
python -m src.run stage-a

echo "Stage A complete. preds/ and cache/ are populated for the GPU stage."
