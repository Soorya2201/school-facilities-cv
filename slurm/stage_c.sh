#!/bin/bash
#SBATCH --job-name=schoolcv-stageC
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/stageC_%j.out
#
# Stage C -- fuse, calibrate, write outputs. Seconds of CPU; no GPU, no network.
set -euo pipefail

module purge
module load conda
source activate schoolcv
export SCHOOLCV_CACHE="/scratch1/$USER/schoolcv/cache"
mkdir -p logs

python -m src.run stage-c
echo "Stage C complete -> out/measurements.csv, out/metrics.json, out/reliability.png"
