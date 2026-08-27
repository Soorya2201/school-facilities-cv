#!/bin/bash
#SBATCH --job-name=schoolcv-stageB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/stageB_%A_%a.out
#
# Stage B -- GPU node, fully OFFLINE.
#
# One model family per array task. The array is not just for speed: it isolates
# failures, so if the 32B VLM OOMs, the solar segmenter and the detector still
# land and the pipeline still produces a complete (if less confident) result.
set -euo pipefail

module purge
module load conda cuda/12.4
source activate schoolcv

export SCRATCH_DIR="/scratch1/$USER/schoolcv"
export HF_HOME="$SCRATCH_DIR/hf"
export SCHOOLCV_CACHE="$SCRATCH_DIR/cache"
export HF_HUB_OFFLINE=1            # weights were staged in Stage A
export TOKENIZERS_PARALLELISM=false
mkdir -p logs

TASKS=(solar owl vlm)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Stage B task: $TASK on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ "$TASK" == "vlm" ]]; then
  # vLLM is required rather than plain transformers because it returns token
  # logprobs -- that is what makes P(yes) a probability instead of a number the
  # model wrote down.
  MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
  python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --dtype bfloat16 \
      --max-model-len 8192 --limit-mm-per-prompt image=2 \
      --gpu-memory-utilization 0.92 --port 8000 &
  VLLM_PID=$!
  trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
  # Wait for readiness rather than sleeping a fixed amount.
  for _ in $(seq 1 120); do
    curl -sf http://localhost:8000/v1/models >/dev/null && break
    sleep 10
  done
  python -m src.tier1_gpu --task vlm --vlm-model "$MODEL"
else
  python -m src.tier1_gpu --task "$TASK" --device cuda --tta 8
fi

echo "Stage B task $TASK complete -> preds/tier1_${TASK}.jsonl"
