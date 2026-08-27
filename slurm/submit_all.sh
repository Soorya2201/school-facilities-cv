#!/bin/bash
# Chain the three stages with dependencies so the whole run is one command.
set -euo pipefail
mkdir -p logs
A=$(sbatch --parsable slurm/stage_a.sh)
B=$(sbatch --parsable --dependency=afterok:"$A" slurm/stage_b.sh)
C=$(sbatch --parsable --dependency=afterany:"$B" slurm/stage_c.sh)
echo "Stage A: $A"
echo "Stage B: $B  (array, after A)"
echo "Stage C: $C  (after B; 'afterany' so a single failed model does not"
echo "              block the rest of the pipeline from producing output)"
