#!/bin/bash
#SBATCH --job-name=vecadd
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

# ./vecadd was already compiled on the login node.
# NOTE: do NOT `module load cuda/11.3` here: it shadows the working nsys.

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,compute_cap --format=csv

echo "=== Run (PASSED / FAILED) ==="
./vecadd vecadd.ptx
echo "vecadd exit code: $?"

echo "=== Profile (nsys) ==="
if command -v nsys >/dev/null 2>&1; then
    nsys profile \
        --stats=true \
        --trace=cuda,nvtx,osrt \
        -o vecadd_profile \
        ./vecadd vecadd.ptx || echo "nsys profiling failed (the run above is still valid)"
else
    echo "nsys not found, skipping profiling"
fi