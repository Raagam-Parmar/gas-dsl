#!/bin/bash
#SBATCH --job-name=@KERNEL@
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

# Generated from submit.template.sh by 02_upload.sh (@KERNEL@, @PTX@ filled in).
# ./@KERNEL@ was already compiled on the login node.
# NOTE: do NOT `module load cuda/11.3` here: it shadows the working nsys.

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,compute_cap --format=csv

echo "=== Run (PASSED / FAILED) ==="
./@KERNEL@ @PTX@
echo "@KERNEL@ exit code: $?"

echo "=== Profile (nsys) ==="
if command -v nsys >/dev/null 2>&1; then
    nsys profile \
        --force-overwrite true \
        --stats=true \
        --trace=cuda,nvtx,osrt \
        -o @KERNEL@_profile \
        ./@KERNEL@ @PTX@ || echo "nsys profiling failed (the run above is still valid)"
else
    echo "nsys not found, skipping profiling"
fi