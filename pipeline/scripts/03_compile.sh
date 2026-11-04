#!/bin/bash
# Compile the host harness on the LOGIN node (nvcc is only available there
# after loading the CUDA module), exactly like run_job.sh does.
set -uo pipefail
source "$(dirname "$0")/config.sh"
echo "Compiling on login node (arch=$ARCH, $CUDA_MODULE)..."
$SSH "$REMOTE" "
    cd ~/$REMOTE_DIR &&
    module unload gnu12 &&
    module load $CUDA_MODULE &&
    nvcc -arch=$ARCH vecadd_host.cu -o $KERNEL_NAME \
         -L\$(dirname \$(dirname \$(which nvcc)))/lib64/stubs -lcuda
"
if [ $? -ne 0 ]; then echo "Compile failed"; exit 1; fi
echo "Compile OK"