#!/bin/bash
# Compile the host harness on the LOGIN node (nvcc needs the CUDA module there).
set -uo pipefail
source "$(dirname "$0")/config.sh"
echo "Compiling $HOST_NAME -> $KERNEL_NAME on login node (arch=$ARCH, $CUDA_MODULE)..."
$SSH "$REMOTE" "
    cd ~/$REMOTE_DIR &&
    module unload gnu12 &&
    module load $CUDA_MODULE &&
    nvcc -arch=$ARCH $HOST_NAME -o $KERNEL_NAME \
         -L\$(dirname \$(dirname \$(which nvcc)))/lib64/stubs -lcuda
"
if [ $? -ne 0 ]; then echo "Compile failed"; exit 1; fi
echo "Compile OK"