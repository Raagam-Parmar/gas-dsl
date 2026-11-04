#!/bin/bash
# Copy the PTX, host .cu harness and Slurm script to the HPC.
set -euo pipefail
source "$(dirname "$0")/config.sh"
$SSH "$REMOTE" "mkdir -p ~/$REMOTE_DIR"
$SCP "$PTX_FILE" "$HOST_FILE" "$SUBMIT_FILE" "$REMOTE:~/$REMOTE_DIR/"
echo "Uploaded to $REMOTE:~/$REMOTE_DIR/"
