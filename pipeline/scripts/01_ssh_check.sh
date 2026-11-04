#!/bin/bash
# Verify we can reach the HPC and create the remote working dir.
set -euo pipefail
source "$(dirname "$0")/config.sh"
$SSH "$REMOTE" "hostname && mkdir -p ~/$REMOTE_DIR && echo 'Remote dir ready: ~/$REMOTE_DIR'"
