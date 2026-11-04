#!/bin/bash
# Copy job logs and nsys profile back into Pipeline/output/<timestamp>_job<id>/.
set -uo pipefail
source "$(dirname "$0")/config.sh"
JOB_ID=$(cat "$JOBID_FILE")
DEST="$OUTPUT_DIR/$(date +%Y%m%d_%H%M%S)_job${JOB_ID}"
mkdir -p "$DEST"

$SCP "$REMOTE:~/$REMOTE_DIR/job.$JOB_ID.out" "$REMOTE:~/$REMOTE_DIR/job.$JOB_ID.err" "$DEST/" 2>/dev/null
$SCP "$REMOTE:~/$REMOTE_DIR/${KERNEL_NAME}_profile.*" "$DEST/" 2>/dev/null || echo "(no profile files)"

echo "Results saved to $DEST"
echo "=== STDOUT ==="; cat "$DEST/job.$JOB_ID.out" 2>/dev/null
echo "=== STDERR ==="; cat "$DEST/job.$JOB_ID.err" 2>/dev/null