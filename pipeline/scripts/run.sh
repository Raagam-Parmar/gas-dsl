#!/bin/bash
# Full pipeline in one script: check -> upload -> compile -> run job -> fetch.
# Usage: bash scripts/run.sh [ptx_file] [host_cu_file]
#   no args -> uses the defaults in config.sh
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
[ $# -ge 1 ] && export PTX_FILE="$(realpath "$1")"
[ $# -ge 2 ] && export HOST_FILE="$(realpath "$2")"
source "$D/config.sh"

# ---------------- 1. SSH check ----------------
echo "[1/5] Checking SSH connection..."
$SSH "$REMOTE" "hostname && mkdir -p ~/$REMOTE_DIR && echo 'Remote dir ready: ~/$REMOTE_DIR'" || exit 1

# ---------------- 2. Upload ----------------
echo "[2/5] Uploading..."
for f in "$PTX_FILE" "$HOST_FILE" "$SUBMIT_TEMPLATE"; do
    [ -f "$f" ] || { echo "Missing file: $f"; exit 1; }
done
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
sed -e "s/@KERNEL@/$KERNEL_NAME/g" -e "s/@PTX@/$PTX_NAME/g" "$SUBMIT_TEMPLATE" > "$TMP/submit.sh"
$SCP "$PTX_FILE" "$HOST_FILE" "$TMP/submit.sh" "$REMOTE:~/$REMOTE_DIR/" || exit 1
echo "Uploaded $PTX_NAME, $HOST_NAME, submit.sh to $REMOTE:~/$REMOTE_DIR/"

# ---------------- 3. Compile (login node) ----------------
echo "[3/5] Compiling $HOST_NAME -> $KERNEL_NAME (arch=$ARCH, $CUDA_MODULE)..."
$SSH "$REMOTE" "
    cd ~/$REMOTE_DIR &&
    module unload gnu12 &&
    module load $CUDA_MODULE &&
    nvcc -arch=$ARCH $HOST_NAME -o $KERNEL_NAME \
         -L\$(dirname \$(dirname \$(which nvcc)))/lib64/stubs -lcuda
" || { echo "Compile failed"; exit 1; }
echo "Compile OK"

# ---------------- 4. Submit + wait ----------------
echo "[4/5] Submitting job..."
JOB_ID=$($SSH "$REMOTE" "cd ~/$REMOTE_DIR && rm -f job.*.out job.*.err ${KERNEL_NAME}_profile.* && sbatch --parsable submit.sh")
[ -z "$JOB_ID" ] && { echo "Submit failed"; exit 1; }
echo "Job ID: $JOB_ID"
trap 'echo; echo "Interrupted: cancelling job $JOB_ID"; $SSH "$REMOTE" "scancel $JOB_ID"; rm -rf "$TMP"; exit 130' INT

LAST=""
while true; do
    STATUS=$($SSH "$REMOTE" "squeue -j $JOB_ID -h -o '%T (%r) elapsed=%M'" 2>/dev/null)
    [ -z "$STATUS" ] && break
    if [ "$STATUS" != "$LAST" ]; then echo "[$(date +%H:%M:%S)] $STATUS"; LAST="$STATUS"; fi
    sleep 3
done
echo "Job $JOB_ID finished."

# ---------------- 5. Fetch results ----------------
echo "[5/5] Fetching results..."
DEST="$OUTPUT_DIR/$(date +%Y%m%d_%H%M%S)_job${JOB_ID}"
mkdir -p "$DEST"
$SCP "$REMOTE:~/$REMOTE_DIR/job.$JOB_ID.out" "$REMOTE:~/$REMOTE_DIR/job.$JOB_ID.err" "$DEST/" 2>/dev/null
$SCP "$REMOTE:~/$REMOTE_DIR/${KERNEL_NAME}_profile.*" "$DEST/" 2>/dev/null || echo "(no profile files)"
echo "Results saved to $DEST"
echo "=== STDOUT ==="; cat "$DEST/job.$JOB_ID.out" 2>/dev/null
echo "=== STDERR ==="; cat "$DEST/job.$JOB_ID.err" 2>/dev/null
