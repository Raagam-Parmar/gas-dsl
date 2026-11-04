#!/bin/bash
# Submit the Slurm job, then poll squeue, printing state + reason each time.
# Ctrl-C cancels the remote job so it doesn't linger in the queue.
set -uo pipefail
source "$(dirname "$0")/config.sh"

JOB_ID=$($SSH "$REMOTE" "cd ~/$REMOTE_DIR && rm -f job.*.out job.*.err ${KERNEL_NAME}_profile.* && sbatch --parsable submit.sh")
if [ -z "$JOB_ID" ]; then echo "Submit failed"; exit 1; fi
echo "$JOB_ID" > "$JOBID_FILE"
echo "Job ID: $JOB_ID"

trap 'echo; echo "Interrupted: cancelling job $JOB_ID"; $SSH "$REMOTE" "scancel $JOB_ID"; exit 130' INT

LAST=""
while true; do
    # %T = state (PENDING/RUNNING/...), %r = reason, %M = elapsed time
    STATUS=$($SSH "$REMOTE" "squeue -j $JOB_ID -h -o '%T (%r) elapsed=%M'" 2>/dev/null)
    [ -z "$STATUS" ] && break
    if [ "$STATUS" != "$LAST" ]; then
        echo "[$(date +%H:%M:%S)] $STATUS"
        LAST="$STATUS"
    fi
    sleep 3
done
echo "Job $JOB_ID finished."