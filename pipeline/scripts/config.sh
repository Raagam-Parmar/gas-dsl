#!/bin/bash
# Shared settings. Source this from the other scripts.

HPC_USER="s112301019-unnikrishnan"
HPC_HOST="192.168.1.133"          # institute network only
REMOTE_DIR="pipeline/vecadd"      # relative to remote home
KERNEL_NAME="vecadd"
ARCH="sm_80"                      # A30
CUDA_MODULE="cuda/11.3"           # same module as the working run_job.sh

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PTX_FILE="$PIPELINE_DIR/ptx/vecadd.ptx"
HOST_FILE="$PIPELINE_DIR/host/vecadd_host.cu"
SUBMIT_FILE="$PIPELINE_DIR/scripts/submit.sh"
OUTPUT_DIR="$PIPELINE_DIR/output"
JOBID_FILE="$PIPELINE_DIR/.last_job_id"

# Reuse one SSH connection so you only authenticate once per run.
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ControlMaster=auto -o ControlPath=/tmp/hpc-%r@%h:%p -o ControlPersist=10m"

if [ -n "${SSHPASS:-}" ] && command -v sshpass >/dev/null 2>&1; then
    SSH="sshpass -e ssh $SSH_OPTS"
    SCP="sshpass -e scp $SSH_OPTS"
else
    SSH="ssh $SSH_OPTS"
    SCP="scp $SSH_OPTS"
fi
REMOTE="$HPC_USER@$HPC_HOST"