#!/bin/bash
# Single place to edit. Sourced by every script (runs on your laptop).
# No password stored. Use key auth:  ssh-copy-id s112301019-unnikrishnan@192.168.1.133
# or export SSHPASS (needs `sshpass`):  read -rs SSHPASS && export SSHPASS

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- WHAT TO RUN (edit these, or pass as args to run_all.sh) ----------------
PTX_FILE="${PTX_FILE:-$PIPELINE_DIR/ptx/vecadd.ptx}"
HOST_FILE="${HOST_FILE:-$PIPELINE_DIR/host/vecadd_host.cu}"
# Name of the compiled binary, remote folder and profile. Defaults to the PTX file name.
KERNEL_NAME="${KERNEL_NAME:-$(basename "$PTX_FILE" .ptx)}"

# ---------------- HPC settings ----------------
HPC_USER="s112301019-unnikrishnan"
HPC_HOST="192.168.1.133"          # institute network only
ARCH="sm_80"                      # A30
CUDA_MODULE="cuda/11.3"
REMOTE_DIR="pipeline/$KERNEL_NAME"   # relative to remote home

# ---------------- Derived (don't edit) ----------------
PTX_NAME="$(basename "$PTX_FILE")"
HOST_NAME="$(basename "$HOST_FILE")"
SUBMIT_TEMPLATE="$PIPELINE_DIR/scripts/submit.template.sh"
OUTPUT_DIR="$PIPELINE_DIR/output"

SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ControlMaster=auto -o ControlPath=/tmp/hpc-%r@%h:%p -o ControlPersist=10m"
if [ -n "${SSHPASS:-}" ] && command -v sshpass >/dev/null 2>&1; then
    SSH="sshpass -e ssh $SSH_OPTS"
    SCP="sshpass -e scp $SSH_OPTS"
else
    SSH="ssh $SSH_OPTS"
    SCP="scp $SSH_OPTS"
fi
REMOTE="$HPC_USER@$HPC_HOST"
