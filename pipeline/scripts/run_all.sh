#!/bin/bash
# Full pipeline: check -> upload -> compile (login node) -> sbatch -> fetch.
set -uo pipefail
D="$(dirname "$0")"
bash "$D/01_ssh_check.sh"   || exit 1
bash "$D/02_upload.sh"      || exit 1
bash "$D/03_compile.sh"     || exit 1
bash "$D/04_run_job.sh"     || exit 1
bash "$D/05_fetch_results.sh"