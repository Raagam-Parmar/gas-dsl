#!/bin/bash
# Usage: bash scripts/run_all.sh [ptx_file] [host_cu_file]
#   no args -> uses the defaults in config.sh
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
[ $# -ge 1 ] && export PTX_FILE="$(realpath "$1")"
[ $# -ge 2 ] && export HOST_FILE="$(realpath "$2")"
bash "$D/01_ssh_check.sh"   || exit 1
bash "$D/02_upload.sh"      || exit 1
bash "$D/03_compile.sh"     || exit 1
bash "$D/04_run_job.sh"     || exit 1
bash "$D/05_fetch_results.sh"