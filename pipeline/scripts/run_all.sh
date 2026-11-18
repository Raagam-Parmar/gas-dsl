#!/bin/bash
# Run all PTX and Host code pairs on the Bhavani HPC cluster
set -uo pipefail

D="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$(cd "$D/.." && pwd)"
PTX_DIR="$PIPELINE_DIR/ptx"
HOST_DIR="$PIPELINE_DIR/host"

echo "=========================================================="
echo " Running all PTX + Host benchmarks on Bhavani Cluster"
echo "=========================================================="

FAILED=()
PASSED=()

for ptx in "$PTX_DIR"/*.ptx; do
    [ -f "$ptx" ] || continue
    name="$(basename "$ptx" .ptx)"
    host="$HOST_DIR/${name}_host.cu"

    if [ ! -f "$host" ]; then
        echo "[-] Skipping $name: Host file $host not found."
        continue
    fi

    echo ""
    echo "=========================================================="
    echo " Starting: $name ($ptx, $host)"
    echo "=========================================================="

    if bash "$D/run.sh" "$ptx" "$host"; then
        PASSED+=("$name")
    else
        echo "[!] Error running $name"
        FAILED+=("$name")
    fi
done

echo ""
echo "=========================================================="
echo " Execution Summary"
echo "=========================================================="
echo "Successful (${#PASSED[@]}): ${PASSED[*]:-None}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed (${#FAILED[@]}): ${FAILED[*]}"
    exit 1
fi
