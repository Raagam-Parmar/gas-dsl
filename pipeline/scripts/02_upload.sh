#!/bin/bash
# Generate submit.sh from the template, then copy PTX + host .cu + submit.sh to the HPC.
set -euo pipefail
source "$(dirname "$0")/config.sh"
for f in "$PTX_FILE" "$HOST_FILE" "$SUBMIT_TEMPLATE"; do
    [ -f "$f" ] || { echo "Missing file: $f"; exit 1; }
done
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
sed -e "s/@KERNEL@/$KERNEL_NAME/g" -e "s/@PTX@/$PTX_NAME/g" "$SUBMIT_TEMPLATE" > "$TMP/submit.sh"

$SSH "$REMOTE" "mkdir -p ~/$REMOTE_DIR"
$SCP "$PTX_FILE" "$HOST_FILE" "$TMP/submit.sh" "$REMOTE:~/$REMOTE_DIR/"
echo "Uploaded $PTX_NAME, $HOST_NAME, submit.sh to $REMOTE:~/$REMOTE_DIR/"