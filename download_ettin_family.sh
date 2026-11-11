#!/usr/bin/env bash
# Download the full Ettin model family (encoders + decoders) from HuggingFace.
set -uo pipefail
HF=/mnt/local-disk/decision-models/rdt/.venv-hf312/bin/hf
DEST=/mnt/local-disk/decision-models/models
SIZES="17m 32m 68m 150m 400m 1b"
FAILED=(); OK=()
for size in $SIZES; do
  for repo in "ettin-encoder-$size" "ettin-decoder-$size" "ettin-enc-from-dec-$size" "ettin-dec-from-enc-$size"; do
    echo "=== [$repo] $(date +%H:%M:%S)"
    if $HF download "jhu-clsp/$repo" --local-dir "$DEST/$repo" >/dev/null 2>&1; then
      OK+=("$repo"); echo "    OK"
    else
      FAILED+=("$repo"); echo "    FAILED"
      # retry once, showing errors
      if $HF download "jhu-clsp/$repo" --local-dir "$DEST/$repo" 2>&1 | tail -2; then
        OK+=("$repo (retry)")
      fi
    fi
  done
done
echo "=== SUMMARY $(date)"
echo "OK: ${#OK[@]}"
printf '  %s\n' "${OK[@]}"
echo "FAILED: ${#FAILED[@]:-0}"
printf '  %s\n' "${FAILED[@]:-none}"
