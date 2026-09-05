#!/bin/bash
# Read every remaining shelf through the API, store by store.
#
# One store at a time, and never the store the already-running Kaufland process
# owns: `pending` is derived from which answer files exist, so two readers
# pointed at the same store would both claim the same bands and the API would
# be paid twice for one shelf.
set -u
cd /Users/tandrei/WinePriceTagOCR
SP=/private/tmp/claude-501/-Users-tandrei-WinePriceTagOCR/b4826c39-75ff-4373-b95b-5c0453d344d6/scratchpad

while IFS=$'\t' read -r n store; do
  [ -z "${store:-}" ] && continue
  echo "=== $store ($n bands) $(date +%H:%M) ==="
  .venv/bin/python -m wine_ocr read --work out/work \
      --model claude-opus-5 --effort medium --max-tokens 40000 \
      --workers 12 --store "$store" 2>&1 | grep -E 'band\(s\)|Bands read|Failed|Approx cost'
done < "$SP/stores.txt"

# Kaufland's own process has exited by now; sweep whatever it left behind.
echo "=== final sweep $(date +%H:%M) ==="
.venv/bin/python -m wine_ocr read --work out/work \
    --model claude-opus-5 --effort medium --max-tokens 40000 --workers 12 \
    2>&1 | grep -E 'band\(s\)|Bands read|Failed|Approx cost'

.venv/bin/python -m wine_ocr collect --work out/work --out out/table \
    --root "Mystery shopping" 2>&1 | tail -6
echo "ALL DONE $(date +%H:%M)"
