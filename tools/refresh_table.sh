#!/bin/bash
# Rebuild the output table from whatever the reading pass has answered so far.
# Collect is local and idempotent, so this is safe to run mid-run: it simply
# includes more rows each time.
cd /Users/tandrei/WinePriceTagOCR
for i in $(seq 1 40); do
  answered=$(ls out/work/answers 2>/dev/null | wc -l | tr -d ' ')
  .venv/bin/python -m wine_ocr collect --work out/work --out out/table \
      --root "Mystery shopping" >/dev/null 2>&1
  rows=$(( $(wc -l < out/table/wines.csv) - 1 ))
  echo "$(date +%H:%M)  answers=$answered  rows=$rows"
  sleep 900
done
