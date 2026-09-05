#!/bin/bash
# Wait for the agent lane to go quiet, then sweep the remainder through the API.
#
# Both lanes derive their work from which answer files exist, so running them at
# the same time on the same photos means each claims the same bands and the API
# is billed for shelves the agents were getting free. Waiting for quiet is the
# cheap way to avoid that.
set -u
cd /Users/tandrei/WinePriceTagOCR

prev=-1; quiet=0
for i in $(seq 1 60); do
  n=$(ls out/work/answers 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" = "$prev" ]; then quiet=$((quiet+1)); else quiet=0; fi
  echo "$(date +%H:%M) answers=$n quiet_ticks=$quiet"
  [ "$quiet" -ge 3 ] && break     # ~3 min with no new answer = agents done
  prev=$n
  sleep 60
done

left=$(.venv/bin/python -m wine_ocr status --work out/work 2>/dev/null | awk '/Pending/{print $2}')
echo "agents finished; $left bands left for the API"

if [ "${left:-0}" -gt 0 ]; then
  .venv/bin/python -m wine_ocr read --work out/work \
      --model claude-opus-5 --effort medium --max-tokens 40000 --workers 8 \
      2>&1 | grep -E 'band\(s\)|Bands read|Failed|Approx cost|tokens'
fi

.venv/bin/python -m wine_ocr collect --work out/work --out out/table --root "Mystery shopping" 2>&1 | tail -7
.venv/bin/python -m wine_ocr status --work out/work 2>&1 | head -3
echo "FINISHED $(date +%H:%M)"
