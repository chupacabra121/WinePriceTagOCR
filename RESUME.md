# Stopping and resuming the run

Nothing needs to be shut down gracefully. Every answer the reading pass produces
is written to its own file the moment it is finished, so closing the laptop
loses only the photos that happened to be in flight — and those come back on the
next pass because they were never marked done.

## To stop

Close the laptop, or quit the session. That is all.

There is no state anywhere except files on disk:

| | |
|---|---|
| `.cache/vision/` | every photo's OCR, keyed by image hash. Never re-read. |
| `out/work/manifest.jsonl` | the full job list — 2337 shelves across 443 photos |
| `out/work/crops/`, `out/work/briefs/` | what each reading agent is given |
| `out/work/answers/` | **the run's progress.** One file per finished shelf. |
| `out/table/` | the CSVs and `wines.xlsx`, rebuilt from the answers |

## To see where it got to

```bash
wine-ocr status --work out/work
```

## To resume

Start a Claude Code session in this folder and say:

> Continue the wine OCR run — resume the reading pass where it left off.

It will run the two commands below itself. The first prints the work list, which
contains only photos that still have unanswered shelves; the second hands that
list to `tools/read_shelves.workflow.js`.

```bash
wine-ocr briefs --work out/work --compact
```

Re-running the whole prep is safe and cheap if you ever want to — the OCR is
cached, so it finishes in about a minute and produces identical job ids, which
means every answer already on disk still counts.

## To get the table at any point

```bash
wine-ocr collect --work out/work --out out/table --root "Mystery shopping"
```

Local, free, and safe to run mid-flight — it simply includes more rows each
time. `tools/refresh_table.sh` does this on a loop.
