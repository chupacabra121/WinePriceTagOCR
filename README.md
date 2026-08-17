# WinePriceTagOCR

Point it at a folder of shop photos, get back a spreadsheet of wines and prices.

It reads the wine name off the bottle label *and* the price tag, pairs each
bottle with the tag that prices it, normalises the result, and writes a table
with the store, the price, the price per litre and a provenance trail for every
row.

```
wine-ocr extract data/photos --out out
```

```
out/wines.xlsx          Wines · Needs review · Duplicates · Unmatched tags · Errors
out/wines.csv           the same flat table
out/unmatched_tags.csv  tags that were seen but could not be tied to a wine
out/extractions.jsonl   raw model output, one record per photo
```

---

## Why two passes

The photos this was built against are 4284×5712 iPhone shots of a full chiller
bay. The vision model downscales anything longer than 2576px on its long edge,
which for these is a 2.2× reduction. Measured on a real tag rail:

| | prices | wine names on the tag |
|---|---|---|
| whole photo, downscaled to 2576px | readable | **illegible** |
| shelf band cropped at native resolution | readable | readable |

The wine name lives in ~10px of small print at the top of each tag. It does not
survive the downscale, so one call per photo cannot work for shelf shots. The
pipeline therefore:

1. **Locates the shelves** — one cheap call on the whole photo at 1568px returns
   a band per shelf (bottle row + the tag rail beneath it), plus the store name
   and currency, and marks shelves that hold no wine so they are never read again.
2. **Reads each band at native resolution** — every band is cropped and, if it
   still exceeds 2576px on either axis, split into overlapping tiles so that
   *nothing sent to the model is ever downscaled*. A band keeps bottles and
   their tags in one frame, which is what makes the pairing checkable.

Splitting is horizontal wherever possible, because a vertical cut can separate
bottles from the tags that price them. When a band is too tall to avoid one
(a close-up, or a single band covering most of the frame) the bottom row keeps
the tag rail, and the tiles above it are told that no price in view belongs to
their bottles — otherwise a tag clipped at the edge invites a confident, wrong
pairing.

`--tiling whole` skips all of this and makes one call per photo. It is ~6× cheaper
and fine when the tags are large in frame (a single bottle, or a close-up of a
rail); on a wide shelf shot it will get prices without reliable names.

## Quickstart — run it on your own machine

Photos stay local. Only code lives on GitHub.

```bash
git clone https://github.com/chupacabra121/WinePriceTagOCR.git
cd WinePriceTagOCR
pip install -e .

wine-ocr init "Annabella, Kaufland, Mega Image"   # folders + config + .env
# put your key in .env, then copy each shop's photos into its folder

wine-ocr estimate data/photos                     # what it will cost
wine-ocr extract data/photos                      # do it
```

HEIC works as-is, so iPhone photos need no conversion.

### Why photos do not belong in the repo

| | |
|---|---|
| A phone photo | ~3.5 MB |
| 500 of them | ~1.7 GB |
| GitHub's hard per-file limit | 100 MB |
| GitHub's soft repo limit | ~5 GB |

Beyond size, git stores a full copy of every version forever, so re-shooting a
shelf grows the repo permanently even if you delete the old file. Git is built
for text it can diff; a JPEG is an opaque blob it can only accumulate.

`.gitignore` therefore excludes `data/photos/`, `out/` and `.env`. The four
sample photos under `data/samples/` are tracked deliberately, as test fixtures.

If you do want your photos backed up, use iCloud, Drive, or an external disk —
not version control. If you want them off your machine and into a run, upload
them to a Claude conversation instead.

### Putting a folder structure into git (when you do want to)

The gotcha that trips everyone: **git cannot track an empty directory.** It
tracks files, and infers folders from their paths, so an empty `data/photos/`
simply will not commit. The fix is a placeholder file — this repo ships
`data/photos/.gitkeep` and `out/.gitkeep` for exactly that reason, and
`wine-ocr init` creates them for any folder it makes.

To add a folder of files:

```bash
git add data/photos/annabella   # or: git add -A
git commit -m "Add photos"
git push
```

Or drag the folder onto the GitHub web UI, which preserves relative paths but
caps out at 100 files and 25 MB per file. GitHub Desktop is the friendliest
option if you would rather not use the terminal.

## Organising photos

Put each store in its own folder. The folder name becomes the store.

```
data/photos/
├── annabella/IMG_5755.HEIC
├── kaufland/…
└── mega-image/…
```

`wine-ocr init "Annabella, Kaufland"` creates these, slugifying names to match
the aliases in `config/stores.yaml` (`Mega Image` → `mega-image`). A folder with
no alias still works — it just gets prettified back (`some-new-shop` → `Some New
Shop`).

Store attribution is resolved in this order, and the losing candidates are kept
in the row rather than discarded:

| Precedence | Source | `store_source` |
|---|---|---|
| 1 | `--store "Annabella"` on the command line | `cli` |
| 2 | an alias in `config/stores.yaml` | `folder-map` |
| 3 | the folder name, prettified | `folder` |
| 4 | the retailer name read off the photo | `photo` |
| 5 | none of the above → `Unknown` | `none` |

A folder beats the photo read because a folder is a deliberate statement, while
signage is often partly occluded. Every row also carries
`store_read_from_photo`, so a wrong call is visible and fixable in the sheet.

Copy `config/stores.example.yaml` to `config/stores.yaml` to map messy folder
names (`anabela`, `annabella-rm-valcea`) onto one canonical store.

## Usage

```bash
# Create folders, config and .env
wine-ocr init "Annabella, Kaufland"

# See what a run would cost before spending anything
wine-ocr estimate data/photos

# See how a photo will be cut up — no API calls
wine-ocr plan data/samples/annabella/IMG_5755.HEIC --out /tmp/tiles

# Normal run
wine-ocr extract data/photos --out out

# One store, cheap mode, write crops of anything doubtful
wine-ocr extract data/photos/annabella --store Annabella --tiling whole --crops

# Big library: half price, runs in the background for minutes to hours
wine-ocr extract data/photos --batch

# Add a new shopping trip to an existing table
wine-ocr extract data/photos/new-trip --append

# Check flagged rows against the crops they came from (opens in a browser)
wine-ocr review out/wines.csv --photos data/photos

# Score an extraction against hand-read ground truth
wine-ocr verify out/wines.csv
```

`init`, `estimate`, `plan`, `review` and `verify` make no API calls at all.

Useful flags: `--effort low|medium|high|xhigh|max` (default `high`),
`--model`, `--workers`, `--limit N`, `--no-cache`.

## What you get per row

Identification: `wine_name`, `producer`, `vintage`, `wine_type`, `sweetness`,
`grape_varieties`, `region`, `country`, `volume_ml`, `abv_percent`.

Price: `price`, `currency`, `price_per_litre`, `price_kind`, `original_price`,
`discount_pct`, `promo_text`.

Trust: `needs_review`, `review_reasons`, `pairing_confidence`, `price_check`,
`unit_price_check`, `name_source`, plus `raw_tag_text` and `raw_label_text` —
verbatim transcriptions, so any surprising row can be audited without
re-opening the photo.

Provenance: `store`, `store_source`, `photo`, `shelf`, `photo_taken_at`, GPS,
bounding boxes, `photo_sha256`, `model`, `extracted_at`.

### Three cross-checks

Extraction errors that look plausible in a spreadsheet are the dangerous ones,
so three independent checks run on every row:

- **`price_check`** re-parses the verbatim `price_text` and compares it to the
  number. This catches decimal-separator misreads — necessary here, because a
  single Annabella rail carries both `58.99` and `54,59`.
- **`unit_price_check`** compares the computed price-per-litre against the
  RON/L figure the retailer prints in the tag's small print. Agreement confirms
  the price *and* the volume were both read correctly; a clean 0.75 ratio
  implicates the volume specifically.
- **cross-sighting agreement** — when the same wine turns up in two photos at
  two prices, the row is flagged with both values rather than silently taking one.

Anything that fails, or that the model marked low-confidence, lands on the
**Needs review** sheet with a reason. Nothing is dropped: unresolvable tags go
to `unmatched_tags.csv`, and duplicate sightings are linked via `duplicate_of`
rather than deleted.

## Checking the output

Two commands close the loop, and neither costs anything to run.

**`wine-ocr review`** builds a self-contained HTML sheet putting each row's crop
next to the values read from it. A wrong pairing is obvious the moment you see a
Terra Romana bottle labelled as a Cotnari — which is not obvious at all in a
spreadsheet cell. Flagged rows only by default; `--all` for everything.

**`wine-ocr verify`** scores an extraction against
`data/samples/annabella/ground_truth.csv` — the top tag rail of `IMG_5755`,
eleven tags transcribed by hand at native resolution. It reports:

```
Prices found              ?/11
Names correct             ?/10 of tags found
Volumes correct           ?/11
Rows outside ground truth n  (not scored)
```

**These numbers are unmeasured so far** — the scorer is tested against synthetic
extractions, but nothing has been scored against real model output yet.

Ground truth covers one rail, so scoring is recall over that rail; rows from
other shelves are listed but never counted against you. Matching is on price —
the field that can be transcribed by eye with near-certainty — and a matched row
is then checked for a distinctive name token, which is what separates a good
read from one that got the digits right and the product wrong.

This is the number to watch on the first live run, and again after any prompt
change.

## Cost

Measured on the four sample photos (upper bound; ignores caching):

| Mode | Calls / photo | Cost / photo | With `--batch` |
|---|---|---|---|
| `--tiling auto` (default) | ~7 | ~$0.34 | ~$0.17 |
| `--tiling whole` | 1 | ~$0.05 | ~$0.03 |

Three things bring the real figure down: the layout pass skips shelves with no
wine, the system prompt is cached across every call in a run, and every response
is cached on disk by image hash. Re-running after changing output columns or
review rules costs nothing.

Run `wine-ocr estimate` on your own photos for a real number.

## Sample photos

`data/samples/annabella/` holds four real shots from an Annabella store in
Romania: two of a chiller bay and two of a wine display, each scene photographed
twice from slightly different angles — which is what the cross-photo dedup is
for. They are the fixture the geometry and end-to-end tests run against.

## Development

```bash
pip install -e ".[dev]"
pytest
```

102 tests, no API key required. The end-to-end test drives the whole pipeline
against a stubbed client using a real sample photo, so the only thing untested
without a key is the model's answer itself. `test_tiles_are_never_downscaled`
and `test_images_reach_the_model_at_native_resolution` are regression guards on
the resolution finding above.

```
wine_ocr/
├── cli.py         extract / estimate / plan commands
├── extract.py     the two passes, prompt caching, concurrency, Batch API
├── images.py      HEIC, EXIF, and the band/tile geometry
├── models.py      Pydantic schemas — the field descriptions are the prompt
├── prompts.py     system prompts (bump PROMPT_VERSION to invalidate the cache)
├── normalize.py   prices, volumes, vintages, the cross-checks
├── output.py      rows, dedup, CSV/Excel/JSONL writers
├── stores.py      store attribution
├── schema.py      Pydantic → structured-outputs JSON Schema
├── review.py      the HTML review sheet
├── verify.py      ground-truth scoring
└── cache.py       on-disk response cache
```

## Known limits

- **Not yet run against the live API.** Every offline stage is tested, but the
  prompts have not been tuned against real model output. Expect a first pass to
  need prompt adjustment — `PROMPT_VERSION` exists for exactly that loop.
- Bottle-to-tag pairing is hardest on crowded shelves where bottles are wider
  than their tags; those rows should come back `medium` or `low` confidence.
- Vintages are frequently absent from Romanian shelf tags and unreadable on
  angled labels, so `vintage` will often be null.
- Enrichment from external sources (grape, region, ratings for wines whose label
  does not state them) is deliberately out of scope — the table records what is
  in the photo.
