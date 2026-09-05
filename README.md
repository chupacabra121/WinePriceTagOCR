# WinePriceTagOCR

Point it at a folder of shop photos, get back a spreadsheet of wines and prices.

Local OCR reads the photo at full resolution and works out where the shelves
are; a vision model then does the part that needs judgement — which bottle a
price belongs to, and what that bottle is. The result is a table with the store,
the price, the price per litre and a provenance trail for every row, written back
out in the shape of your photo folders.

```
wine-ocr prep "Mystery shopping"      # local: read every photo, cut out the shelves
wine-ocr briefs                       # the work list for the reading pass
wine-ocr collect                      # local: assemble the table
```

```
out/Hypermarket - Kaufland/Hypermarket - Kaufland.csv   one CSV per photo folder
out/index.csv                                           row counts and median price per folder
out/wines.csv                                           the same rows, flat
out/wines.xlsx                    Wines · Needs review · Duplicates · Unmatched tags · Errors
out/work/                         crops, OCR digests, and one answer file per shelf
```

---

## What the photographs will and will not give you

The design follows from measurements on the library itself — 4284x5712 phone
shots of Romanian shop shelves — plus a survey in which ten agents each examined
four photos from a different group of chains and magnified the tags by eye.

**Local OCR reads the prices, and it is free.** Apple's Vision framework runs
offline on the native frame in about a second and a half. On the hand-read rail
in `data/samples/annabella/` it returned all eleven prices at maximum
confidence. It supports `ro-RO`, so Romanian diacritics survive.

**The name is on the tag, not on the bottle** — in almost every chain. On
Kaufland, Carrefour, Auchan, Penny, ATAC, Shop&Go and the cash-and-carry
fixtures the article line is the *largest* text on the label and reads cleanly:
`ZURZUR FETEASCA NEAGRA DS 3L`, `PURCARI CHARDONNAY S 0.75L`. Bottle labels are
the weaker source: the brand word survives OCR but the varietal that
distinguishes the SKU does not (`DEALONILE HOLDONE` for *Dealurile Moldovei*),
and three different Jidvei wines side by side all come back as `JIDVEI`.

The exception is old-style paper tags, like Annabella's, whose article line is
about ten pixels tall. Magnified 4x at native resolution it reads
`TERA ROMANA VSTI 1 7L` — the information is not in the pixels, and no amount of
cropping recovers it. There the bottle is the fallback. Which case a photo is in
is something the model decides from the crop, not something assumed.

**Reading name and price off the same tag avoids the pairing problem
entirely.** One tag faces anywhere from one to twelve bottles, so matching tags
to bottles by position is unreliable — Auchan measured 4 of 4 wrong on a clean
shot. Two values printed on one piece of e-ink cannot be mispaired.

**Assume more than one number per tag.** Every chain surveyed prints at least
two: a container deposit (`+ garantie SGR 0,50 Lei`), a per-litre figure, a
multibuy tier, a loyalty price, a struck-through original. Kaufland's
`Preț+garanție` total is a well-formed price exactly 0.50 above the real one.
Several chains print no decimal separator at all — `88` large, a small raised
`19` beside it.

**Measured end to end.** Scored against the hand-transcribed rail with
`wine-ocr verify`: **11/11 prices, 10/10 names, 11/11 volumes.** The names come
off the tag — `MUSCAT OTTONEL ACADEMICIAN ALB DEMISEC 0.75L` where the ground
truth only asserted "ACADEMICIAN".

Put together: prices come from local OCR at native resolution, names come from
the shelf tag, and the model is asked only for what neither can do — telling a
tag's price from its four decoys, and saying which wine the tag is for.

## How a photo becomes rows

**Prep is local and free.** OCR runs at native resolution, tiled at 1200px
because Vision downscales its own input too. The tag rails then fall out of the
geometry: a rail is a horizontal run of price-shaped text at a common height, so
clustering the prices finds the shelves without a single model call. Each rail
plus the bottles standing above it becomes a *band*, and each band is written
out as a JPEG crop next to a text digest of what OCR read there.

Rail detection has to survive real shop fixtures, so:

| | |
|---|---|
| Shelves slope in an angled shot | prices are chained left to right, with the tolerance growing over distance, so the chain follows the slope instead of snapping |
| Carrefour, Kaufland and Mega Image use electronic labels | `88` with a small raised `19` reaches OCR as `8819`; a run-together number on a rail is read as lei-and-bani, and the row is flagged so the model checks it |
| A snack rack stands beside the wine bay | a band's top edge is the nearest rail *that overlaps it horizontally*, so an unrelated fixture cannot crop the bottles out |
| Lidl prints `1 L = 53,32 Lei` under the price | a number markedly shorter than the rail's own digits is small print, not a price |
| `1827` and `1958` are wines, not prices | a run of inferred prices must stretch across the frame like a rail; two of them on neighbouring bottles is a coincidence |
| A shelf's prices are unreadable | any stretch of the photo no rail claimed is still cropped and read, so a shelf is never dropped silently |

**Reading is the only part that needs a model.** One agent per photo opens each
of its band crops and returns one record per price. It is given the OCR digest
alongside the image, which is what lets it work from a downscaled crop without
losing the prices: the digest carries the native-resolution reading the image no
longer shows. The instruction is explicit that the price comes from the digest
and the name comes from the bottle.

Because the work list is a directory of files, the run is resumable by
construction. Every answer is a file; `wine-ocr status` counts what is left and
`wine-ocr briefs` hands back only the photos that still need reading, so an
overnight run that is interrupted picks up exactly where it stopped.

**Collect is local and free.** Answers are validated, cross-checked, deduplicated
and written out. A malformed answer is not repaired or discarded — the job simply
stays pending and comes back on the next pass.

## Running without an API key

The reading pass is a set of instruction files and image paths, so it does not
have to go through the API at all. `wine-ocr briefs` prints a work list that a
Claude Code agent run can execute directly:

```bash
wine-ocr prep "Mystery shopping" --work out/work
wine-ocr briefs --work out/work > work.json      # one entry per photo still to read
# hand work.json to tools/read_shelves.workflow.js
wine-ocr status --work out/work                  # how much is left
wine-ocr collect --work out/work --out out --root "Mystery shopping"
```

`wine-ocr extract` still drives the Anthropic API directly if you would rather
have a key do the work; everything else is identical.

### Running it overnight

The reading pass is the slow half, and it is chunked by design. Ask for a batch
of photos, read them, repeat — each round only ever sees what is still
unanswered, so nothing is done twice and stopping is free:

```bash
wine-ocr briefs --work out/work --limit 40      # the next 40 unanswered photos
# ... read them ...
wine-ocr status --work out/work                 # 312 of 443 photos still to go
```

Answers land in `out/work/answers/` as they are produced. A run killed halfway
leaves every answer it had already written, and `wine-ocr collect` will build a
table from whatever is there — so there is a usable spreadsheet at every point,
not only at the end.

Ordering is smallest-first: photos with fewest shelves go first, so an
interrupted run has finished whole photos rather than leaving many half-read.


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

The local-first pipeline — no API key, three commands:

```bash
# Read every photo locally: OCR, shelf detection, crops, briefs
wine-ocr prep "Mystery shopping" --work out/work

# Show what local OCR saw on one photo, and the shelves it derived
wine-ocr ocr "Mystery shopping/Hypermarket - Kaufland/IMG_3448.HEIC"

# The work list for the reading pass — only photos not yet answered
wine-ocr briefs --work out/work
wine-ocr briefs --work out/work --store Kaufland --limit 20

# How much is left
wine-ocr status --work out/work

# Build the table from whatever has been answered
wine-ocr collect --work out/work --out out --root "Mystery shopping"
```

Driving it through the Anthropic API instead:

```bash
wine-ocr init "Annabella, Kaufland"     # folders, config and .env
wine-ocr estimate data/photos           # what a run would cost
wine-ocr extract data/photos --out out  # do it
wine-ocr extract data/photos --batch    # half price, slower
```

Checking the result, either way:

```bash
wine-ocr review out/wines.csv --photos "Mystery shopping"   # crops beside values
wine-ocr verify out/wines.csv                               # score vs ground truth
```

Everything except `extract` makes no API calls at all.

Useful flags: `--max-tile` (local OCR resolution), `--store`, `--limit N`,
`--no-cache`; and for `extract`, `--effort low|medium|high|xhigh|max`,
`--model`, `--workers`.

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

Measured on 2026-08-17, running the local-first pipeline over the four sample
photos:

```
Prices found              11/11 (100%)
Names correct             10/10 (100%) of tags found
Volumes correct           11/11 (100%)
Rows outside ground truth 44  (not scored)
```

The names are the part worth looking at, because they are what a price table is
useless without: `MUSCAT OTTONEL ACADEMICIAN ALB DEMISEC 0.75L`, `DOMENIILE
SAMBURESTI FETEASCA NEAGRA 0.75 L`, `CASTEL HUNIADE VIN MERLOT + PINOT NOIR`.
All of them came off the tag rather than the bottle.

Ground truth covers one rail, so scoring is recall over that rail; rows from
other shelves are listed but never counted against you. Matching is on price —
the field that can be transcribed by eye with near-certainty — and a matched row
is then checked for a distinctive name token, which is what separates a good
read from one that got the digits right and the product wrong.

This is the number to watch on the first live run, and again after any prompt
change.

## Cost

The local-first pipeline replaces the old two-vision-pass design, which spent
about seven model calls per photo. Prep now does the shelf-finding for nothing:

| | Model calls | Time |
|---|---|---|
| `wine-ocr prep` on 448 photos | 0 | ~20 min of local CPU |
| Reading those photos | 1 turn per photo | depends on concurrency |
| `wine-ocr collect` | 0 | seconds |

Local OCR is cached on disk by image hash and prompt version, so re-running prep
after a geometry change re-reads nothing. Re-running `collect` after changing
output columns or review rules is free by construction — it never calls anything.

`wine-ocr estimate` still gives a dollar figure for the API path.

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

168 tests, no API key required, and none of them are stubs where it matters:
`tests/test_prep.py` runs the real Vision helper over the real sample photos, so
the geometry is measured rather than mocked. `test_the_ground_truth_rail_is_found_in_full`
is the regression guard on the finding this design rests on — that local OCR
returns every price on a hand-transcribed rail.

The Swift helper is compiled on first use, so `swiftc` (Xcode command line
tools) is the only extra requirement. macOS only: Vision has no equivalent
elsewhere, and the API path in `extract.py` is the fallback on other platforms.

```
wine_ocr/
├── cli.py         prep / briefs / status / collect / ocr, and the API commands
├── vision.py      local OCR through Apple Vision, cached by image hash
├── layout.py      rails, bands, tag boxes — the shelf geometry, no model calls
├── prep.py        crops, digests, per-photo briefs, the resumable manifest
├── collect.py     answers back into rows, refereed against the OCR prices
├── mirror.py      one CSV per source folder, plus the index
├── extract.py     the API path: two vision passes, prompt caching, Batch API
├── images.py      HEIC, EXIF, and the crop geometry
├── models.py      Pydantic schemas — the field descriptions are the prompt
├── prompts.py     system prompts (bump PROMPT_VERSION to invalidate the cache)
├── normalize.py   prices, volumes, vintages, the cross-checks
├── output.py      rows, dedup, CSV/Excel/JSONL writers
├── stores.py      store attribution
├── schema.py      Pydantic → structured-outputs JSON Schema
├── review.py      the HTML review sheet
├── verify.py      ground-truth scoring
└── cache.py       on-disk cache, shared by the OCR and API paths

tools/
├── visionocr.swift          the Vision helper
└── read_shelves.workflow.js the reading pass, one agent per photo
```

## Known limits

These come from a survey of the photo library itself: ten agents each examined
four photos from a different group of Romanian chains, magnified tags by eye,
and reported where the design breaks. Most of what follows is their findings.

- **Every chain prints more than one number per tag.** A container deposit
  (`+ garantie SGR 0,50 Lei`), a per-litre figure, multibuy tiers, a loyalty
  price, a struck-through original. Kaufland's `Preț+garanție` total is the
  nastiest: a well-formed price exactly 0.50 above the real one. Local OCR
  filters the obvious decoys and hands the rest to the model with their wording;
  a tag whose largest number is a three-for price can still be got wrong.
- **Several chains print the price with no decimal separator** — `88` with a
  small raised `19`. The separator is inferred two digits from the end, which is
  right for Romanian retail but is a reading nothing has independently verified.
  Those rows say so in the digest and are worth spot-checking.
- **A tag can face a dozen bottles.** One tag is one row, so a shelf of twelve
  identical facings yields one price, not twelve — but where several SKUs share
  a facing block, which bottle a tag refers to is genuinely ambiguous.
- **Lidl hangs its rail above the products.** The pipeline assumes tags price the
  bottles above them and tells the model to check the tag's own text for the
  exception, but a Lidl shelf read purely on geometry will be off by one row.
- **OCR confidence is not accuracy.** The engine reports 1.0 on readings that are
  plainly wrong when magnified, and 0.3 on text that is perfectly correct. It is
  used as a weak hint only; nothing is filtered on it alone.
- **The raised bani digits are the real risk, not the missing separator.** The
  reading pass found the large lei figure almost always right and the small
  raised pair often wrong in a consistent way — 9 read as 0 or 3, 4 read as 1:
  `37 49` came back as `37.40`, `28 99` as `28.90`, `44 99` as `44.00`. The
  `DECIMAL POINT INFERRED` warning flags the right tags for the wrong reason.
- **A tag's printed left border becomes a leading 1.** `34,49` reads as
  `134.49`, `30,49` as `130.40`, `56,49` as `156.49`. On a rail whose other
  tags are two-digit, a price in the 100-199 band is worth re-reading.
- **Campaign tags print three prices in exact ratio.** Profi's "MULTI PROFIT
  APP" 2+1 tags carry the campaign per-bottle price, the three-bottle total at
  twice the shelf price, and the shelf price itself; La Cocos and Supeco print
  similar tiers. Collection flags any price with a 2x or one-third sibling on
  the same rail, but the pick itself still needs the image.
- **Panoramas produce fabricated numbers.** Stitch ghosting on the four Kaufland
  panoramas turned a 15.19 tag into `155,49`. They should be shot again as
  ordinary frames.
- Vintages are frequently absent from Romanian shelf tags and unreadable on
  angled labels, so `vintage` will often be null.
- Enrichment from external sources (grape, region, ratings for wines whose label
  does not state them) is deliberately out of scope — the table records what is
  in the photo.
