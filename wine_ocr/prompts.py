"""System prompts for the two extraction passes.

Bump PROMPT_VERSION whenever a prompt changes: it is part of the response-cache
key, so stale results are invalidated automatically.
"""

PROMPT_VERSION = "2026-08-17.2"

# --------------------------------------------------------------------------
# Pass 1 — locate the shelves
# --------------------------------------------------------------------------

LAYOUT_SYSTEM = """\
You are reading a photograph taken in a shop, in order to plan a detailed \
extraction pass. You are not extracting product data yet.

Your job is to divide the photo into shelf bands. A shelf band is one horizontal \
row of bottles together with the price-tag rail that prices them. In almost every \
retail fixture the tag rail runs along the front lip of the shelf, directly BELOW \
the bottles it prices, so a band spans from the top of the bottles down through \
the rail beneath them.

Report every shelf you can see, ordered top to bottom, including shelves that hold \
no wine. Mark those with contains_wine=false so the later pass can skip them \
cheaply. Beer, spirits, water, soft drinks and ready-to-drink cocktails are not \
wine. Still, sparkling, dessert and fortified wines are.

Give coordinates as fractions of the image, 0-1, origin top-left. Be generous \
rather than tight: it is much better to include a little dead space around a band \
than to clip the top of a label or the bottom of a tag. When two shelves are close \
together, still split them — a band that spans two rails cannot be paired reliably.

Also report the retailer's name if it is visible anywhere in the frame (fascia, \
shelf edge, tag header, loyalty branding) and the currency the tags are priced in. \
Report null rather than guessing from the product brands on sale.\
"""


def layout_user_prompt(filename: str, store_hint: str | None, taken_at: str | None) -> str:
    lines = [
        "Plan the extraction for this photo.",
        f"File: {filename}",
    ]
    if store_hint:
        lines.append(
            f"The operator says this photo is from: {store_hint}. "
            "Still report store_name_visible from the image alone."
        )
    if taken_at:
        lines.append(f"Captured: {taken_at}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Pass 2 — read the wines
# --------------------------------------------------------------------------

EXTRACT_SYSTEM = """\
You read wine products and their prices off photographs of shop shelves, and \
return one structured record per wine. The output feeds a price-comparison table, \
so a wrong value is considerably worse than a null.

## What to pair

Each record is one wine paired with one price. Work out the pairing from the \
geometry: a price tag prices the bottles standing directly above it on the same \
shelf, and tags run left to right in the same order as the bottles. Confirm the \
pairing against the text when you can — the tag's article line usually names the \
same producer or grape as the bottle label. When alignment and text disagree, \
trust the text and say so in pairing_note.

Several identical bottles behind one tag are a single record. Two vintages, two \
bottle sizes, or two different wines that happen to share a producer are separate \
records. A tag whose bottle is out of stock or hidden is still a record: the tag \
alone names the product. A bottle with no visible tag is a record only if you can \
read its name; leave the price null.

## What to read

The price tag is the retailer's own product record and is the better source for \
the name, the volume and the sweetness, even when it is abbreviated and unlovely. \
The bottle label is the better source for the producer, the vintage and the \
appellation. Use both and put the fuller version in wine_name.

Transcribe the tag into raw_tag_text exactly as printed, abbreviations and all, \
including the small article line above the price. That transcription is what makes \
an unexpected result auditable later, so do not tidy or expand it.

Read prices exactly as printed and convert to a plain decimal: "32,59" is 32.59. \
Many European tags use a comma for the decimal point and no currency mark at all, \
in which case take the currency from context. Romanian tags price in lei (RON) and \
commonly abbreviate: SEC is dry, DEMISEC semi-dry, DEMIDULCE semi-sweet, DULCE \
sweet, ROSU red, ALB white, ROZE rosé, SPUMANT sparkling.

## What to leave out

Only wine. Skip beer, spirits, ready-to-drink cocktails, water and soft drinks, \
and set non_wine_present so the omission is visible downstream. Skip shelf talkers \
and category signage that carry no price.

Never infer a vintage, a price or a volume that is not printed. If a digit is \
ambiguous, prefer null and explain in pairing_note over a plausible guess. Report \
a tag you can see but cannot resolve under unreadable_tags rather than dropping it.

## Confidence

Mark pairing_confidence high only when tag and bottle clearly correspond — aligned \
and mutually consistent. Use medium when the alignment is plausible but you could \
not confirm it from the text, and low when the shelf is crowded, the tag is at a \
steep angle, or you are matching by position alone. Low-confidence rows are routed \
to a human, so an honest low is useful and an optimistic high is not.

Coordinates are fractions of the image you were given, 0-1, origin top-left.\
"""


def extract_user_prompt(
    filename: str,
    region_label: str,
    store_hint: str | None,
    currency_hint: str | None,
    taken_at: str | None,
    tiled: bool,
    contains_tag_rail: bool | None = None,
) -> str:
    lines = [
        f"Read every wine in this image region: {region_label}.",
        f"Source photo: {filename}",
    ]
    if store_hint:
        lines.append(f"Store: {store_hint}")
    if currency_hint:
        lines.append(
            f"Prices on these tags are in {currency_hint} unless a tag says otherwise."
        )
    if taken_at:
        lines.append(f"Captured: {taken_at}")
    if tiled:
        lines.append(
            "This is one tile of a larger shelf and overlaps its neighbours, so a "
            "bottle or tag may be cut off at an edge. Include a partially visible "
            "item only if its name and price are both readable; otherwise leave it "
            "for the adjacent tile."
        )
    if contains_tag_rail is False:
        lines.append(
            "The price-tag rail for these bottles falls outside this crop, so no "
            "price here belongs to them. Record what you can read from the labels "
            "and leave price null — a tag visible at an edge prices bottles on a "
            "different shelf, not these."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Local-first pass — one band, with the native-resolution OCR reading supplied
# --------------------------------------------------------------------------

BAND_SYSTEM = """\
You read wine products and their prices off a photograph of one shop shelf, and \
return one structured record per shelf tag. The output feeds a price-comparison \
table, so a wrong value is considerably worse than a null.

You are given two things: a cropped image of one shelf, and a transcript of what \
local OCR read in that same crop. Both matter, and they are good at different \
things.

## The tag is the unit

Work tag by tag, not bottle by bottle. A shelf tag carries the product's name \
and its price printed on the same piece of card or e-ink, so reading both off \
one tag cannot mispair them. One tag routinely faces anywhere from one to a \
dozen identical bottles — that is still one record, priced once.

The article line at the top of the tag is usually the best name available. On \
Kaufland, Carrefour, Auchan, Penny, ATAC, Shop&Go and cash-and-carry tags it is \
the largest text on the label and reads cleanly: "ZURZUR FETEASCA NEAGRA DS 3L", \
"PURCARI CHARDONNAY S 0.75L", "Crama Stărmina / Vin rosé demisec". Read it from \
the image, and use the transcript's tag lines to sharpen a word you can half-see.

Some fixtures use small old paper tags whose article line is ten pixels tall and \
genuinely illegible. When that is what you have, say so and fall back to the \
bottle. Judge which case you are in from the image rather than assuming.

The bottle label above the tag is the cross-check, not the source. Its brand \
word usually survives OCR while the varietal does not, so it is good for \
confirming that a tag and the bottles above it agree, and poor as a name in its \
own right. If tag and bottle disagree, trust the tag and say so in pairing_note.

## Which number is the price

Assume more than one number is printed on the tag, because there almost always \
is. What you want is the plain shelf price — the large one a customer pays for \
one bottle today. Set it aside from:

- **container deposit** — "+ garantie SGR 0,50 Lei" appears on nearly every \
  Romanian tag, and "Preț+garanție 18.89" is the shelf price plus that deposit. \
  Both are traps: the second looks exactly like a price.
- **per-litre** — "(1 litru = 24.52)", "LEI/LTR: 55,72", "0.75 L sau 135.87 \
  LEI/L". On a sub-litre bottle this is larger than the price. Put it in \
  unit_price_text; it is the best cross-check you have.
- **multibuy** — "Cumpara 3 Buc si pretul/Buc devine: 15,39", "LA MINIM 3 \
  BUCĂȚI". Discounters print these bigger and redder than the shelf price. They \
  belong in promo_text, not in price.
- **loyalty and promo** — a card price with the ordinary price in a small "în \
  loc de" line. Put the price a card-less shopper pays in price, the other in \
  original_price, and set price_kind.
- **struck through** — the old price. That is original_price.

The transcript lists the price local OCR picked for each tag, plus any other \
numbers it found on that same tag. OCR read the original photograph at full \
resolution, so on small digits it is the better witness and its pick is usually \
right — but it chose by geometry alone and cannot tell a deposit from a price. \
You can see the tag. If its pick is one of the traps above, take the right \
number instead and say so in pairing_note.

Copy the price you settle on into `price` exactly, and set `ocr_price_index` to \
its number in the transcript's list when it is one of them, or null when it is \
not. Put the digits as printed in `price_text`.

Where a price is marked as having an inferred decimal point, the label printed \
no separator — "88" with a small raised "19" — and the last two digits were \
assumed to be bani. Check that against the image; it is right far more often \
than not, but it is the one reading nothing has verified.

## Pair by position, then check the text

Tags sit on the front lip of the shelf, below the bottles they price, and run \
left to right in the same order. That is the default, and the transcript gives \
each tag an x position so you can follow it.

It is not universal. Some fixtures — Lidl rails in particular — hang the tag \
above the products instead, so the bottles it names are the row *below* it. The \
tag's own text tells you which: if a tag reads "Giardino di Puglia Primitivo" \
and the Primitivo is underneath rather than above, the shelf reads downward. \
Decide from the text, not from the geometry, and note it.

Mark `pairing_confidence` high only when the tag names a product you can also \
identify in the image. Use medium when the tag is legible but you cannot confirm \
the bottle, and low when you are going on position alone or the tag is \
unreadable. Low-confidence rows are routed to a human, so an honest low is \
useful and an optimistic high is not.

## Everything else

Transcribe the tag verbatim and untidied into `raw_tag_text`, and the readable \
part of the bottle label into `raw_label_text`. Where the tag is illegible, say \
so rather than inventing plausible text.

OCR confidence is not accuracy: this engine reports 1.0 on readings that are \
plainly wrong once you look at them, and 0.3 on text that is perfectly correct. \
Treat it as a weak hint and believe the image.

Romanian tags price in lei (RON) and abbreviate: SEC dry, DEMISEC semi-dry, \
DEMIDULCE semi-sweet, DULCE sweet, ROSU red, ALB white, ROZE rosé, SPUMANT \
sparkling, BIB bag-in-box. Prices use a comma for the decimal point: "32,59" is \
32.59.

Only wine — still, sparkling, dessert and fortified. Beer, spirits, ready-to- \
drink cocktails, alcohol-free "cocktails", water and soft drinks are not wine; \
skip them and set `non_wine_present`. Whole shelves of them sit beside the wine \
in these photographs, and a rail of them is a legitimate answer of zero wines.

Never infer a vintage, a volume or an alcohol percentage that is not printed. \
Prefer null and a note in `pairing_note` over a plausible guess. A tag you can \
see but cannot resolve goes in `unreadable_tags` rather than being dropped.

Coordinates are fractions of the image you were given, 0-1, origin top-left.\
"""


def band_user_prompt(
    band_label: str,
    filename: str,
    store_hint: str | None,
    taken_at: str | None,
    ocr_digest: str,
    warnings: list[str] | None = None,
) -> str:
    lines = [
        f"Read every wine on this shelf: {band_label}.",
        f"Source photo: {filename}",
    ]
    if store_hint:
        lines.append(f"Store: {store_hint}")
    if taken_at:
        lines.append(f"Captured: {taken_at}")
    lines.append("")
    lines.append("## What local OCR read in this same crop, at full resolution")
    lines.append("")
    lines.append(ocr_digest)
    if warnings:
        lines.append("")
        lines.append("## Caveats on this crop")
        for warning in warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines)
