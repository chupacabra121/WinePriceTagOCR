"""System prompts for the two extraction passes.

Bump PROMPT_VERSION whenever a prompt changes: it is part of the response-cache
key, so stale results are invalidated automatically.
"""

PROMPT_VERSION = "2026-08-13.1"

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
            "This is one tile of a wider shelf and overlaps its neighbours, so a "
            "bottle or tag may be cut off at the left or right edge. Include a "
            "partially visible item only if its name and price are both readable; "
            "otherwise leave it for the adjacent tile."
        )
    return "\n".join(lines)
