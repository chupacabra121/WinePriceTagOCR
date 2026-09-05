"""Pydantic schemas for structured extraction.

Design constraints imposed by the Claude structured-outputs feature:

* Every field must be *required*; optionality is expressed as ``Optional[X]``
  (i.e. a nullable type), never as a Python default. A field with a default is
  emitted as non-required and the API rejects that.
* Numeric/length constraints (``ge``, ``le``, ``max_length`` ...) are not
  supported by the schema compiler, so they are deliberately absent. Validating
  ranges client-side would only turn a slightly-out-of-range model answer into a
  hard parse failure, which is worse than clamping later.
* ``description`` on each field is the highest-leverage part of this file: it is
  the field-level instruction the model actually reads.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

WineType = Literal[
    "red", "white", "rose", "sparkling", "dessert", "fortified", "unknown"
]
Sweetness = Literal["dry", "semi_dry", "semi_sweet", "sweet", "unknown"]
PriceKind = Literal["shelf", "promo", "member", "clearance", "unknown"]
NameSource = Literal["price_tag", "bottle_label", "both", "inferred"]
Confidence = Literal["high", "medium", "low"]


class BBox(BaseModel):
    """Axis-aligned box, normalised 0-1 against the image it was read from.

    Origin is top-left. Used only to cut review crops, so approximate is fine.
    """

    x0: float = Field(description="Left edge, 0-1 fraction of image width.")
    y0: float = Field(description="Top edge, 0-1 fraction of image height.")
    x1: float = Field(description="Right edge, 0-1 fraction of image width.")
    y1: float = Field(description="Bottom edge, 0-1 fraction of image height.")


# --------------------------------------------------------------------------
# Pass 1 — layout
# --------------------------------------------------------------------------


class ShelfBand(BaseModel):
    """One shelf: a row of bottles plus the price-tag rail that prices them."""

    label: str = Field(
        description="Short human label, e.g. 'shelf 2 of 5, red wines'."
    )
    contains_wine: bool = Field(
        description=(
            "True if this shelf holds wine bottles (still, sparkling, dessert or "
            "fortified). False for beer, spirits, water, soft drinks or "
            "ready-to-drink cocktails."
        )
    )
    bottles_y0: float = Field(
        description="Top of the bottle row, 0-1 fraction of image height."
    )
    bottles_y1: float = Field(
        description="Bottom of the bottle row (where bottles meet the shelf)."
    )
    tags_y0: float = Field(
        description=(
            "Top of the price-tag rail for this shelf, 0-1. Tag rails normally sit "
            "on the front lip directly BELOW the bottles they price. If no rail is "
            "visible, repeat bottles_y1."
        )
    )
    tags_y1: float = Field(
        description="Bottom of the price-tag rail, 0-1. If none visible, repeat bottles_y1."
    )
    x0: float = Field(description="Left edge of the shelf, 0-1 fraction of image width.")
    x1: float = Field(description="Right edge of the shelf, 0-1 fraction of image width.")


class PhotoLayout(BaseModel):
    """Cheap first-pass read of a photo: what is in it and where the shelves are."""

    photo_kind: Literal[
        "shelf", "single_bottle", "tag_only", "not_retail", "unreadable"
    ] = Field(description="Overall character of the photo.")
    store_name_visible: Optional[str] = Field(
        description=(
            "Retailer name if it appears anywhere in the photo (signage, shelf "
            "branding, tag header, loyalty logo). Null if not visible. Do not guess "
            "from product brands."
        )
    )
    currency_guess: Optional[str] = Field(
        description=(
            "ISO 4217 code inferred from the tags or signage, e.g. RON, EUR, USD. "
            "Null if there is no evidence."
        )
    )
    bands: List[ShelfBand] = Field(
        description=(
            "One entry per shelf, top to bottom. Include shelves that hold no wine "
            "with contains_wine=false rather than omitting them."
        )
    )
    notes: Optional[str] = Field(
        description="Anything that will hamper extraction: glare, blur, occlusion, steep angle."
    )


# --------------------------------------------------------------------------
# Pass 2 — wines
# --------------------------------------------------------------------------


class WineEntry(BaseModel):
    """One wine product paired with one price."""

    wine_name: str = Field(
        description=(
            "Full product name as printed. Prefer the price tag's wording, which is "
            "the retailer's own product record, and expand it with the bottle label "
            "where the label is clearer. Keep the original language; do not translate."
        )
    )
    producer: Optional[str] = Field(
        description="Winery, domain or brand, e.g. 'Cotnari', 'Purcari', 'Jidvei'. Null if unclear."
    )
    vintage: Optional[str] = Field(
        description=(
            "Four-digit vintage year exactly as printed, or 'NV' when the label says "
            "non-vintage. Null if no year is visible. Never infer a year."
        )
    )
    wine_type: WineType = Field(description="Colour/style category.")
    sweetness: Sweetness = Field(
        description=(
            "Sweetness if stated. Romanian tags use SEC=dry, DEMISEC=semi_dry, "
            "DEMIDULCE=semi_sweet, DULCE=sweet."
        )
    )
    grape_varieties: List[str] = Field(
        description="Grapes named on tag or label, e.g. ['Feteasca Neagra']. Empty list if none."
    )
    region: Optional[str] = Field(description="Wine region or appellation if printed.")
    country: Optional[str] = Field(description="Country of origin if printed or unambiguous.")
    volume_ml: Optional[float] = Field(
        description="Bottle volume in millilitres, e.g. 750 for '0.75L'. Null if not printed."
    )
    abv_percent: Optional[float] = Field(description="Alcohol by volume as a number, e.g. 13.5.")

    price: Optional[float] = Field(
        description=(
            "Price as a decimal number in major currency units. '32,59' becomes "
            "32.59. Null only if no price is legible."
        )
    )
    currency: Optional[str] = Field(
        description="ISO 4217 code for the price. Romanian lei is RON."
    )
    price_text: Optional[str] = Field(
        description=(
            "The price exactly as printed, including separators and any currency "
            "mark, e.g. '32,59'. Used to cross-check the parsed number."
        )
    )
    price_kind: PriceKind = Field(
        description="'promo' or 'clearance' only when the tag actually says so."
    )
    original_price: Optional[float] = Field(
        description="Struck-through or 'was' price when a reduction is shown. Null otherwise."
    )
    unit_price_text: Optional[str] = Field(
        description="Per-litre or per-unit price as printed in the tag's small print."
    )
    promo_text: Optional[str] = Field(
        description="Promotional wording on or beside the tag, e.g. '1+1 gratis', '-25%'."
    )

    raw_tag_text: Optional[str] = Field(
        description=(
            "Verbatim transcription of the whole price tag, including the small "
            "article line above the price. Transcribe what you see even where it is "
            "abbreviated; do not tidy it up."
        )
    )
    raw_label_text: Optional[str] = Field(
        description="Verbatim transcription of the readable text on the bottle label."
    )

    ocr_price_index: Optional[int] = Field(
        description=(
            "When the caller listed prices that local OCR read off the tag rail, "
            "the 1-based position in that list of the price this record uses. "
            "Null if the price came from somewhere else, or if no list was given."
        )
    )
    name_source: NameSource = Field(description="Where the name was read from.")
    pairing_confidence: Confidence = Field(
        description=(
            "high = tag and bottle clearly correspond (aligned and names agree). "
            "medium = alignment is plausible but unverified. "
            "low = guessed, crowded shelf, or tag with no matching bottle."
        )
    )
    pairing_note: Optional[str] = Field(
        description="One short clause on why confidence is not high. Null when it is."
    )
    bottle_bbox: Optional[BBox] = Field(description="Box around the bottle, if visible.")
    tag_bbox: Optional[BBox] = Field(description="Box around the price tag, if visible.")


class UnreadableTag(BaseModel):
    """A price tag that is present but could not be tied to an identifiable wine."""

    raw_text: Optional[str] = Field(description="Whatever of the tag could be read.")
    price: Optional[float] = Field(description="Price if the number was legible.")
    currency: Optional[str] = Field(description="ISO 4217 code if determinable.")
    reason: str = Field(description="Why it could not be resolved, in a few words.")
    tag_bbox: Optional[BBox] = Field(description="Box around the tag.")


class BandExtraction(BaseModel):
    """Everything read out of one image region (a shelf band or a tile of one)."""

    wines: List[WineEntry] = Field(
        description=(
            "One entry per distinct wine-and-price pairing. Several identical "
            "bottles behind one tag are ONE entry. Different vintages or bottle "
            "sizes of the same wine are separate entries."
        )
    )
    unreadable_tags: List[UnreadableTag] = Field(
        description="Price tags that are visible but could not be resolved to a wine."
    )
    non_wine_present: bool = Field(
        description="True if the region also holds beer, spirits, water or soft drinks."
    )
    notes: Optional[str] = Field(description="Brief note on anything that limited the read.")
