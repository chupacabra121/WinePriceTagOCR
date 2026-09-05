"""Shelf geometry derived from local OCR, with no model call.

The original pipeline spent one vision call per photo asking the model where the
shelves were. It does not need to: a shelf is *defined* by its tag rail, and a
tag rail is a horizontal run of price-shaped text at a common height. Local OCR
already reports every price with a box, so the rails fall out of the geometry
for free — faster, cheaper, and repeatable.

Measured on ``data/samples/annabella/IMG_5755.HEIC``: the eleven prices of the
top rail come back at confidence 1.0 with x-centres in exactly the left-to-right
order of the hand-read ground truth, and the bottle-label text above them lands
at matching x. That correspondence is what this module turns into structure.

Vocabulary
----------
rail    a horizontal run of prices — the front lip of one shelf
band    one rail plus the bottles standing above it, which is the unit the
        model is asked to read
slot    one price on a rail plus the label text directly above it
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .normalize import parse_price_text
from .vision import OCRLine, OCRPage

# A price token is mostly digits with a two-digit tail. Leading/trailing junk is
# tolerated because OCR routinely glues a tag border or a stray mark onto the
# number ("|50.89", "'25.49", '"25.49').
PRICE_RE = re.compile(r"^[^\d]{0,3}(\d{1,4})\s?[.,·'’]\s?(\d{2})([^\d]{0,4})$")

# A trailing word decides whether "0.75L" is a price or a bottle. Currency marks
# ride along on real prices — Penny's OCR returns "13,49 El" and "16,85LE" — so
# letters cannot simply be banned; it is the units of measure that disqualify.
VOLUME_SUFFIXES = {
    "L", "ML", "CL", "DL", "LT", "LTR", "LITRU", "LITRI", "LITER", "LITRE",
    "KG", "G", "GR", "GRAM", "GRAME", "BUC", "X",
}

# Electronic shelf labels — Carrefour, Kaufland, Mega Image — print the lei part
# large and the bani part small and raised, with no separator between them:
# "88" then a smaller "19". Vision reads that as one box, "8819". Magnifying the
# crop does not separate them, so the decimal point has to be inferred, and in
# this market it always falls two digits from the end.
# No letter may touch the digits: "750-ml", "856L" and "O87135" are a volume, a
# unit price and a barcode, and every one of them parses as a plausible price
# once you let a letter ride along.
BARE_PRICE_RE = re.compile(r"^[^\dA-Za-z]{0,2}(\d{3,5})[^\dA-Za-z]{0,2}$")

# Bounds on an inferred price, in major units. Wide enough for a 3L bag-in-box
# at one end and a miniature at the other; tight enough to reject an EAN
# fragment or a "750" that came off a bottle label.
BARE_MIN = 0.9
BARE_MAX = 3000.0

# A bare number has to look like a price to be taken as one. The two tests that
# actually work on this library:
#
#   height — a price is set in the tag's largest type, so it is about as tall as
#   the separated prices elsewhere in the photo. "2021" printed on a bottle label
#   is a quarter of that.
#
#   company — a run of bare numbers that is *only* bare numbers must stretch
#   across the frame like a real rail. Vintages and brand years ("1827", "1958")
#   cluster on adjacent bottles instead, so a short huddle of them is rejected.
BARE_MIN_HEIGHT_RATIO = 0.6

# Some labels put enough space between the lei and the bani that Vision returns
# two boxes rather than one — "15" and ",99", or "170" and "29". The bani box is
# smaller, sits immediately to the right, and overlaps vertically.
SPLIT_MAX_GAP = 0.55        # multiples of the lei box's width
SPLIT_MIN_HEIGHT_RATIO = 0.25
SPLIT_MAX_HEIGHT_RATIO = 0.92
BARE_ONLY_MIN_TOKENS = 3
BARE_ONLY_MIN_SPAN = 0.25
# Two is enough when they are far enough apart to be tags on a rail rather than
# two numbers on neighbouring bottles.
BARE_ONLY_PAIR_SPAN = 0.40
# Tags in one fixture are printed in one type size, so a bare-only rail whose
# boxes differ wildly in height is a coincidence, not a shelf.
BARE_ONLY_MAX_HEIGHT_SPREAD = 1.8

# A vertical stretch of the photo taller than this with no band over it is
# assumed to be a shelf whose prices were never read, and gets a band of its own
# so the model still sees it. Losing a whole shelf silently is the worst failure
# this pipeline has.
COVERAGE_GAP = 0.14

# A fallback strip is cut into tiles no larger than this many source pixels on
# an edge. The reasoning is the opposite of the intuitive one: where local OCR
# found nothing, the model is the only reader left, so it needs *more*
# resolution, not less. A whole 4284x5712 frame squeezed into a 1600px crop is
# a 2.7x reduction and the tags are unreadable in it; a tile of this size
# reduces by about 1.4x and they survive.
FALLBACK_TILE_PX = 2200
FALLBACK_TILE_OVERLAP = 0.12
FALLBACK_MAX_TILES = 9

# Glyphs rotated 180 degrees, as Vision tends to transcribe them. Used only to
# *detect* an upside-down reading: the repair is not reliable enough to trust
# (a flipped 5 reads as S, which Vision often then calls a 9), so a hit marks
# the token for the model to resolve rather than silently rewriting it.
_UNFLIP = {
    "0": "0", "1": "1", "8": "8", "6": "9", "9": "6",
    "S": "5", "s": "5", "Z": "2", "z": "2", "E": "3", "h": "4", "L": "7",
    "'": ".", "°": ".", "·": ".", "*": ".", "`": ".", "‘": ".", "’": ".",
}

MIN_CONF = 0.5

# Two prices belong to the same rail when they sit at a similar height and are
# not too far apart horizontally. The y tolerance scales with token height so a
# close-up (tall digits) is as tolerant as a wide shot (short digits), and the
# floor keeps it workable when boxes are tiny.
RAIL_Y_TOL_FACTOR = 1.1
RAIL_Y_TOL_FLOOR = 0.012
# Tags on a hypermarket rail can be a third of the frame apart — Carrefour's
# electronic labels sit one per facing on a sparsely stocked shelf. Anything
# tighter than this splits those rails into unusable fragments.
RAIL_X_GAP_MAX = 0.35
# A rail photographed at an angle slopes across the frame, so two prices far
# apart horizontally may legitimately differ in height by more than two
# neighbours do. Without this the chain snaps and one shelf becomes three.
# Distinct rails sit an order of magnitude further apart than this allows.
RAIL_SLOPE_TOL = 0.10

# Within a rail, the price is the largest thing on the tag. Anything markedly
# shorter is small print — a per-litre line like "1 L = 53,32 Lei", an article
# code, a unit weight — and must not be mistaken for the price.
RAIL_MIN_HEIGHT_RATIO = 0.55

# A band crop narrower than this is hard to read and gives the model no context
# for pairing, so a lone tag's band is widened to at least this much of the frame.
MIN_BAND_WIDTH = 0.24

# The shelf tag around a price. A Romanian shelf tag sets the article line above
# the price and the small print below it, so the box reaches further up than
# down, and only about a price-width to either side — wide enough to catch the
# article line, narrow enough not to annex the neighbouring tag.
TAG_BOX_ABOVE = 3.2      # multiples of the price box's height
TAG_BOX_BELOW = 2.4
TAG_BOX_SIDES = 1.1      # multiples of the price box's width

# Numbers that are printed on a shelf tag but are not what the wine costs. Every
# chain surveyed prints at least two of these, and several print five. The worst
# is Kaufland's "Preț+garanție", which is a well-formed price exactly 0.50 above
# the real one — plausible enough to survive any range check.
# The per-litre figure a Romanian tag is required to print, in the several
# shapes it actually appears in: "1 L = 53,32 Lei", "1 LT 19.00 LEI",
# "(1 litru = 24.52)", "LEI/LTR: 55,72", "0.75 L sau 135.87 LEI/L".
# Two orders occur: the marker before the number ("1 L = 53,32 Lei") and the
# number before the marker ("0.75 L sau 135.87 LEI/L").
UNIT_PRICE_RE = re.compile(
    r"(?:1\s*(?:l|lt|ltr|litr[ux]?)|lei\s*/\s*l(?:tr)?|pre[tț]\s*/\s*l|/\s*l\b)"
    r"[^\d]{0,8}(\d{1,4}[.,]\d{2})"
    r"|(\d{1,4}[.,]\d{2})\s*(?:lei\s*)?/\s*l(?:tr|itru)?\b",
    re.I,
)
VOLUME_ON_TAG_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*[-\s]?(l|lt|litri|litru|ml|cl)\b", re.I
)
_VOLUME_FACTOR = {"l": 1.0, "lt": 1.0, "litri": 1.0, "litru": 1.0,
                  "ml": 0.001, "cl": 0.01}

# How close the arithmetic has to come. Retailers round the printed per-litre
# figure to two places, so exact equality is not available.
RECONCILE_TOLERANCE = 0.02

DECOY_RE = re.compile(
    r"garan[tț]|sgr|"                       # container deposit
    r"litru|lei\s*/\s*l|/\s*l\b|/\s*ltr|pre[tț]\s*/\s*l|\bsau\b|"   # per-litre
    r"\bbuc\b|bucat|bucă[tț]|cumpar|minim|economis|"                # multibuy
    r"valabil|\bîn loc de\b|\bin loc de\b|"                         # promo framing
    r"\d{2}\.\d{2}\.\d{4}",                                         # a date
    re.I,
)
# One shelf is never most of the frame. Without a cap, a rail whose only
# x-overlapping neighbour is several shelves above inherits everything between
# them, and the crop stops being a shelf at all.
MAX_BAND_HEIGHT = 0.45


@dataclass(frozen=True)
class PriceToken:
    """One price read off a rail.

    ``kind`` is ``"separated"`` when the tag printed a decimal separator and OCR
    read it, and ``"bare"`` when the separator had to be inferred from an
    electronic label's run-together digits. The difference matters downstream:
    a bare price is worth a second look from the model, a separated one is not.
    """

    line: OCRLine
    value: float
    kind: str = "separated"

    @property
    def decimal_inferred(self) -> bool:
        """The separator was not printed and had to be assumed."""
        return self.kind == "bare"

    @property
    def needs_corroboration(self) -> bool:
        """Whether this reading is only safe with other prices lined up beside it.

        Both reconstructed forms qualify. A bare number could be a vintage; a
        merged pair could be two unrelated boxes that happened to sit close
        together. A printed separator needs no such company.
        """
        return self.kind in {"bare", "split"}

    @property
    def cx(self) -> float:
        return self.line.cx

    @property
    def cy(self) -> float:
        return self.line.cy


def looks_flipped(text: str) -> bool:
    """True when ``text`` is not a price but its 180-degree reading would be."""
    if PRICE_RE.match(text.strip()):
        return False
    reversed_glyphs = "".join(_UNFLIP.get(c, c) for c in reversed(text.strip()))
    return bool(PRICE_RE.match(reversed_glyphs))


def price_tokens(page: OCRPage, min_conf: float = MIN_CONF) -> list[PriceToken]:
    """Every price-shaped line on the page, sorted left to right.

    Both printed forms are collected: separated ("54,59") and the run-together
    form an electronic label produces ("8819"). Bare tokens are the riskier
    half — a bare number could be a barcode fragment or a volume — so they are
    only ever trusted after ``find_rails`` has seen them line up with others.
    """
    separated: list[PriceToken] = []
    candidates: list[PriceToken] = []
    for line in page.lines:
        if line.conf < min_conf:
            continue
        text = line.text.strip()

        # "3 Buc x 15.39 = 46.17 lei" and "+ garantie SGR 0,50 Lei" are numbers
        # on a shelf tag that are not what the wine costs.
        if DECOY_RE.search(text):
            continue

        if match := PRICE_RE.match(text):
            if _is_measure(match.group(3)):
                continue
            value = parse_price_text(f"{match.group(1)}.{match.group(2)}")
            if value is not None and value > 0:
                separated.append(PriceToken(line, value, "separated"))
            continue

        if match := BARE_PRICE_RE.match(text):
            digits = match.group(1)
            value = int(digits) / 100.0
            if BARE_MIN <= value <= BARE_MAX and len(set(digits)) > 1:
                candidates.append(PriceToken(line, value, "bare"))

    floor = _bare_height_floor(separated)
    bare = [t for t in candidates if t.line.height >= floor]

    # A merge wins over a bare reading of its own lei part: "170" beside a small
    # "29" is 170.29, not 1.70.
    split, consumed = _split_prices(page, min_conf)
    bare = [t for t in bare if id(t.line) not in consumed]
    separated = [t for t in separated if id(t.line) not in consumed]
    return sorted(separated + bare + split, key=lambda t: t.cx)


_LEI_RE = re.compile(r"^[^\dA-Za-z]{0,2}(\d{1,4})[^\dA-Za-z]{0,2}$")
_BANI_RE = re.compile(r"^[.,·'’]?\s?(\d{2})[^\dA-Za-z]{0,3}$")


def _split_prices(
    page: OCRPage, min_conf: float
) -> tuple[list[PriceToken], set[int]]:
    """Prices Vision returned as two boxes: the lei, then the bani.

    Auchan and the cash-and-carry chains set the bani small and raised with a
    visible gap, which is enough for Vision to call it a separate line. Left
    alone, the lei part looks like a bare integer and the bani part like noise,
    and neither is a price — so a whole chain returns almost nothing.
    """
    candidates = [l for l in page.lines if l.conf >= min_conf]
    out: list[PriceToken] = []
    consumed: set[int] = set()

    for left in candidates:
        lei = _LEI_RE.match(left.text.strip())
        if not lei:
            continue
        for right in candidates:
            if right is left or id(right) in consumed:
                continue
            bani = _BANI_RE.match(right.text.strip())
            if not bani:
                continue
            gap = right.x0 - left.x1
            if gap < -0.2 * left.width or gap > SPLIT_MAX_GAP * left.width:
                continue
            if min(left.y1, right.y1) <= max(left.y0, right.y0):
                continue  # no vertical overlap: different rows, not one price
            ratio = right.height / left.height if left.height else 0
            if not SPLIT_MIN_HEIGHT_RATIO <= ratio <= SPLIT_MAX_HEIGHT_RATIO:
                continue

            value = int(lei.group(1)) + int(bani.group(1)) / 100.0
            if not BARE_MIN <= value <= BARE_MAX:
                continue
            merged = OCRLine(
                text=f"{lei.group(1)},{bani.group(1)}",
                conf=min(left.conf, right.conf),
                x0=left.x0, y0=min(left.y0, right.y0),
                x1=right.x1, y1=max(left.y1, right.y1),
                height_px=left.height_px,
            )
            out.append(PriceToken(merged, value, "split"))
            consumed.add(id(left))
            consumed.add(id(right))
            break
    return out, consumed


def _is_measure(suffix: str) -> bool:
    """True when the text trailing a number names a unit rather than a currency."""
    letters = re.sub(r"[^A-Za-z]", "", suffix).upper()
    return bool(letters) and letters in VOLUME_SUFFIXES


def _bare_height_floor(separated: Sequence[PriceToken]) -> float:
    """The smallest box a bare number may occupy and still be read as a price.

    Only meaningful when the photo also contains separated prices: every tag in
    a fixture is printed in one type size, so those are a direct reference.

    There is deliberately no page-wide fallback. In a shop that is entirely
    electronic labels there is nothing to compare against, and the obvious proxy
    — "a price is among the largest text on the page" — is simply false: a
    Carrefour label's digits are half the height of the brand name on the bottle
    behind it. Weeding out those pages is left to the rail tests, which look at
    whether the numbers line up like tags rather than at how big they are.
    """
    if not separated:
        return 0.0
    heights = sorted(t.line.height for t in separated)
    return BARE_MIN_HEIGHT_RATIO * heights[len(heights) // 2]


def flipped_lines(page: OCRPage, min_conf: float = MIN_CONF) -> list[OCRLine]:
    """Lines that read as a price only when turned upside down.

    Vision decides text orientation per region, and gets it wrong on a narrow
    strip of bare digits often enough to matter. Any hit here means the page's
    prices should not be trusted without a second opinion.
    """
    return [
        l for l in page.lines
        if l.conf >= min_conf and looks_flipped(l.text)
    ]


@dataclass
class Rail:
    """A horizontal run of prices: the front lip of one shelf."""

    tokens: list[PriceToken]

    @property
    def y0(self) -> float:
        return min(t.line.y0 for t in self.tokens)

    @property
    def y1(self) -> float:
        return max(t.line.y1 for t in self.tokens)

    @property
    def x0(self) -> float:
        return min(t.line.x0 for t in self.tokens)

    @property
    def x1(self) -> float:
        return max(t.line.x1 for t in self.tokens)

    @property
    def cy(self) -> float:
        return sum(t.cy for t in self.tokens) / len(self.tokens)

    @property
    def median_height(self) -> float:
        heights = sorted(t.line.height for t in self.tokens)
        return heights[len(heights) // 2]


def _tolerance(a: PriceToken, b: PriceToken) -> float:
    base = max(
        RAIL_Y_TOL_FACTOR * max(a.line.height, b.line.height),
        RAIL_Y_TOL_FLOOR,
    )
    return base + RAIL_SLOPE_TOL * abs(b.cx - a.cx)


def _prune_small(tokens: list[PriceToken]) -> list[PriceToken]:
    """Drop tokens far shorter than the rail's own digits.

    Lidl prints ``1 L = 53,32 Lei`` under the price, and Vision sometimes hands
    back the ``53,32`` on its own. It sits at the same height as the rail and
    would otherwise be adopted as a second price for that tag.
    """
    if len(tokens) < 2:
        return tokens
    heights = sorted(t.line.height for t in tokens)
    median = heights[len(heights) // 2]
    if median <= 0:
        return tokens
    return [t for t in tokens if t.line.height >= RAIL_MIN_HEIGHT_RATIO * median]


def find_rails(tokens: Sequence[PriceToken]) -> list[Rail]:
    """Group price tokens into rails by chaining left-to-right neighbours.

    Chaining rather than clustering on y alone is deliberate: a shelf photo-
    graphed from an angle has a rail that slopes across the frame, so its first
    and last price can sit further apart in y than two different rails do at the
    same x. Consecutive prices along one rail are always close, though, so
    single-linkage along x follows the slope.

    A rail of one survives only if its price carried a printed decimal
    separator. A lone run-together number with nothing to corroborate it is far
    more likely to be a barcode or a volume than a price.
    """
    ordered = sorted(tokens, key=lambda t: t.cx)
    parent = list(range(len(ordered)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, a in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            b = ordered[j]
            if b.line.x0 - a.line.x1 > RAIL_X_GAP_MAX:
                break  # ordered by x, so nothing further can be close enough
            if abs(a.cy - b.cy) <= _tolerance(a, b):
                union(i, j)

    groups: dict[int, list[PriceToken]] = {}
    for i, token in enumerate(ordered):
        groups.setdefault(find(i), []).append(token)

    rails: list[Rail] = []
    for group in groups.values():
        kept = _prune_small(sorted(group, key=lambda t: t.cx))
        if kept and _is_plausible_rail(kept):
            rails.append(Rail(kept))
    return sorted(rails, key=lambda r: r.cy)


def _is_plausible_rail(tokens: Sequence[PriceToken]) -> bool:
    """Whether a chain of tokens is really a shelf's tag rail.

    One separated price is enough on its own — the decimal separator is strong
    evidence of a price. A chain made only of inferred prices has to earn it:
    several of them, spread across the frame the way tags on a rail are.
    """
    if any(not t.needs_corroboration for t in tokens):
        return True
    if len(tokens) < 2:
        return False

    heights = [t.line.height for t in tokens]
    if min(heights) <= 0 or max(heights) / min(heights) > BARE_ONLY_MAX_HEIGHT_SPREAD:
        return False

    span = max(t.line.x1 for t in tokens) - min(t.line.x0 for t in tokens)
    if len(tokens) >= BARE_ONLY_MIN_TOKENS:
        return span >= BARE_ONLY_MIN_SPAN
    return span >= BARE_ONLY_PAIR_SPAN


@dataclass
class Slot:
    """One shelf tag: a price, the tag it is printed on, and the bottles above it.

    The tag is the unit rather than the bottle because a survey of the whole
    photo library found that the tag's article line is the most reliable name
    in almost every chain — it is the largest text on a Kaufland, Carrefour,
    Auchan or ATAC label — while bottle labels come back as brand words with the
    varietal corrupted, and one tag routinely faces a dozen bottles. Name and
    price read off the same object cannot be mispaired.
    """

    price: PriceToken
    tag_lines: list[OCRLine] = field(default_factory=list)
    other_prices: list[PriceToken] = field(default_factory=list)
    labels: list[OCRLine] = field(default_factory=list)
    # Filled by `reconcile`: what the tag's own printed arithmetic says about
    # the price read off it.
    check: Optional[str] = None

    @property
    def label_text(self) -> str:
        return " ".join(l.text for l in self.labels)

    @property
    def tag_text(self) -> str:
        return " ".join(l.text for l in self.tag_lines)

    def box(self) -> tuple[float, float, float, float]:
        """The tag this price is printed on, in whole-photo fractions."""
        line = self.price.line
        h, w = line.height, line.width
        return (
            line.x0 - TAG_BOX_SIDES * w,
            line.y0 - TAG_BOX_ABOVE * h,
            line.x1 + TAG_BOX_SIDES * w,
            line.y1 + TAG_BOX_BELOW * h,
        )

    def contains(self, line: OCRLine) -> bool:
        x0, y0, x1, y1 = self.box()
        return x0 <= line.cx <= x1 and y0 <= line.cy <= y1


@dataclass
class Band:
    """One rail plus the bottles above it — the unit handed to the model."""

    index: int
    rail: Optional[Rail]
    x0: float
    y0: float
    x1: float
    y1: float
    slots: list[Slot] = field(default_factory=list)
    stray_lines: list[OCRLine] = field(default_factory=list)
    # Set when the band is a whole-photo fallback rather than a detected shelf.
    is_fallback: bool = False

    @property
    def label(self) -> str:
        if self.is_fallback:
            if self.y0 <= 0.001 and self.y1 >= 0.999:
                return "whole photo (no rail detected)"
            return f"unread strip {self.y0:.2f}-{self.y1:.2f} (no rail detected)"
        n = len(self.slots)
        return f"shelf {self.index + 1} ({n} price{'s' if n != 1 else ''})"

    @property
    def prices(self) -> list[float]:
        return [s.price.value for s in self.slots]


def _rail_above(rail: Rail, others: Sequence[Rail], min_overlap: float = 0.15) -> Optional[Rail]:
    """The nearest rail above ``rail`` that belongs to the same fixture.

    "Above" alone is not enough: a photo often catches a neighbouring fixture —
    a snack rack beside a wine bay — whose rails interleave vertically with the
    ones we care about. Taking the nearest rail above without checking that the
    two overlap horizontally puts the top of a band at an unrelated shelf and
    crops the bottles out of it entirely.
    """
    span = rail.x1 - rail.x0
    best: Optional[Rail] = None
    for other in others:
        if other is rail or other.cy >= rail.cy:
            continue
        shared = min(rail.x1, other.x1) - max(rail.x0, other.x0)
        if span <= 0 or shared / span < min_overlap:
            continue
        if best is None or other.cy > best.cy:
            best = other
    return best


def _default_band_height(rails: Sequence[Rail]) -> float:
    """How tall a band is when there is no rail above it to stop at.

    The typical gap between two stacked rails is the best available estimate of
    one shelf's height, so a first band reaches back by that much rather than
    swallowing the whole ceiling of a wide shot.
    """
    gaps: list[float] = []
    for rail in rails:
        above = _rail_above(rail, rails)
        if above is not None:
            gaps.append(rail.y0 - above.y1)
    usable = sorted(g for g in gaps if g > 0.01)
    if usable:
        return usable[len(usable) // 2]
    return 0.30


def _band_bounds(
    rail: Rail,
    rails: Sequence[Rail],
    default_height: float,
    pad_x: float = 0.02,
    pad_below: float = 0.012,
) -> tuple[float, float, float, float]:
    """The crop for a band: its rail, plus the bottles standing above it.

    The top stops at the bottom of the nearest rail above *in the same fixture* —
    those bottles stand on that shelf and are priced by this rail.
    """
    above = _rail_above(rail, rails)
    top = above.y1 if above is not None else rail.y0 - default_height
    top = max(top, rail.y0 - MAX_BAND_HEIGHT)

    x0 = max(0.0, rail.x0 - pad_x)
    x1 = min(1.0, rail.x1 + pad_x)
    # A rail of one tag would otherwise crop to a tall ribbon: mostly bottle,
    # no neighbours, nothing to judge the pairing against. Widen it around the
    # rail's centre, then slide the window back inside the frame if it hangs off
    # an edge. Sliding rather than clamping keeps x1 > x0.
    if x1 - x0 < MIN_BAND_WIDTH:
        centre = (rail.x0 + rail.x1) / 2
        half = min(MIN_BAND_WIDTH, 1.0) / 2
        x0, x1 = centre - half, centre + half
        if x0 < 0.0:
            x0, x1 = 0.0, min(1.0, x1 - x0)
        elif x1 > 1.0:
            x0, x1 = max(0.0, x0 - (x1 - 1.0)), 1.0

    return (x0, max(0.0, min(top, rail.y0)), x1, min(1.0, rail.y1 + pad_below))


def _assign_labels(page: OCRPage, band: Band) -> None:
    """Attach every non-price line in the band to the price it sits above.

    A tag prices the bottles directly above it, and both are read in the same
    left-to-right order, so x is the pairing signal. A line is given to the
    slot whose price box it overlaps most in x; failing any overlap, to the
    nearest price centre within half a slot's width. Anything further away is
    kept as a stray rather than forced onto a slot, because a confidently wrong
    pairing is worse than an unpaired line.
    """
    if not band.slots:
        band.stray_lines = list(page.within(band.x0, band.y0, band.x1, band.y1))
        return

    price_lines = {id(s.price.line) for s in band.slots}

    # Half the median spacing between neighbouring prices: the widest a slot can
    # plausibly reach without stealing from its neighbour.
    centres = [s.price.cx for s in band.slots]
    gaps = [b - a for a, b in zip(centres, centres[1:])] or [0.1]
    reach = max(0.02, sorted(gaps)[len(gaps) // 2] * 0.6)

    for line in page.within(band.x0, band.y0, band.x1, band.y1):
        if id(line) in price_lines:
            continue

        # A line inside a tag's own box belongs to that tag — most importantly
        # the article line above the price, which is the name we actually want.
        owner = next((s for s in band.slots if s.contains(line)), None)
        if owner is not None:
            owner.tag_lines.append(line)
            continue

        # Otherwise it is bottle text, assigned to the price it stands over.
        best: Optional[Slot] = None
        best_overlap = 0.0
        for slot in band.slots:
            overlap = line.x_overlap(slot.price.line)
            if overlap > best_overlap:
                best, best_overlap = slot, overlap
        if best is None:
            nearest = min(band.slots, key=lambda s: abs(s.price.cx - line.cx))
            if abs(nearest.price.cx - line.cx) <= reach:
                best = nearest
        if best is None:
            band.stray_lines.append(line)
        else:
            best.labels.append(line)

    _attach_other_prices(page, band)
    for slot in band.slots:
        slot.labels.sort(key=lambda l: (l.y0, l.x0))
        slot.tag_lines.sort(key=lambda l: (l.y0, l.x0))
        slot.check = reconcile(slot)


def reconcile(slot: Slot) -> Optional[str]:
    """Check a price against the per-litre figure printed on its own tag.

    Romanian tags print a unit price, which makes every tag self-checking:
    price divided by volume must equal it. That is worth a great deal here,
    because it is the only verification available without another pair of eyes.

    It also catches the failure that turned out to be the most common one in
    practice. Lidl prints an "18+" age roundel immediately left of the price,
    and OCR absorbs it as a leading digit — 18.49 comes back as "918.49",
    9.69 as "29,69". The result is a well-formed, confident, wrong number that
    no range check would ever question. The tag's own arithmetic does.
    """
    text = " ".join(l.text for l in slot.tag_lines)
    if not text:
        return None

    unit_match = UNIT_PRICE_RE.search(text)
    if not unit_match:
        return None
    unit = parse_price_text(unit_match.group(1) or unit_match.group(2))
    if not unit or unit <= 0:
        return None

    litres = _litres_on_tag(text)
    if not litres:
        return None

    expected = unit * litres
    price = slot.price.value
    if expected <= 0:
        return None
    if abs(price - expected) / expected <= RECONCILE_TOLERANCE:
        return f"confirmed by the tag's own per-litre line ({unit:g}/L x {litres:g}L)"

    # A spurious digit glued to the front of the price is the common failure.
    digits = f"{price:.2f}".replace(".", "")
    if len(digits) > 3:
        trimmed = int(digits[1:]) / 100.0
        if trimmed > 0 and abs(trimmed - expected) / expected <= RECONCILE_TOLERANCE:
            return (
                f"DISAGREES with the tag's own per-litre line: {unit:g}/L x "
                f"{litres:g}L = {expected:.2f}, not {price:.2f}. Dropping the "
                f"leading digit gives {trimmed:.2f}, which does reconcile — an "
                "icon beside the price is often read as a digit. Check the image."
            )
    return (
        f"disagrees with the tag's own per-litre line: {unit:g}/L x {litres:g}L "
        f"= {expected:.2f}, not {price:.2f}. Check the image."
    )


def _litres_on_tag(text: str) -> Optional[float]:
    """The bottle volume printed on a tag, in litres.

    Skips the "1 L" that introduces the per-litre figure itself — otherwise
    every tag reports a one-litre bottle and the check becomes a tautology.
    """
    without_unit_clause = UNIT_PRICE_RE.sub(" ", text)
    for match in VOLUME_ON_TAG_RE.finditer(without_unit_clause):
        try:
            value = float(match.group(1).replace(",", "."))
        except ValueError:
            continue
        litres = value * _VOLUME_FACTOR[match.group(2).lower()]
        if 0.1 <= litres <= 20:
            return litres
    return None


def _attach_other_prices(page: OCRPage, band: Band) -> None:
    """Record the other money-shaped numbers printed on each tag.

    Every chain surveyed puts more than one number on a tag — a deposit, a
    per-litre figure, a three-for price, a struck-through original. They are
    filtered out of the rail, but hiding them from the model would be a mistake:
    when the rail's pick is the wrong one, these are the evidence that says so.
    """
    for slot in band.slots:
        x0, y0, x1, y1 = slot.box()
        for line in page.within(x0, y0, x1, y1):
            if line is slot.price.line:
                continue
            text = line.text.strip()
            if PRICE_RE.match(text) or BARE_PRICE_RE.match(text):
                slot.other_prices.append(PriceToken(line, 0.0, "context"))


def build_bands(page: OCRPage, min_conf: float = MIN_CONF) -> list[Band]:
    """Turn a page of OCR into shelf bands ready to be cropped and read.

    A photo with no detectable rail — a single bottle, a close-up of one tag, a
    shot where every price failed to resolve — still yields one whole-frame
    band, so nothing is silently dropped.
    """
    rails = find_rails(price_tokens(page, min_conf))
    default_height = _default_band_height(rails)

    bands: list[Band] = []
    for i, rail in enumerate(rails):
        x0, y0, x1, y1 = _band_bounds(rail, rails, default_height)
        band = Band(i, rail, x0, y0, x1, y1,
                    slots=[Slot(t) for t in rail.tokens])
        _assign_labels(page, band)
        bands.append(band)

    bands.extend(_coverage_bands(page, bands))
    bands.sort(key=lambda b: (b.y0, b.x0))
    for i, band in enumerate(bands):
        band.index = i
    return bands


def _coverage_bands(page: OCRPage, bands: Sequence[Band]) -> list[Band]:
    """Bands over the parts of the photo no rail claimed.

    Rail detection is good but not complete: a shelf whose prices were blurred,
    or whose two electronic labels sat too close together to pass the rail
    tests, produces no band and would otherwise vanish from the run without
    anything recording that it had been there.

    Every uncovered stretch taller than ``COVERAGE_GAP`` therefore gets a band
    anyway, flagged as a fallback. It costs one extra read of a strip that is
    often empty, and it makes the pipeline lossless by construction.
    """
    spans = sorted((b.y0, b.y1) for b in bands)
    merged: list[list[float]] = []
    for y0, y1 in spans:
        if merged and y0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], y1)
        else:
            merged.append([y0, y1])

    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for y0, y1 in merged:
        if y0 - cursor >= COVERAGE_GAP:
            gaps.append((cursor, y0))
        cursor = max(cursor, y1)
    if 1.0 - cursor >= COVERAGE_GAP:
        gaps.append((cursor, 1.0))

    extra: list[Band] = []
    for y0, y1 in gaps:
        # A strip with no text at all is bare shelf, floor or ceiling; reading it
        # would tell us nothing.
        if not page.within(0.0, y0, 1.0, y1):
            continue
        for x0, ty0, x1, ty1 in _fallback_tiles(page, y0, y1):
            band = Band(0, None, x0, ty0, x1, ty1, is_fallback=True)
            if not page.within(x0, ty0, x1, ty1):
                continue
            _assign_labels(page, band)
            extra.append(band)
    return extra


def _fallback_tiles(
    page: OCRPage, y0: float, y1: float
) -> list[tuple[float, float, float, float]]:
    """Cut an unclaimed strip into overlapping tiles the model can actually read."""
    width_px = max(1, page.width)
    height_px = max(1, page.height * (y1 - y0))

    def divisions(extent_px: float) -> int:
        return max(1, int(-(-extent_px // FALLBACK_TILE_PX)))

    cols, rows = divisions(width_px), divisions(height_px)
    while cols * rows > FALLBACK_MAX_TILES and (cols > 1 or rows > 1):
        if rows >= cols:
            rows -= 1
        else:
            cols -= 1

    def spans(lo: float, hi: float, n: int) -> list[tuple[float, float]]:
        if n <= 1:
            return [(lo, hi)]
        length = hi - lo
        span = length / (n - (n - 1) * FALLBACK_TILE_OVERLAP)
        step = (length - span) / (n - 1)
        return [(lo + i * step, min(hi, lo + i * step + span)) for i in range(n)]

    return [
        (cx0, ry0, cx1, ry1)
        for ry0, ry1 in spans(y0, y1, rows)
        for cx0, cx1 in spans(0.0, 1.0, cols)
    ]


def digest(band: Band, max_label_lines: int = 6) -> str:
    """A compact text rendering of what local OCR read in this band.

    This is handed to the model alongside the image crop. The crop has to be
    downscaled to reach the model, so it cannot show what OCR saw at native
    resolution; this closes that gap. Confidences are included because the model
    needs to know which readings to lean on and which to re-read from the image.
    """
    lines: list[str] = []
    if band.rail is None:
        lines.append("No price rail was detected by local OCR in this image.")
    else:
        lines.append(
            f"Local OCR read {len(band.slots)} price(s) along this shelf's tag "
            "rail, listed left to right. x is the horizontal centre, 0-1 across "
            "the crop you were given."
        )
    for i, slot in enumerate(band.slots, start=1):
        conf = slot.price.line.conf
        head = (
            f"{i}. price {slot.price.value:.2f} "
            f"(read as {slot.price.line.text.strip()!r}, confidence {conf:.1f}, "
            f"x={_rescale(slot.price.cx, band.x0, band.x1):.3f})"
        )
        if slot.price.decimal_inferred:
            head += (
                " — DECIMAL POINT INFERRED: the label printed no separator, so "
                "the last two digits were taken as bani. Check this one against "
                "the image."
            )
        lines.append(head)
        if slot.check:
            lines.append(f"   {slot.check}")
        if slot.tag_lines:
            shown = slot.tag_lines[:max_label_lines]
            body = "; ".join(f"{l.text.strip()!r}({l.conf:.1f})" for l in shown)
            lines.append(f"   text on the same tag: {body}")
        if slot.other_prices:
            shown = slot.other_prices[:4]
            body = "; ".join(f"{t.line.text.strip()!r}" for t in shown)
            lines.append(
                f"   other numbers printed on this tag: {body} — one of these may "
                "be the real shelf price"
            )
        if slot.labels:
            shown = slot.labels[:max_label_lines]
            body = "; ".join(f"{l.text.strip()!r}({l.conf:.1f})" for l in shown)
            lines.append(f"   bottle text above the tag: {body}")
    if band.stray_lines:
        shown = band.stray_lines[:12]
        body = "; ".join(f"{l.text.strip()!r}" for l in shown)
        lines.append(f"Other text in this crop, not tied to a price: {body}")
    return "\n".join(lines)


def _rescale(v: float, lo: float, hi: float) -> float:
    """Map a whole-photo fraction into a crop's own 0-1 frame."""
    if hi <= lo:
        return 0.0
    return min(1.0, max(0.0, (v - lo) / (hi - lo)))
