"""Match a read wine to a standardized brand list.

The reference sheet is a flat SKU catalogue: Category, Group, Winery, Label,
Assortment, SKU name, Sweetness, Colour. What the pipeline has instead is the
tag's article line, half-abbreviated and often clipped — "AEROSOLI VIN ALB SEC
ST. 0.75L" or "...ANESC CABERNET SAUVIGNON SEC 750ML".

Matching is therefore not string similarity: most of the characters in both
strings are grape, colour, sweetness and volume words that every row shares.
The signal is carried by the few brand-bearing tokens, so tokens are weighted by
how rare they are in the reference (plain IDF) and a candidate only wins if it
shares a genuinely rare token with the wine — "PURCARI" identifies a row,
"CABERNET SEC 0.75L" identifies nothing.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

# Words that appear on nearly every Romanian wine tag. They carry no brand
# information, and leaving them in lets a shelf full of dry reds all match each
# other.
STOP = {
    "VIN", "VINUL", "VINURI", "WINE",
    "ALB", "ALBA", "ROSU", "ROSIE", "ROZE", "ROSE", "NEGRU", "WHITE", "RED",
    "SEC", "SECO", "DEMISEC", "DULCE", "DEMIDULCE", "DRY", "SWEET",
    "DS", "DD", "SGR", "NRB", "CC", "PL", "ST", "STICLA", "BOTTLE", "BIB",
    "L", "ML", "CL", "LT", "LITRU", "LITRI", "LITRE", "LITER", "LITERS",
    "LEI", "BUC", "PRET", "PRETUL", "PET", "DAMIGEANA", "DAMIGEAN",
    "SAU", "DE", "LA", "SI", "CU", "DIN", "PE", "AL", "A", "R", "S", "N",
    "VOL", "ALC", "GARANTIE", "AMBALAJ", "TVA",
    # varietals — descriptive, never brand-bearing on their own
    "CABERNET", "SAUVIGNON", "BLANC", "MERLOT", "CHARDONNAY", "RIESLING",
    "PINOT", "NOIR", "GRIGIO", "GRIS", "SYRAH", "SHIRAZ", "MUSCAT",
    "OTTONEL", "FETEASCA", "REGALA", "NEAGRA", "ALBA", "TAMAIOASA",
    "ROMANEASCA", "BUSUIOACA", "BOHOTIN", "ALIGOTE", "TRAMINER", "SPUMANT",
    "CUPAJ", "BLEND", "ROSATO", "BIANCO", "ROSSO", "VARIETAL", "SORT",
    # Appellation and quality classifications. These are rare enough in the
    # reference to score as strong anchors, but they describe a legal category,
    # not a producer — "DOC-CMD" once carried a Ceptura wine to a Cotnari row.
    "DOC", "DOCG", "CMD", "CMI", "CIB", "DOP", "IGP", "IGT", "AOC", "AOP",
    "VS", "VSO", "VSOC", "PGI", "PDO", "RESERVE", "REZERVA", "RISERVA",
    "SUPERIOR", "SUPERIORE", "CLASSICO", "GRAND", "CRU", "SELECTION",
    "SELECTIE", "COLLECTION", "COLECTIA", "EDITION", "EDITIE", "PREMIUM",
    # Words the reading pass writes *about* a tag rather than off it. Left in,
    # they behave as strong anchors because the reference never uses them:
    # "etichetă ilizibilă" once matched Cotnari's "Eticheta Galbena", and
    # "article line partly illegible" matched a winery called Business Line.
    # "ETICHETA" is deliberately NOT here: Cotnari sells Eticheta Galbena and
    # Eticheta Neagra, so the word is a brand. The gloss problem it caused
    # ("etichetă ilizibilă") is handled by stripping parentheticals and by the
    # words below.
    "LABEL", "LINE", "ARTICLE", "ILIZIBIL", "ILIZIBILA",
    "ILLEGIBLE", "UNREADABLE", "PARTLY", "CREST", "OVAL", "CREM", "TEXT",
    "BAG", "BOX", "IN", "OUT", "ANI", "YEARS", "UNICA", "MATURAT", "BARIC",
    "BARICURI", "BARRIQUE", "OAK", "STEJAR", "IMAGE", "PHOTO", "CROP", "TAG",
    "VISIBLE", "PARTIAL", "BLURRED", "BLUR", "PROMO", "REDUCERE", "OFERTA",
}

_DIA = str.maketrans("ăâîșțĂÂÎȘȚáéíóúüöäàèìòùçñ", "aaistAAISTaeiouuoaaeioucn")
_TOKEN = re.compile(r"[A-Z0-9]+")


_PARENS = re.compile(r"\([^)]*\)")


def tokens(text: Optional[str]) -> list[str]:
    """Brand-bearing tokens of a name, in order, stopwords and noise removed."""
    if not text:
        return []
    # The reading pass often appends its own gloss in brackets — "(oval-crest
    # label)", "(article line partly illegible)". That is commentary about the
    # photograph, not part of the product name, and its words are rare enough in
    # the reference to outscore the real brand.
    text = _PARENS.sub(" ", text)
    flat = unicodedata.normalize("NFKD", text.translate(_DIA))
    flat = "".join(c for c in flat if not unicodedata.combining(c)).upper()
    out = []
    for tok in _TOKEN.findall(flat):
        if tok in STOP or len(tok) < 2:
            continue
        if tok.isdigit():          # volumes, ABV, years
            continue
        if re.fullmatch(r"\d+[A-Z]*", tok):
            continue
        out.append(tok)
    return out


@dataclass(frozen=True)
class BrandRow:
    category: str
    group: str
    winery: str
    label: str
    assortment: str
    sku: str
    key_tokens: frozenset


@dataclass
class Match:
    row: Optional[BrandRow]
    score: float
    shared: tuple

    @property
    def matched(self) -> bool:
        return self.row is not None


class BrandIndex:
    """Inverted index over the reference sheet, weighted by token rarity."""

    # A candidate must share at least one token this rare with the wine. Common
    # tokens ("DOMENIILE", "CRAMA") are shared by hundreds of unrelated rows, so
    # agreement on them alone is not evidence of the same brand.
    MIN_ANCHOR_IDF = 3.0
    # A two-letter token is never a brand. "...AS CASTEL" once matched a Greek
    # Moscofilero on the strength of the "AS" left behind by a clipped article
    # line, which is the failure this rules out.
    MIN_ANCHOR_LEN = 3
    MIN_PREFIX = 4
    MAX_PREFIX = 8
    # Below this the best candidate is a coincidence, not a match. Calibrated so
    # a single mid-rarity token is not enough but two are.
    MIN_SCORE = 4.0
    # A win this narrow over the runner-up is a coin toss between two brands.
    MIN_MARGIN = 1.0

    def __init__(self, rows: Iterable[dict]):
        self.rows: list[BrandRow] = []
        df: defaultdict[str, int] = defaultdict(int)

        for r in rows:
            # Label and winery name the brand; the SKU string adds the
            # abbreviations a tag actually prints ("SV BL", "CB. S").
            key = frozenset(
                tokens(r.get("Label")) + tokens(r.get("Winery"))
                + tokens(r.get("Assortment (product line)"))
                + tokens(r.get("SKU name (unique)"))
            )
            if not key:
                continue
            row = BrandRow(
                (r.get("Category") or "").strip(), (r.get("Group") or "").strip(),
                (r.get("Winery") or "").strip(), (r.get("Label") or "").strip(),
                (r.get("Assortment (product line)") or "").strip(),
                (r.get("SKU name (unique)") or "").strip(), key,
            )
            self.rows.append(row)
            for tok in key:
                df[tok] += 1

        n = max(1, len(self.rows))
        self.idf = {t: math.log(n / c) for t, c in df.items()}
        self.postings: defaultdict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(self.rows):
            for tok in row.key_tokens:
                self.postings[tok].append(i)

        # Shelf tags abbreviate by truncation — "ETICH GALB" for "Eticheta
        # Galbena", "TAM. ROM" for "Tamaioasa Romaneasca". Indexing every
        # reference token by its leading letters lets a truncated tag word find
        # the full one. Four letters is the shortest prefix that stays
        # discriminating; three collects half the sheet.
        self.by_prefix: defaultdict[str, set] = defaultdict(set)
        for tok in self.idf:
            for n in range(self.MIN_PREFIX, min(len(tok), self.MAX_PREFIX) + 1):
                self.by_prefix[tok[:n]].add(tok)

    def match(self, *texts: Optional[str]) -> Match:
        """Best brand row for a wine, or an unmatched Match."""
        want = set()
        for t in texts:
            want.update(tokens(t))
        if not want:
            return Match(None, 0.0, ())

        anchors = [
            t for t in want
            if len(t) >= self.MIN_ANCHOR_LEN
            and self.idf.get(t, 0.0) >= self.MIN_ANCHOR_IDF
        ]
        if not anchors:
            return Match(None, 0.0, ())

        scores: defaultdict[int, float] = defaultdict(float)
        for tok in anchors:
            for i in self.postings.get(tok, ()):
                scores[i] += self.idf[tok]
        if not scores:
            return Match(None, 0.0, ())

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_i, best = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0

        # Prefer the shortest label among ties: an exact brand beats a row that
        # merely contains the brand plus extra words.
        tied = [i for i, s in ranked if s >= best - 1e-9]
        if len(tied) > 1:
            best_i = min(tied, key=lambda i: (len(self.rows[i].key_tokens),
                                              self.rows[i].label))
            runner = best  # a genuine tie carries no margin

        row = self.rows[best_i]
        shared = tuple(sorted(want & row.key_tokens, key=lambda t: -self.idf[t]))
        if best < self.MIN_SCORE:
            return Match(None, best, shared)
        if len(tied) == 1 and (best - runner) < self.MIN_MARGIN:
            return Match(None, best, shared)
        return Match(row, best, shared)


    def match_wine(self, name, producer, tag_text, label_text) -> Match:
        """Match a wine, cleanest evidence first.

        The tag transcription and bottle-label text are richer than the parsed
        name but far noisier — a multibuy panel or a neighbouring brand caught in
        the same crop contributes tokens that outvote the real one. So the
        parsed name is tried alone first, and the raw text is only consulted
        when that yields nothing.
        """
        for attempt in ((name,), (name, producer), (name, producer, tag_text, label_text)):
            m = self.match(*attempt)
            if m.matched:
                return m
        return m


    def candidates(self, *texts: Optional[str], limit: int = 14) -> list[tuple]:
        """Top scoring reference rows for a wine, best first.

        Used to narrow 10,000 SKUs down to a shortlist a model can actually
        weigh. The index is good at recall and unreliable at the final choice —
        "MARTINI ASTI" retrieves the right row but ranks an Italian Barbera above
        it, because both share the appellation token — so the shortlist is the
        product here, not the winner.
        """
        want = set()
        for t in texts:
            want.update(tokens(t))
        # Retrieval uses *every* token, not just the rare ones that decide a
        # winner. A winery name is common by construction — "COTNARI" tags 200
        # rows, so its IDF is low — and gating retrieval on rarity meant a tag
        # reading "COTNARI ETICH NG" never saw a single Cotnari row. Reviewers
        # reported that repeatedly: the right row was in the sheet and absent
        # from the shortlist. Rarity still ranks; it no longer filters.
        retrieval = {t for t in want if len(t) >= self.MIN_ANCHOR_LEN}
        # Expand each token to the full reference words it could be a truncation
        # of, at a discount: a prefix hit is weaker evidence than an exact word.
        expanded: dict[str, float] = {t: 1.0 for t in retrieval}
        for tok in list(retrieval):
            if len(tok) < self.MIN_PREFIX:
                continue
            for full in self.by_prefix.get(tok[:self.MAX_PREFIX], ()) or \
                    self.by_prefix.get(tok[:self.MIN_PREFIX], ()):
                if full != tok and full.startswith(tok):
                    expanded.setdefault(full, 0.6)
        if not expanded:
            return []
        retrieval = expanded
        scores: defaultdict[int, float] = defaultdict(float)
        for tok, discount in retrieval.items():
            weight = self.idf.get(tok)
            if weight is None:
                continue
            for i in self.postings.get(tok, ()):
                scores[i] += weight * discount

        # One entry per distinct brand: the sheet lists a label once per SKU, and
        # ten volumes of the same wine would otherwise fill the whole shortlist.
        seen: dict[tuple, tuple] = {}
        for i, sc in sorted(scores.items(), key=lambda kv: -kv[1]):
            row = self.rows[i]
            key = (row.group, row.winery, row.label)
            if key not in seen:
                seen[key] = (row, sc, tuple(sorted(want & row.key_tokens,
                                                   key=lambda t: -self.idf[t]))[:4])
            if len(seen) >= limit:
                break
        return list(seen.values())
