"""Geometry tests for rail detection and band derivation.

These are the rules that decide what the model is ever shown, so most of them
are regression guards on failures actually seen in the photo library: a rail
split into fragments by a sloping shelf, a bottle's brand year adopted as a
price, a whole shelf dropped because its labels were electronic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wine_ocr.layout import (
    MIN_BAND_WIDTH,
    build_bands,
    digest,
    find_rails,
    flipped_lines,
    looks_flipped,
    price_tokens,
)
from wine_ocr.vision import OCRLine, OCRPage


def line(text: str, x0: float, y0: float, *, w: float = 0.04, h: float = 0.010,
         conf: float = 1.0) -> OCRLine:
    return OCRLine(text, conf, x0, y0, x0 + w, y0 + h, int(h * 4000))


def page(*lines: OCRLine) -> OCRPage:
    return OCRPage(Path("test.jpg"), 3000, 4000, tuple(lines))


def rail_at(y: float, values: list[str], *, x0: float = 0.1, step: float = 0.09,
            h: float = 0.010) -> list[OCRLine]:
    return [line(v, x0 + i * step, y, h=h) for i, v in enumerate(values)]


# --------------------------------------------------------------------------
# Price tokens
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("54,59", 54.59),
        ("58.99", 58.99),
        ("|50.89", 50.89),
        ('"25.49', 25.49),
        ("2.69", 2.69),
    ],
)
def test_separated_prices_are_read(text, expected):
    tokens = price_tokens(page(*rail_at(0.5, [text, "10,00"])))
    assert any(abs(t.value - expected) < 0.001 for t in tokens)


def test_electronic_labels_have_their_decimal_point_inferred():
    """Carrefour prints "88" then a smaller "19"; Vision returns "8819"."""
    tokens = price_tokens(page(*rail_at(0.3, ["5149", "8819", "2905", "4435"])))
    assert [t.value for t in tokens] == [51.49, 88.19, 29.05, 44.35]
    assert all(t.decimal_inferred for t in tokens)


@pytest.mark.parametrize("text", ["750-ml", "750 ml", "856L", "O87135", "0.75L"])
def test_a_number_touching_a_letter_is_never_a_price(text):
    """Volumes and barcodes all parse as plausible prices if letters ride along."""
    tokens = price_tokens(page(*rail_at(0.5, [text, "1234", "5678"])))
    assert all(text.strip() not in t.line.text for t in tokens)


def test_small_bare_numbers_are_dropped_when_real_prices_set_the_scale():
    """A vintage on a bottle label is a quarter the height of a shelf price."""
    lines = rail_at(0.5, ["54,59", "37,99", "25,49"], h=0.010)
    lines += [line("2021", 0.2, 0.30, h=0.0025), line("1827", 0.4, 0.31, h=0.0025)]
    values = [t.value for t in price_tokens(page(*lines))]
    assert values == [54.59, 37.99, 25.49]


# --------------------------------------------------------------------------
# Rails
# --------------------------------------------------------------------------


def test_a_sloping_rail_stays_one_rail():
    """A shelf shot from an angle drifts in y across the frame."""
    lines = [line(v, 0.1 + i * 0.09, 0.40 + i * 0.006)
             for i, v in enumerate(["25,99", "14,29", "17,29", "25,49", "25,89"])]
    rails = find_rails(price_tokens(page(*lines)))
    assert len(rails) == 1
    assert len(rails[0].tokens) == 5


def test_two_shelves_stay_two_rails():
    lines = rail_at(0.30, ["54,59", "37,99", "25,49"])
    lines += rail_at(0.62, ["12,99", "18,49", "22,09"])
    assert len(find_rails(price_tokens(page(*lines)))) == 2


def test_widely_spaced_electronic_labels_still_form_a_rail():
    """Carrefour's labels sit a third of the frame apart on a sparse shelf."""
    lines = [line("5149", 0.14, 0.25), line("8819", 0.46, 0.25),
             line("2905", 0.70, 0.25), line("4435", 0.94, 0.25)]
    rails = find_rails(price_tokens(page(*lines)))
    assert len(rails) == 1 and len(rails[0].tokens) == 4


def test_two_brand_years_on_neighbouring_bottles_are_not_a_rail():
    """'1827' is a wine, not 18.27 lei — and two of them sit close together."""
    lines = [line("1827", 0.34, 0.74), line("1827", 0.38, 0.73)]
    assert find_rails(price_tokens(page(*lines))) == []


def test_a_lone_inferred_price_is_not_a_rail():
    assert find_rails(price_tokens(page(line("2021", 0.4, 0.5)))) == []


def test_a_lone_printed_price_is_a_rail():
    """A close-up of one tag is a legitimate photo."""
    rails = find_rails(price_tokens(page(line("54,59", 0.4, 0.5))))
    assert len(rails) == 1


def test_per_litre_small_print_is_not_adopted_as_a_price():
    """Lidl prints '1 L = 53,32 Lei' under the price, in much smaller type."""
    lines = rail_at(0.40, ["39,99", "37,99", "33,49"], h=0.012)
    lines.append(line("53,32", 0.11, 0.404, h=0.003))
    rails = find_rails(price_tokens(page(*lines)))
    assert len(rails) == 1
    assert 53.32 not in [t.value for t in rails[0].tokens]


# --------------------------------------------------------------------------
# Bands
# --------------------------------------------------------------------------


def test_a_band_reaches_up_to_the_shelf_above_it():
    lines = rail_at(0.30, ["54,59", "37,99", "25,49"])
    lines += rail_at(0.62, ["12,99", "18,49", "22,09"])
    bands = [b for b in build_bands(page(*lines)) if not b.is_fallback]
    lower = max(bands, key=lambda b: b.y1)
    assert lower.y0 == pytest.approx(0.31, abs=0.001)  # bottom of the rail above


def test_a_band_ignores_a_neighbouring_fixtures_rail():
    """A snack rack beside the wine bay interleaves in y but not in x."""
    wine_top = rail_at(0.30, ["54,59", "37,99", "25,49"], x0=0.20, step=0.15)
    wine_bottom = rail_at(0.62, ["12,99", "18,49", "22,09"], x0=0.20, step=0.15)
    snacks = rail_at(0.55, ["2,69", "2,69"], x0=0.90, step=0.04)
    bands = build_bands(page(*wine_top, *wine_bottom, *snacks))
    lower = next(b for b in bands if b.slots and b.slots[0].price.value == 12.99)
    # Not 0.71: that is the snack rail, which prices nothing on this shelf.
    assert lower.y0 == pytest.approx(0.31, abs=0.001)


def test_a_single_tag_band_is_widened_for_context():
    bands = build_bands(page(line("54,59", 0.48, 0.5)))
    band = next(b for b in bands if b.slots)
    assert band.x1 - band.x0 == pytest.approx(MIN_BAND_WIDTH, abs=0.001)


def test_bands_are_never_degenerate():
    """A tag at the frame edge must not produce x1 <= x0."""
    for x in (0.0, 0.02, 0.5, 0.97, 0.96):
        bands = build_bands(page(line("54,59", x, 0.5, w=0.03)))
        for band in bands:
            assert band.x1 > band.x0 and band.y1 > band.y0


def test_an_unclaimed_stretch_of_photo_still_gets_read():
    """A shelf whose prices were never resolved must not vanish silently."""
    lines = rail_at(0.10, ["54,59", "37,99", "25,49"])
    lines.append(line("Domeniile Recas", 0.4, 0.70, w=0.2))
    bands = build_bands(page(*lines))
    fallbacks = [b for b in bands if b.is_fallback]
    assert fallbacks, "the lower half of the photo was dropped"
    assert any(b.y0 <= 0.70 <= b.y1 for b in fallbacks)


def test_an_empty_stretch_is_not_read():
    """Bare shelf or ceiling has nothing to say and should not cost a call."""
    bands = build_bands(page(*rail_at(0.10, ["54,59", "37,99"])))
    assert [b for b in bands if b.is_fallback] == []


def test_labels_are_paired_with_the_price_below_them():
    lines = rail_at(0.50, ["54,59", "37,99", "25,49"], x0=0.10, step=0.30)
    lines += [line("Purcari", 0.10, 0.30, w=0.05),
              line("Recas", 0.40, 0.30, w=0.05),
              line("Jidvei", 0.70, 0.30, w=0.05)]
    band = next(b for b in build_bands(page(*lines)) if b.slots)
    assert [s.label_text for s in band.slots] == ["Purcari", "Recas", "Jidvei"]


def test_tag_small_print_is_kept_apart_from_bottle_labels():
    lines = rail_at(0.50, ["54,59", "37,99"], x0=0.10, step=0.30)
    lines += [line("Purcari", 0.10, 0.30, w=0.05),          # above the rail
              line("COTN EUF 0.75", 0.10, 0.505, w=0.05)]   # on the tag
    band = next(b for b in build_bands(page(*lines)) if b.slots)
    assert band.slots[0].label_text == "Purcari"
    assert [l.text for l in band.slots[0].tag_lines] == ["COTN EUF 0.75"]


# --------------------------------------------------------------------------
# Upside-down readings
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["68'0S", "6E'9Z", "66'SS"])
def test_an_upside_down_price_is_detected(text):
    assert looks_flipped(text)


@pytest.mark.parametrize("text", ["54,59", "Purcari", "0.75L", ""])
def test_ordinary_text_is_not_called_upside_down(text):
    assert not looks_flipped(text)


def test_flipped_lines_are_reported_for_the_whole_page():
    assert len(flipped_lines(page(line("68'0S", 0.2, 0.3), line("54,59", 0.4, 0.3)))) == 1


# --------------------------------------------------------------------------
# Digest
# --------------------------------------------------------------------------


def test_the_digest_lists_prices_left_to_right_with_positions():
    lines = rail_at(0.50, ["54,59", "37,99", "25,49"], x0=0.10, step=0.30)
    band = next(b for b in build_bands(page(*lines)) if b.slots)
    text = digest(band)
    assert text.index("54.59") < text.index("37.99") < text.index("25.49")
    assert "x=" in text


def test_the_digest_warns_when_a_decimal_point_was_inferred():
    lines = [line("5149", 0.14, 0.25), line("8819", 0.46, 0.25),
             line("2905", 0.70, 0.25), line("4435", 0.94, 0.25)]
    band = next(b for b in build_bands(page(*lines)) if b.slots)
    assert "DECIMAL POINT INFERRED" in digest(band)


# --------------------------------------------------------------------------
# Prices Vision returned as two boxes
# --------------------------------------------------------------------------


def test_a_price_split_into_two_boxes_is_reassembled():
    """Auchan and the cash-and-carry chains set the bani small and separate."""
    lines = [line("15", 0.10, 0.50, w=0.030, h=0.012),
             line(",99", 0.132, 0.502, w=0.014, h=0.007),
             line("27", 0.30, 0.50, w=0.030, h=0.012),
             line(",49", 0.332, 0.502, w=0.014, h=0.007)]
    values = sorted(t.value for t in price_tokens(page(*lines)))
    assert values == [15.99, 27.49]


def test_a_superscript_split_without_a_separator_is_reassembled():
    lines = [line("170", 0.10, 0.50, w=0.040, h=0.012),
             line("29", 0.142, 0.500, w=0.016, h=0.006),
             line("55", 0.30, 0.50, w=0.030, h=0.012),
             line("99", 0.332, 0.500, w=0.016, h=0.006)]
    assert sorted(t.value for t in price_tokens(page(*lines))) == [55.99, 170.29]


def test_numbers_on_different_rows_are_not_merged():
    """Vertical overlap is what separates one price from two."""
    lines = [line("15", 0.10, 0.30, w=0.030, h=0.012),
             line(",99", 0.132, 0.60, w=0.014, h=0.007)]
    assert [t.value for t in price_tokens(page(*lines))] == []


def test_a_distant_pair_is_not_merged():
    lines = [line("15", 0.10, 0.50, w=0.030, h=0.012),
             line(",99", 0.40, 0.502, w=0.014, h=0.007)]
    assert [t.value for t in price_tokens(page(*lines))] == []


def test_a_lone_reassembled_price_is_not_a_rail():
    """Two boxes that happened to sit together need company to be believed."""
    lines = [line("15", 0.10, 0.50, w=0.030, h=0.012),
             line(",99", 0.132, 0.502, w=0.014, h=0.007)]
    assert find_rails(price_tokens(page(*lines))) == []


# --------------------------------------------------------------------------
# Decoy numbers printed on a tag
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "+ garantie SGR 0,50 Lei",
    "Garanție 0.50 lei",
    "(1 litru = 24.52)",
    "LEI/LTR: 55,72",
    "0.75 L sau 135.87 LEI/L",
    "Cumpara 3 Buc si pretul/Buc devine: 15,39",
    "3 Buc x 15.39 = 46.17 lei",
    "LA MINIM 3 BUCĂȚI",
    "valabil: 12.08.2026 - 18.08.2026",
    "în loc de 24,40",
])
def test_a_decoy_number_on_a_tag_is_never_a_price(text):
    tokens = price_tokens(page(*rail_at(0.5, [text, "54,59", "37,99"])))
    assert all(t.line.text != text for t in tokens)


def test_the_tags_own_article_line_is_kept_with_the_tag():
    """Not treated as a bottle label: it is the best name source in most chains."""
    lines = rail_at(0.50, ["54,59", "37,99"], x0=0.10, step=0.40, h=0.012)
    lines.append(line("ZURZUR FETEASCA NEAGRA DS 3L", 0.10, 0.478, w=0.10, h=0.005))
    lines.append(line("Purcari", 0.10, 0.30, w=0.05))
    band = next(b for b in build_bands(page(*lines)) if b.slots)
    assert "ZURZUR" in band.slots[0].tag_text
    assert band.slots[0].label_text == "Purcari"


def test_other_numbers_on_a_tag_are_surfaced_not_hidden():
    lines = rail_at(0.50, ["54,59", "37,99"], x0=0.10, step=0.40, h=0.012)
    lines.append(line("72,79", 0.10, 0.516, w=0.03, h=0.005))
    band = next(b for b in build_bands(page(*lines)) if b.slots)
    assert any("72,79" in t.line.text for t in band.slots[0].other_prices)
    assert "other numbers printed on this tag" in digest(band)


# --------------------------------------------------------------------------
# The tag's own arithmetic
# --------------------------------------------------------------------------


def slot_with(tag_text: str, price: str = "18,49"):
    lines = rail_at(0.50, [price, "37,99"], x0=0.10, step=0.40, h=0.012)
    lines.append(line(tag_text, 0.10, 0.478, w=0.10, h=0.005))
    band = next(b for b in build_bands(page(*lines)) if b.slots)
    return band.slots[0], band


def test_a_price_is_confirmed_by_its_own_per_litre_line():
    slot, _ = slot_with("Cervus 0,75 L  1 L = 24,65 Lei")
    assert slot.check and slot.check.startswith("confirmed")


def test_an_icon_read_as_a_leading_digit_is_caught_and_diagnosed():
    """Lidl's '18+' roundel turns 18.49 into '918.49' at full confidence."""
    slot, band = slot_with("Cervus 0,75 L  1 L = 24,65 Lei", price="918,49")
    assert slot.check and "DISAGREES" in slot.check
    assert "18.49" in slot.check
    assert "Dropping the leading digit" in slot.check
    assert "DISAGREES" in digest(band)


def test_a_plain_disagreement_is_reported_without_a_repair():
    slot, _ = slot_with("Cervus 0,75 L  1 L = 24,65 Lei", price="52,99")
    assert slot.check and slot.check.startswith("disagrees")


@pytest.mark.parametrize("tag", [
    "Cervus 0,75 L 1 LT 24.65 LEI",
    "Cervus 0.75L (1 litru = 24.65)",
    "Cervus 750 ml LEI/LTR: 24,65",
    "Cervus 0.75 L sau 24.65 LEI/L",
])
def test_the_per_litre_line_is_recognised_in_its_several_shapes(tag):
    slot, _ = slot_with(tag)
    assert slot.check and slot.check.startswith("confirmed"), slot.check


def test_the_reference_litre_is_not_mistaken_for_the_bottle():
    """'1 L = ...' introduces the unit price; it is not a one-litre bottle."""
    slot, _ = slot_with("Cervus 1 L = 24,65 Lei")
    assert slot.check is None


def test_a_tag_with_no_per_litre_line_is_simply_unchecked():
    slot, _ = slot_with("Cervus Cepturum vin alb demisec 0,75 L")
    assert slot.check is None


# --------------------------------------------------------------------------
# What happens where local OCR found nothing
# --------------------------------------------------------------------------


def big_page(*lines: OCRLine) -> OCRPage:
    """A 4284x5712 frame — the shape of an iPhone shot in this library."""
    return OCRPage(Path("test.jpg"), 4284, 5712, tuple(lines))


def test_an_unread_photo_is_tiled_rather_than_sent_whole():
    """Where OCR found nothing the model is the only reader left, so it needs
    more resolution, not a 2.7x downscale of the entire frame."""
    bands = build_bands(big_page(line("Purcari", 0.4, 0.5, w=0.2)))
    assert all(b.is_fallback for b in bands)
    assert len(bands) > 1
    for b in bands:
        assert (b.x1 - b.x0) * 4284 <= 2400
        assert (b.y1 - b.y0) * 5712 <= 2400


def test_fallback_tiles_cover_every_part_of_the_photo_holding_text():
    """Empty tiles are floor or ceiling and are skipped; text is never dropped."""
    texts = [line("Purcari", 0.05, 0.08, w=0.1), line("Recas", 0.85, 0.50, w=0.1),
             line("Jidvei", 0.40, 0.93, w=0.1)]
    bands = build_bands(big_page(*texts))
    for text in texts:
        assert any(b.x0 <= text.cx <= b.x1 and b.y0 <= text.cy <= b.y1
                   for b in bands), f"{text.text} fell outside every crop"


def test_a_tile_with_nothing_in_it_costs_no_call():
    bands = build_bands(big_page(line("Purcari", 0.05, 0.05, w=0.1)))
    assert all(b.y0 < 0.5 for b in bands)


def test_fallback_tiling_is_bounded():
    """A panorama must not explode into hundreds of crops."""
    page = OCRPage(Path("p.jpg"), 16348, 3834, (line("x", 0.5, 0.5, w=0.1),))
    bands = build_bands(page)
    assert len(bands) <= 9


def test_a_small_photo_is_not_tiled():
    page = OCRPage(Path("s.jpg"), 1200, 1600, (line("Purcari", 0.4, 0.5, w=0.2),))
    assert len(build_bands(page)) == 1
