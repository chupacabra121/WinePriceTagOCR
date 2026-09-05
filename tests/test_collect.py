"""Collecting agent answers back into rows, and refereeing them against OCR."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wine_ocr.collect import assemble, load_answer, pending, write_answer
from wine_ocr.mirror import write_mirror
from wine_ocr.output import deduplicate
from wine_ocr.prep import Job

WINE = {
    "wine_name": "Cotnari Euforia Busuioaca Roze Sec 0.75L",
    "producer": "Cotnari", "vintage": None, "wine_type": "rose", "sweetness": "dry",
    "grape_varieties": ["Busuioaca de Bohotin"], "region": "Cotnari",
    "country": "Romania", "volume_ml": 750.0, "abv_percent": None,
    "price": 54.59, "currency": "RON", "price_text": "54,59", "price_kind": "shelf",
    "original_price": None, "unit_price_text": "72,79 RON/L", "promo_text": None,
    "raw_tag_text": "COTNARI EUFORIA 0.75 L 54,59", "raw_label_text": "Euforia",
    "ocr_price_index": 2, "name_source": "both",
    "pairing_confidence": "high", "pairing_note": None,
    "bottle_bbox": {"x0": 0.2, "y0": 0.1, "x1": 0.3, "y1": 0.6},
    "tag_bbox": {"x0": 0.2, "y0": 0.8, "x1": 0.3, "y1": 0.95},
}


def extraction(*wines, notes=None):
    return {
        "wines": [dict(WINE, **w) for w in (wines or [{}])],
        "unreadable_tags": [], "non_wine_present": False, "notes": notes,
    }


def job(job_id="abc123", prices=(58.99, 54.59, 54.70), band=0) -> Job:
    return Job(
        job_id=job_id, photo="Annabella/IMG_5755.HEIC",
        photo_abs="/photos/Annabella/IMG_5755.HEIC", photo_sha256="f" * 64,
        photo_size=[4284, 5712], store="Annabella", store_source="folder",
        band_index=band, band_label=f"shelf {band + 1} (3 prices)",
        crop_path="/work/crops/abc123.jpg", crop_box=[0.2, 0.13, 0.81, 0.33],
        crop_size=[1600, 525], ocr_prices=list(prices),
        ocr_digest="1. price 58.99\n2. price 54.59\n3. price 54.70",
        photo_taken_at="2026-06-14 13:01:34",
    )


def collect_one(tmp_path: Path, payload: dict, j: Job | None = None):
    j = j or job()
    write_answer(tmp_path, j.job_id, payload)
    return assemble([j], tmp_path, Path("/photos"), "test-model")


# --------------------------------------------------------------------------
# Reading answers back
# --------------------------------------------------------------------------


def test_a_missing_answer_leaves_the_job_pending(tmp_path):
    assert pending([job()], tmp_path) == [job()] or len(pending([job()], tmp_path)) == 1


def test_a_written_answer_clears_the_job(tmp_path):
    write_answer(tmp_path, "abc123", extraction())
    assert pending([job()], tmp_path) == []


def test_malformed_json_leaves_the_job_pending(tmp_path):
    """A half-written file must be retried, not silently swallowed."""
    path = tmp_path / "answers" / "abc123.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"wines": [')
    assert len(pending([job()], tmp_path)) == 1


def test_an_answer_that_misses_the_schema_leaves_the_job_pending(tmp_path):
    write_answer(tmp_path, "abc123", {"wines": [{"wine_name": "x"}]})
    assert len(pending([job()], tmp_path)) == 1


def test_a_wrapped_answer_is_still_understood(tmp_path):
    """Agents occasionally nest the object under a key of their own."""
    write_answer(tmp_path, "abc123", {"result": extraction()})
    assert load_answer(tmp_path, "abc123") is not None


# --------------------------------------------------------------------------
# Refereeing the price against local OCR
# --------------------------------------------------------------------------


def test_a_price_matching_its_claimed_ocr_slot_passes(tmp_path):
    rows, _, _ = collect_one(tmp_path, extraction())
    assert rows[0]["ocr_price_check"] == "ok"
    assert rows[0]["ocr_price"] == 54.59
    assert "OCR" not in (rows[0]["review_reasons"] or "")


def test_a_price_not_on_the_rail_is_flagged(tmp_path):
    rows, _, _ = collect_one(tmp_path, extraction({"price": 99.99, "ocr_price_index": 2}))
    assert rows[0]["ocr_price_check"] == "disagrees"
    assert rows[0]["needs_review"] == "yes"


def test_a_mislabelled_index_is_told_apart_from_a_misread_price(tmp_path):
    """Reporting a real rail price under the wrong slot number is a lesser sin."""
    rows, _, _ = collect_one(tmp_path, extraction({"price": 58.99, "ocr_price_index": 2}))
    assert rows[0]["ocr_price_check"] == "index_mismatch"


def test_a_price_on_the_rail_without_an_index_still_passes(tmp_path):
    rows, _, _ = collect_one(tmp_path, extraction({"price": 54.70, "ocr_price_index": None}))
    assert rows[0]["ocr_price_check"] == "ok_unindexed"


def test_an_invented_price_is_flagged(tmp_path):
    rows, _, _ = collect_one(tmp_path, extraction({"price": 12.34, "ocr_price_index": None}))
    assert rows[0]["ocr_price_check"] == "not_on_rail"
    assert rows[0]["needs_review"] == "yes"


def test_a_band_with_no_ocr_prices_says_so_rather_than_passing(tmp_path):
    """'ok' would claim a check happened; nothing was checked."""
    rows, _, _ = collect_one(tmp_path, extraction(), job(prices=()))
    assert rows[0]["ocr_price_check"] == "no_ocr_price"


def test_checks_line_up_with_rows_across_several_bands(tmp_path):
    """Two bands of one photo must not have their verdicts crossed."""
    good, bad = job("aaa", band=0), job("bbb", band=1)
    write_answer(tmp_path, "aaa", extraction())
    write_answer(tmp_path, "bbb", extraction({"price": 12.34, "ocr_price_index": None}))
    rows, _, _ = assemble([good, bad], tmp_path, Path("/photos"), "m")
    by_price = {r["price"]: r["ocr_price_check"] for r in rows}
    assert by_price[54.59] == "ok"
    assert by_price[12.34] == "not_on_rail"


def test_an_unanswered_band_becomes_an_error_row(tmp_path):
    rows, _, errors = assemble([job()], tmp_path, Path("/photos"), "m")
    assert rows == []
    assert len(errors) == 1 and "no usable answer" in errors[0]["detail"]


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_bounding_boxes_map_back_to_the_whole_photo(tmp_path):
    """A box is reported against the crop; the row must locate it in the photo."""
    rows, _, _ = collect_one(tmp_path, extraction())
    x0 = float(rows[0]["bottle_bbox"].split(",")[0])
    # crop starts at x=0.2 of the photo and is 0.61 wide; 0.2 into that crop:
    assert x0 == pytest.approx(0.2 + 0.2 * 0.61, abs=0.01)


def test_the_row_carries_the_store_and_photo_it_came_from(tmp_path):
    rows, _, _ = collect_one(tmp_path, extraction())
    assert rows[0]["store"] == "Annabella"
    assert rows[0]["photo"] == "Annabella/IMG_5755.HEIC"
    assert rows[0]["shelf"] == "shelf 1 (3 prices)"


# --------------------------------------------------------------------------
# Mirrored output
# --------------------------------------------------------------------------


def test_the_output_mirrors_the_input_folders(tmp_path):
    rows = [
        {"photo": "Discounter - LIDL/a.jpg", "price": 10.0, "currency": "RON",
         "wine_name": "A", "store": "Lidl"},
        {"photo": "Discounter - LIDL/b.jpg", "price": 20.0, "currency": "RON",
         "wine_name": "B", "store": "Lidl"},
        {"photo": "Hypermarket - Kaufland/c.jpg", "price": 30.0, "currency": "RON",
         "wine_name": "C", "store": "Kaufland"},
    ]
    index = write_mirror(tmp_path, rows, [])
    assert (tmp_path / "Discounter - LIDL" / "Discounter - LIDL.csv").exists()
    assert (tmp_path / "Hypermarket - Kaufland" / "Hypermarket - Kaufland.csv").exists()
    assert (tmp_path / "index.csv").exists()
    lidl = next(e for e in index if e["folder"] == "Discounter - LIDL")
    assert lidl["wines"] == 2 and lidl["median_price"] == 15.0


def test_duplicates_are_left_out_of_the_folder_sheets(tmp_path):
    rows = [
        {"photo": "S/a.jpg", "price": 10.0, "wine_name": "A", "store": "S"},
        {"photo": "S/b.jpg", "price": 10.0, "wine_name": "A", "store": "S",
         "duplicate_of": "W00001"},
    ]
    index = write_mirror(tmp_path, rows, [])
    assert index[0]["wines"] == 1


def test_photos_in_the_root_still_get_a_sheet(tmp_path):
    index = write_mirror(tmp_path, [{"photo": "a.jpg", "price": 1.0,
                                     "wine_name": "A", "store": "S"}], [])
    assert (tmp_path / "wines.csv").exists()
    assert index[0]["folder"] == "."


# --------------------------------------------------------------------------
# Multibuy tiers printed on the same tag
# --------------------------------------------------------------------------


def test_a_campaign_tier_on_the_rail_is_flagged():
    """Profi's 2+1 tags print shelf / 2x / a third, in exact ratio."""
    from wine_ocr.collect import multibuy_siblings

    rail = [27.13, 81.38, 40.69]  # campaign per-bottle, 3-bottle total, shelf
    assert multibuy_siblings(40.69, rail) == [81.38]
    assert 27.13 in multibuy_siblings(81.38, rail)


def test_ordinary_neighbours_are_not_called_multibuy():
    from wine_ocr.collect import multibuy_siblings

    assert multibuy_siblings(37.99, [25.49, 54.59, 60.99]) == []


def test_a_missing_price_has_no_siblings():
    from wine_ocr.collect import multibuy_siblings

    assert multibuy_siblings(None, [10.0, 20.0]) == []


def test_the_multibuy_flag_reaches_the_row(tmp_path):
    j = job(prices=(27.13, 81.38, 40.69))
    write_answer(tmp_path, j.job_id, extraction({"price": 40.69, "ocr_price_index": 3}))
    rows, _, _ = assemble([j], tmp_path, Path("/photos"), "m")
    assert rows[0]["needs_review"] == "yes"
    assert "multibuy tier" in rows[0]["review_reasons"]


# --------------------------------------------------------------------------
# False prices the reading pass reported again and again
# --------------------------------------------------------------------------


def test_a_bottle_price_under_ten_lei_is_flagged():
    """OCR truncates a digit, then puts the point two from the right: 47,6x -> 4.76."""
    from wine_ocr.collect import looks_like_dropped_digit

    assert looks_like_dropped_digit(4.76, 750.0)
    assert looks_like_dropped_digit(1.46, 750.0)
    assert not looks_like_dropped_digit(58.99, 750.0)


def test_a_cheap_bag_in_box_is_not_mistaken_for_a_dropped_digit():
    """3L boxes genuinely sell near the threshold; size is the discriminator."""
    from wine_ocr.collect import looks_like_dropped_digit

    assert not looks_like_dropped_digit(9.99, 3000.0)


def test_a_per_litre_figure_on_the_rail_is_spotted():
    """A 0.75L tag prints price/0.75 beside the price; OCR promotes it."""
    from wine_ocr.collect import per_litre_siblings

    assert per_litre_siblings(61.39, [81.85]) == [81.85]   # 61.39 / 0.75
    assert per_litre_siblings(41.69, [55.59]) == [55.59]


def test_ordinary_neighbouring_prices_are_not_called_per_litre():
    """The retailer's figure is exact, so the window must not admit coincidence."""
    from wine_ocr.collect import per_litre_siblings

    for price, other in ((37.99, 25.49), (58.99, 54.59), (25.49, 54.70)):
        assert per_litre_siblings(price, [other]) == [], (price, other)


def test_the_dropped_digit_flag_reaches_the_row(tmp_path):
    j = job(prices=(4.76, 54.59, 54.70))
    write_answer(tmp_path, j.job_id, extraction({"price": 4.76, "ocr_price_index": 1}))
    rows, _, _ = assemble([j], tmp_path, Path("/photos"), "m")
    assert rows[0]["needs_review"] == "yes"
    assert "digit dropped" in rows[0]["review_reasons"]
