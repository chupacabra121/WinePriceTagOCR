"""Tests for the HTML review sheet."""

import base64
import io
import re
from pathlib import Path

import pytest
from PIL import Image

from wine_ocr.review import THUMB_HEIGHT, THUMB_WIDTH, build_report

SAMPLES = Path("data/samples")
PHOTO = "annabella/IMG_5755.HEIC"
pytestmark = pytest.mark.skipif(
    not (SAMPLES / PHOTO).exists(), reason="sample photo not present"
)


def _row(**kw):
    base = {
        "row_id": "W00001", "photo": PHOTO, "wine_name": "Cotnari Euforia",
        "price": "54.59", "currency": "RON", "volume_ml": "750",
        "needs_review": "", "review_reasons": "", "raw_tag_text": "COTNARI EUFORIA",
        "bottle_bbox": "0.25,0.10,0.31,0.27", "tag_bbox": "0.25,0.278,0.305,0.315",
        "shelf": "shelf 1/4", "price_check": "ok", "unit_price_check": "ok",
        "duplicate_of": "",
    }
    base.update(kw)
    return base


def _images(doc):
    return [
        Image.open(io.BytesIO(base64.b64decode(b)))
        for b in re.findall(r"base64,([A-Za-z0-9+/=]+)", doc)
    ]


def test_report_embeds_a_crop_from_the_real_photo():
    doc, rendered, crops = build_report([_row()], SAMPLES)
    assert (rendered, crops) == (1, 1)
    assert "Cotnari Euforia" in doc and "54.59" in doc
    assert len(_images(doc)) == 1


def test_thumbnails_are_bounded_on_both_axes():
    """Bottle boxes are tall and narrow; unbounded height ruins the sheet."""
    doc, _, _ = build_report([_row(bottle_bbox="0.25,0.05,0.30,0.95")], SAMPLES)
    (img,) = _images(doc)
    assert img.width <= THUMB_WIDTH
    assert img.height <= THUMB_HEIGHT


def test_falls_back_to_the_tag_box_when_no_bottle_box():
    doc, _, crops = build_report([_row(bottle_bbox="")], SAMPLES)
    assert crops == 1


def test_row_without_any_box_still_renders():
    doc, rendered, crops = build_report([_row(bottle_bbox="", tag_bbox="")], SAMPLES)
    assert (rendered, crops) == (1, 0)
    assert "no crop available" in doc


def test_malformed_box_does_not_break_the_report():
    doc, rendered, crops = build_report(
        [_row(bottle_bbox="not,a,box"), _row(bottle_bbox="0.1,0.1")], SAMPLES
    )
    assert rendered == 2 and crops == 0


def test_missing_photo_is_reported_not_fatal():
    doc, rendered, crops = build_report([_row(photo="nope/missing.HEIC")], SAMPLES)
    assert rendered == 1 and crops == 0
    assert "photo not found" in doc


def test_flagged_rows_are_marked_and_counted():
    doc, _, _ = build_report(
        [_row(needs_review="yes", review_reasons="no price", price="")], SAMPLES
    )
    assert "flagged" in doc
    assert "no price" in doc


def test_html_is_escaped():
    doc, _, _ = build_report(
        [_row(wine_name="<script>alert(1)</script>", raw_tag_text="a & b")], SAMPLES
    )
    assert "<script>alert(1)</script>" not in doc
    assert "&lt;script&gt;" in doc
    assert "a &amp; b" in doc


def test_rows_are_grouped_by_photo():
    doc, rendered, _ = build_report(
        [_row(), _row(photo="annabella/IMG_5756.HEIC")], SAMPLES
    )
    assert rendered == 2
    assert doc.count("<h2>") == 2


def test_accepts_a_generator():
    """build_report iterates twice; a generator must not silently empty out."""
    doc, rendered, _ = build_report((r for r in [_row(), _row()]), SAMPLES)
    assert rendered == 2
    assert "<b>2</b><span>rows</span>" in doc


def test_report_is_self_contained():
    doc, _, _ = build_report([_row()], SAMPLES)
    # No external fetches: everything inline, so it works offline and by email.
    assert "<style>" in doc
    assert not re.search(r'src="(?!data:)', doc)
    assert not re.search(r'<link[^>]+href=', doc)
