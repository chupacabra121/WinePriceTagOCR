"""End-to-end run against a stubbed API, using a real sample photo.

Exercises the whole chain — EXIF orientation, layout pass, band tiling, crop
encoding, response caching, row flattening, dedup, and both writers — so the
only untested link when a real key is present is the model's answer itself.
"""

import json
from pathlib import Path

import pytest

from wine_ocr import extract as ex
from wine_ocr.cache import ResponseCache
from wine_ocr.images import load_oriented, read_meta
from wine_ocr.output import COLUMNS, deduplicate, rows_from_photo, write_csv, write_xlsx
from wine_ocr.stores import resolve_store

SAMPLE = Path("data/samples/annabella/IMG_5755.HEIC")
ROOT = Path("data/samples")  # photos live in per-store folders under a common root
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample photo not present")

LAYOUT = {
    "photo_kind": "shelf",
    "store_name_visible": "Annabella",
    "currency_guess": "RON",
    "notes": None,
    "bands": [
        {
            "label": "shelf 1/2, rose", "contains_wine": True,
            "bottles_y0": 0.107, "bottles_y1": 0.262,
            "tags_y0": 0.276, "tags_y1": 0.320, "x0": 0.185, "x1": 0.795,
        },
        {
            "label": "shelf 2/2, crisps", "contains_wine": False,
            "bottles_y0": 0.40, "bottles_y1": 0.55,
            "tags_y0": 0.56, "tags_y1": 0.59, "x0": 0.18, "x1": 0.80,
        },
    ],
}

WINE = {
    "wine_name": "Cotnari Euforia Busuioaca Roze Sec 0.75L",
    "producer": "Cotnari", "vintage": None, "wine_type": "rose", "sweetness": "dry",
    "grape_varieties": ["Busuioaca de Bohotin"], "region": "Cotnari",
    "country": "Romania", "volume_ml": 750.0, "abv_percent": None,
    "price": 54.59, "currency": "RON", "price_text": "54,59", "price_kind": "shelf",
    "original_price": None, "unit_price_text": "72,79 RON/L", "promo_text": None,
    "raw_tag_text": "COTNARI EUFORIA BUSUIOACA ROZE DO 0.75 L  54,59  72,79 RON/L",
    "raw_label_text": "Euforia", "ocr_price_index": 2, "name_source": "both",
    "pairing_confidence": "high", "pairing_note": None,
    "bottle_bbox": {"x0": 0.2, "y0": 0.1, "x1": 0.3, "y1": 0.6},
    "tag_bbox": {"x0": 0.2, "y0": 0.8, "x1": 0.3, "y1": 0.95},
}

EXTRACTION = {
    "wines": [WINE],
    "unreadable_tags": [
        {"raw_text": "?? 60.99", "price": 60.99, "currency": "RON",
         "reason": "tag cut off at frame edge", "tag_bbox": None}
    ],
    "non_wine_present": False,
    "notes": None,
}


class _Stub:
    """Stands in for anthropic.Anthropic, returning canned structured output."""

    def __init__(self):
        self.calls = []
        self.messages = self

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs["output_config"]["format"]["schema"]
        is_layout = "bands" in schema.get("properties", {})
        payload = LAYOUT if is_layout else EXTRACTION
        return _StubStream(payload)


class _StubStream:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        text = json.dumps(self.payload)
        return type(
            "Msg", (), {
                "content": [type("B", (), {"type": "text", "text": text})()],
                "stop_reason": "end_turn",
                "usage": type("U", (), {
                    "input_tokens": 4800, "output_tokens": 700,
                    "cache_read_input_tokens": 1200, "cache_creation_input_tokens": 0,
                })(),
            },
        )()


def _run(tmp_path, tiling="auto"):
    client = _Stub()
    settings = ex.Settings(tiling=tiling, workers=1)
    cache = ResponseCache(tmp_path / "cache", enabled=True)
    usage = ex.Usage()
    meta = read_meta(SAMPLE)
    img = load_oriented(SAMPLE)
    result = ex.extract_photo(
        client, settings, img, SAMPLE, meta.sha256, None, meta.taken_at, cache, usage
    )
    return client, result, meta, cache, usage


def test_full_run_produces_a_table(tmp_path):
    client, result, meta, cache, usage = _run(tmp_path)

    assert result.layout is not None
    assert result.layout.store_name_visible == "Annabella"
    # The crisps shelf is skipped, so only the wine band is read.
    assert all("crisps" not in r.region.label for r in result.regions)
    assert result.regions and all(r.error is None for r in result.regions)

    store = resolve_store(
        SAMPLE, ROOT, None, {}, result.layout.store_name_visible
    )
    assert store.store == "Annabella"  # from the folder name

    rows, unmatched = rows_from_photo(result, meta, store, "claude-opus-5", ROOT)
    assert rows and unmatched

    final = deduplicate(rows)
    primary = [r for r in final if not r["duplicate_of"]]
    # The same wine appears in both overlapping tiles; dedup keeps one.
    assert len(primary) == 1

    row = primary[0]
    assert row["price"] == 54.59
    assert row["currency"] == "RON"
    assert row["volume_ml"] == 750
    assert row["price_per_litre"] == 72.79
    assert row["price_check"] == "ok"
    assert row["unit_price_check"] == "ok"      # agrees with the printed RON/L
    assert row["needs_review"] == ""            # nothing to flag
    assert row["store_source"] == "folder"
    assert row["photo"] == "annabella/IMG_5755.HEIC"

    # Boxes were mapped out of tile space into whole-photo fractions.
    x0, y0, x1, y1 = (float(v) for v in row["bottle_bbox"].split(","))
    assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0

    write_csv(tmp_path / "wines.csv", final, COLUMNS)
    write_xlsx(tmp_path / "wines.xlsx", final, unmatched, [])
    assert (tmp_path / "wines.csv").exists()
    assert (tmp_path / "wines.xlsx").stat().st_size > 4000


def test_images_reach_the_model_at_native_resolution(tmp_path):
    """Regression guard on the finding that motivated tiling."""
    import base64
    import io

    from PIL import Image

    client, _, _, _, _ = _run(tmp_path)
    extraction_calls = [
        c for c in client.calls
        if "bands" not in c["output_config"]["format"]["schema"].get("properties", {})
    ]
    assert extraction_calls
    for call in extraction_calls:
        block = call["messages"][0]["content"][0]
        img = Image.open(io.BytesIO(base64.b64decode(block["source"]["data"])))
        assert max(img.size) <= 2576


def test_system_prompt_carries_a_cache_breakpoint(tmp_path):
    client, _, _, _, _ = _run(tmp_path)
    for call in client.calls:
        assert call["system"][-1]["cache_control"] == {"type": "ephemeral"}


def test_second_run_is_served_from_cache(tmp_path):
    client1, _, _, _, _ = _run(tmp_path)
    first = len(client1.calls)
    assert first > 0

    # Same cache directory, fresh client: nothing should hit the API.
    client2 = _Stub()
    settings = ex.Settings(tiling="auto", workers=1)
    cache = ResponseCache(tmp_path / "cache", enabled=True)
    meta = read_meta(SAMPLE)
    ex.extract_photo(
        client2, settings, load_oriented(SAMPLE), SAMPLE, meta.sha256, None,
        meta.taken_at, cache, ex.Usage(),
    )
    assert client2.calls == []
    assert cache.hits > 0


def test_whole_tiling_makes_exactly_one_call(tmp_path):
    client, result, _, _, _ = _run(tmp_path, tiling="whole")
    assert len(client.calls) == 1        # no layout pass, no tiles
    assert len(result.regions) == 1
