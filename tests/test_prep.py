"""The prep pass: local OCR, crops, briefs and a resumable manifest.

Runs against the real sample photos, because the whole point of prep is that it
touches real pixels — and it needs no API key, so it is a normal test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wine_ocr.cache import ResponseCache
from wine_ocr.collect import pending, write_answer
from wine_ocr.images import iter_photos
from wine_ocr.prep import (
    brief_path, job_id, prepare, read_manifest, write_briefs, write_manifest,
)
from wine_ocr.vision import BINARY, VisionUnavailable, ensure_binary

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples" / "annabella"

vision_required = pytest.mark.skipif(
    not BINARY.with_suffix(".swift").exists(),
    reason="the Vision helper source is missing",
)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    try:
        ensure_binary()
    except VisionUnavailable as exc:  # pragma: no cover - non-macOS
        pytest.skip(str(exc))
    work = tmp_path_factory.mktemp("work")
    cache = ResponseCache(Path(".cache/vision"))
    photos = iter_photos(SAMPLES)
    result = prepare(photos, SAMPLES.parent, work, cache)
    write_manifest(work / "manifest.jsonl", result.jobs)
    write_briefs(work, result.jobs)
    return work, result


@vision_required
def test_every_sample_photo_yields_at_least_one_shelf(prepared):
    _, result = prepared
    assert result.photos == 4
    assert result.errors == []
    assert result.bands >= 4


@vision_required
def test_the_ground_truth_rail_is_found_in_full(prepared):
    """IMG_5755's top rail is eleven hand-read prices; OCR must find them all."""
    _, result = prepared
    jobs = [j for j in result.jobs if j.photo.endswith("IMG_5755.HEIC")]
    expected = {58.99, 54.59, 54.70, 50.89, 55.99, 25.49, 37.99, 27.99, 34.59, 60.99}
    found = {p for j in jobs for p in j.ocr_prices}
    assert expected <= found, f"missing {sorted(expected - found)}"


@vision_required
def test_every_job_has_a_crop_on_disk(prepared):
    work, result = prepared
    for job in result.jobs:
        path = Path(job.crop_path)
        assert path.exists() and path.stat().st_size > 0
        assert path.parent == work / "crops"


@vision_required
def test_crops_stay_within_the_vision_limit(prepared):
    """A crop larger than the limit is bytes the model will throw away."""
    from PIL import Image
    from wine_ocr.prep import CROP_MAX_EDGE

    _, result = prepared
    for job in result.jobs[:12]:
        with Image.open(job.crop_path) as img:
            assert max(img.size) <= CROP_MAX_EDGE


@vision_required
def test_the_store_comes_from_the_folder(prepared):
    _, result = prepared
    assert {j.store for j in result.jobs} == {"Annabella"}
    assert {j.store_source for j in result.jobs} == {"folder"}


@vision_required
def test_the_manifest_round_trips(prepared):
    work, result = prepared
    again = read_manifest(work / "manifest.jsonl")
    assert [j.job_id for j in again] == [j.job_id for j in result.jobs]
    assert again[0].ocr_prices == result.jobs[0].ocr_prices


@vision_required
def test_one_brief_per_photo_naming_every_crop(prepared):
    work, result = prepared
    for job in result.jobs:
        text = brief_path(work, job.photo_sha256).read_text(encoding="utf-8")
        assert job.crop_path in text
        assert str(work / "answers" / f"{job.job_id}.json") in text
        assert job.ocr_digest.splitlines()[0] in text


@vision_required
def test_a_brief_carries_the_schema_the_agent_must_produce(prepared):
    work, result = prepared
    text = brief_path(work, result.jobs[0].photo_sha256).read_text(encoding="utf-8")
    assert '"pairing_confidence"' in text and '"ocr_price_index"' in text


@vision_required
def test_prep_is_idempotent(prepared, tmp_path):
    """Re-running must reuse job ids, or every answer already written is orphaned."""
    _, result = prepared
    cache = ResponseCache(Path(".cache/vision"))
    again = prepare(iter_photos(SAMPLES), SAMPLES.parent, tmp_path, cache)
    assert [j.job_id for j in again.jobs] == [j.job_id for j in result.jobs]


@vision_required
def test_answers_written_by_hand_clear_their_jobs(prepared, tmp_path):
    """The resume contract: prep, answer some, and only the rest come back."""
    work, result = prepared
    jobs = result.jobs[:3]
    assert len(pending(jobs, tmp_path)) == 3
    write_answer(tmp_path, jobs[0].job_id, {
        "wines": [], "unreadable_tags": [], "non_wine_present": False, "notes": None,
    })
    assert [j.job_id for j in pending(jobs, tmp_path)] == [j.job_id for j in jobs[1:]]


def test_job_ids_change_when_the_crop_moves():
    """A different crop must be a different job, not a silent reuse of an answer."""
    from wine_ocr.layout import Band

    a = Band(0, None, 0.1, 0.2, 0.9, 0.4)
    b = Band(0, None, 0.1, 0.25, 0.9, 0.4)
    assert job_id("sha", a, 0) != job_id("sha", b, 0)
    assert job_id("sha", a, 0) == job_id("sha", a, 0)
