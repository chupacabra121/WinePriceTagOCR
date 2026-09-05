"""Turn the answers written by the reading pass back into rows.

The reading pass writes one JSON file per job into ``answers/``. That directory
is the run's durable state: a run interrupted at 3am leaves every answer it had
already produced, and collecting is a pure local operation that can be repeated
as often as you like.

Collection is also where local OCR gets to referee the model. Every job carries
the prices OCR read off its rail at native resolution; if a returned record
disagrees with the one it claims to be using, that is recorded on the row rather
than resolved silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

from pydantic import ValidationError

from .extract import PhotoResult, RegionResult
from .images import PhotoMeta, Region
from .models import BandExtraction
from .output import rows_from_photo
from .prep import Job
from .stores import StoreResolution

# How far a model-reported price may sit from the OCR price it claims before the
# row is flagged. Exact agreement is the norm; this only absorbs rounding.
PRICE_TOLERANCE = 0.005


class AnswerError(RuntimeError):
    pass


def answer_path(work_dir: Path, job_id: str) -> Path:
    return work_dir / "answers" / f"{job_id}.json"


def write_answer(work_dir: Path, job_id: str, payload: dict) -> Path:
    path = answer_path(work_dir, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    tmp.replace(path)
    return path


def load_answer(work_dir: Path, job_id: str) -> Optional[BandExtraction]:
    """Parse one answer, or None if it is absent or unusable."""
    path = answer_path(work_dir, job_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    # A subagent occasionally wraps the object it was asked for.
    if isinstance(payload, dict) and "wines" not in payload:
        for key in ("result", "extraction", "output", "data"):
            if isinstance(payload.get(key), dict):
                payload = payload[key]
                break
    try:
        return BandExtraction.model_validate(payload)
    except ValidationError:
        return None


def pending(jobs: Iterable[Job], work_dir: Path) -> list[Job]:
    """Jobs with no usable answer yet — the resume list."""
    return [j for j in jobs if load_answer(work_dir, j.job_id) is None]


def _region_for(job: Job) -> Region:
    """The job's crop, in absolute pixels, so bboxes map back to the photo."""
    pw, ph = job.photo_size
    x0, y0, x1, y1 = job.crop_box
    return Region(
        label=job.band_label,
        x0=int(x0 * pw), y0=int(y0 * ph),
        x1=max(int(x1 * pw), int(x0 * pw) + 1),
        y1=max(int(y1 * ph), int(y0 * ph) + 1),
        band_index=job.band_index, tile_index=0, tile_count=1,
        contains_tag_rail=True,
    )


def _ocr_check(job: Job, wine) -> tuple[str, Optional[float]]:
    """Compare a returned price against the OCR price it says it used.

    Returns a verdict and the OCR price it was compared against. The verdict is
    the honest one: ``no_ocr_price`` when the rail gave us nothing to check
    against is very different from ``ok``.
    """
    if wine.price is None:
        return ("no_price", None)
    if not job.ocr_prices:
        return ("no_ocr_price", None)

    index = wine.ocr_price_index
    claimed: Optional[float] = None
    if isinstance(index, int) and 1 <= index <= len(job.ocr_prices):
        claimed = job.ocr_prices[index - 1]
        if abs(claimed - wine.price) < PRICE_TOLERANCE:
            return ("ok", claimed)
        # It named a slot but reported a different number. If the number it
        # reported is *some* price on this rail, it most likely mislabelled the
        # index rather than misread the digits.
        if any(abs(p - wine.price) < PRICE_TOLERANCE for p in job.ocr_prices):
            return ("index_mismatch", claimed)
        return ("disagrees", claimed)

    if any(abs(p - wine.price) < PRICE_TOLERANCE for p in job.ocr_prices):
        return ("ok_unindexed", None)
    return ("not_on_rail", None)


# A 2+1 campaign tag prints three figures in a fixed ratio: the single-bottle
# shelf price, the three-bottle total at twice it, and the campaign per-bottle
# price at a third of that. Reading agents found local OCR picking the wrong
# column on essentially every Profi tag, and the same shape appears on the
# cash-and-carry multibuy insets. The ratio is exact, which makes it checkable
# without looking at the image again.
MULTIBUY_RATIOS = (2.0, 3.0, 1 / 3, 0.5)
RATIO_TOLERANCE = 0.01


def multibuy_siblings(price: Optional[float], rail: Sequence[float]) -> list[float]:
    """Other prices on the same rail that stand in a multibuy ratio to ``price``.

    Their presence does not prove the price is wrong — a rail can legitimately
    hold a 20 lei and a 40 lei wine — but on a tag that prints a campaign tier
    it is the signal that says which number was picked.
    """
    if price is None or price <= 0:
        return []
    out: list[float] = []
    for other in rail:
        if other is None or other <= 0 or abs(other - price) < PRICE_TOLERANCE:
            continue
        for ratio in MULTIBUY_RATIOS:
            if abs(other - price * ratio) <= max(RATIO_TOLERANCE, 0.01 * price * ratio):
                out.append(other)
                break
    return out


# A wine that costs under ten lei for a normal bottle is almost always a price
# that lost a digit. The reading pass reported this repeatedly on electronic
# labels: 47,6x came back as 4.76, 146,7x as 1.46, 34,90 as 3.40 — the digits
# were truncated and the decimal point then inserted two from the right.
DROPPED_DIGIT_MAX_PRICE = 10.0
DROPPED_DIGIT_MAX_VOLUME = 1000.0

# Romanian tags print a per-litre figure beside the price. On a 0.75 L bottle it
# is 1.33x the price, and OCR promotes it to a second price often enough that
# agents reported nine such false prices on a single photo. The ratio is exact,
# so a sibling at price/0.75 identifies it without another look at the image.
COMMON_VOLUMES_L = (0.75, 1.5, 2.0, 3.0, 5.0)
# The retailer computes the per-litre figure from the price, so agreement is
# exact to the bani. A loose tolerance here turns ordinary neighbouring prices
# into false hits: 37.99 and 25.49 are two unrelated wines, but 37.99/1.5 is
# 25.33 and a 1.5% window would call that a match.
PER_LITRE_TOLERANCE = 0.02


def looks_like_dropped_digit(price, volume_ml) -> bool:
    """A bottle price too low to be real for its size."""
    if price is None or price <= 0:
        return False
    if volume_ml and volume_ml > DROPPED_DIGIT_MAX_VOLUME:
        return False  # a 3L bag-in-box genuinely can be cheap per bottle
    return price < DROPPED_DIGIT_MAX_PRICE


def per_litre_siblings(price, rail) -> list:
    """Rail prices that are this price's own per-litre figure, or vice versa."""
    if price is None or price <= 0:
        return []
    out = []
    for other in rail:
        if other is None or other <= 0 or abs(other - price) < PRICE_TOLERANCE:
            continue
        for litres in COMMON_VOLUMES_L:
            for a, b in ((price, other), (other, price)):
                if abs(b - a / litres) <= max(PER_LITRE_TOLERANCE, 0.005 * b):
                    out.append(other)
                    break
            else:
                continue
            break
    return out


def assemble(
    jobs: Iterable[Job],
    work_dir: Path,
    root: Path,
    model: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build wine rows, unmatched-tag rows and error rows from the answers."""
    by_photo: dict[str, list[Job]] = {}
    for job in jobs:
        by_photo.setdefault(job.photo_abs, []).append(job)

    rows: list[dict] = []
    unmatched: list[dict] = []
    errors: list[dict] = []

    for photo_abs, photo_jobs in by_photo.items():
        photo_jobs.sort(key=lambda j: j.band_index)
        first = photo_jobs[0]
        pw, ph = first.photo_size
        meta = PhotoMeta(
            path=Path(photo_abs), sha256=first.photo_sha256,
            width=pw, height=ph, taken_at=first.photo_taken_at,
            gps_lat=None, gps_lon=None,
        )
        store = StoreResolution(first.store, first.store_source, None)

        regions: list[RegionResult] = []
        job_by_region: dict[int, Job] = {}
        checks: dict[int, list[tuple[str, Optional[float]]]] = {}
        for job in photo_jobs:
            extraction = load_answer(work_dir, job.job_id)
            if extraction is None:
                errors.append({
                    "photo": job.photo, "stage": "read",
                    "detail": f"no usable answer for job {job.job_id} ({job.band_label})",
                })
                continue
            region = _region_for(job)
            result_region = RegionResult(region, extraction)
            regions.append(result_region)
            job_by_region[id(result_region)] = job
            checks[id(extraction)] = [_ocr_check(job, w) for w in extraction.wines]

        result = PhotoResult(Path(photo_abs), None, regions, [])
        photo_rows, photo_unmatched = rows_from_photo(result, meta, store, model, root)

        # rows_from_photo emits wines in region order, then wine order, which is
        # exactly the order the checks were built in.
        flat = [c for r in regions for c in checks.get(id(r.extraction), [])]
        # Scope the sibling checks to the row's own rail, not the whole photo.
        # A wide shot carries thirty-odd prices, and among that many some pair
        # will land on a 1.33x or 2x ratio by coincidence — pooling them made the
        # flags fire on a third of all rows, which is noise rather than signal.
        flat_rails: list[list[float]] = []
        for r in regions:
            job = job_by_region.get(id(r))
            rail = list(job.ocr_prices) if job else []
            flat_rails.extend([rail] * len(r.extraction.wines))

        for row, (verdict, ocr_price), rail_prices in zip(photo_rows, flat, flat_rails):
            row["ocr_price"] = ocr_price
            row["ocr_price_check"] = verdict
            reasons: list[str] = []
            if verdict in {"disagrees", "not_on_rail", "index_mismatch"}:
                reasons.append(f"price {verdict} local OCR")
            if looks_like_dropped_digit(row.get("price"), row.get("volume_ml")):
                reasons.append(
                    f"price {row['price']} is too low for a bottle this size — "
                    "likely a digit dropped from the tag"
                )
            per_l = per_litre_siblings(row.get("price"), rail_prices)
            if per_l:
                shown = ", ".join(f"{v:.2f}" for v in sorted(set(per_l))[:2])
                reasons.append(
                    f"a per-litre figure ({shown}) sits on this rail — confirm "
                    "the price is not the LEI/L line"
                )
            siblings = multibuy_siblings(row.get("price"), rail_prices)
            if siblings:
                shown = ", ".join(f"{v:.2f}" for v in sorted(set(siblings))[:3])
                reasons.append(
                    f"multibuy tier on this rail ({shown}) — confirm this is the "
                    "single-bottle price"
                )
            if reasons:
                row["review_reasons"] = "; ".join(
                    filter(None, [row.get("review_reasons"), *reasons])
                )
                row["needs_review"] = "yes"

        rows.extend(photo_rows)
        unmatched.extend(photo_unmatched)

    return rows, unmatched, errors
