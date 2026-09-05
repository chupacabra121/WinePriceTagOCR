"""Build the work queue: crops, OCR digests, and one job per shelf band.

This is the half of the pipeline that costs nothing. Everything a model needs in
order to read a shelf is prepared locally — the photo is decoded, OCR'd at
native resolution, cut into bands by its tag rails, and each band is written out
as a JPEG next to a text digest of what OCR read there.

Splitting prep from extraction is what makes an unattended overnight run
practical: prep is deterministic and re-runnable, so a run that dies halfway can
be resumed without redoing any of it, and the expensive half only ever sees work
that is already fully specified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from PIL import Image

from .cache import ResponseCache
from .images import PhotoMeta, load_oriented, read_meta
from .layout import Band, build_bands, digest, flipped_lines
from .stores import StoreResolution, load_store_map, resolve_store
from .vision import DEFAULT_MAX_TILE, OCRPage, read_pages

# Part of every job id, so changing how crops or digests are built produces new
# jobs rather than silently reusing answers written against the old shape.
PREP_VERSION = "2026-08-17.1"

# Long edge of a band crop. The model does not need native resolution here —
# the native-resolution reading is supplied as text alongside — so this is set
# for legible bottle labels rather than for legible small print.
CROP_MAX_EDGE = 1600
CROP_QUALITY = 88


@dataclass
class Job:
    """One band of one photo: everything a model call needs, and nothing else."""

    job_id: str
    photo: str            # relative to the run root
    photo_abs: str
    photo_sha256: str
    photo_size: list[int]
    store: str
    store_source: str
    band_index: int
    band_label: str
    crop_path: str
    crop_box: list[float]  # x0, y0, x1, y1 as fractions of the oriented photo
    crop_size: list[int]
    ocr_prices: list[float]
    ocr_digest: str
    photo_taken_at: Optional[str]
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def job_id(photo_sha: str, band: Band, index: int) -> str:
    raw = "|".join(
        [
            PREP_VERSION, photo_sha, str(index),
            f"{band.x0:.4f},{band.y0:.4f},{band.x1:.4f},{band.y1:.4f}",
            str(len(band.slots)),
        ]
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _write_crop(img: Image.Image, band: Band, dest: Path) -> tuple[int, int]:
    w, h = img.size
    box = (
        int(band.x0 * w), int(band.y0 * h),
        max(int(band.x1 * w), int(band.x0 * w) + 1),
        max(int(band.y1 * h), int(band.y0 * h) + 1),
    )
    crop = img.crop(box)
    longest = max(crop.size)
    if longest > CROP_MAX_EDGE:
        scale = CROP_MAX_EDGE / longest
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
            Image.LANCZOS,
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dest, format="JPEG", quality=CROP_QUALITY, optimize=True)
    return crop.size


def _band_warnings(page: OCRPage, band: Band) -> list[str]:
    out: list[str] = []
    if band.is_fallback:
        out.append(
            "local OCR found no tag rail here, so this crop is a stretch of the "
            "photo that no shelf claimed rather than a shelf. Expect either "
            "nothing readable, or a shelf whose prices OCR could not resolve — "
            "in which case read the prices off the image yourself and say so."
        )
    if flipped_lines(page):
        out.append(
            "local OCR produced at least one upside-down reading in this photo, "
            "so its prices are less trustworthy than usual"
        )
    if band.rail is not None and (band.y1 - band.y0) < 0.04:
        out.append("band is very short — the bottles above the rail may be missing")
    duplicates = [p for p in band.prices if band.prices.count(p) > 1]
    if len(set(duplicates)) >= 3:
        out.append("many repeated prices on this rail — check for a multi-buy strip")
    return out


@dataclass
class PrepResult:
    jobs: list[Job]
    photos: int
    bands: int
    errors: list[dict]
    metas: dict[str, PhotoMeta]


def prepare(
    photos: Sequence[Path],
    root: Path,
    work_dir: Path,
    cache: ResponseCache,
    *,
    store_override: Optional[str] = None,
    store_config: Optional[Path] = None,
    max_tile: int = DEFAULT_MAX_TILE,
    on_photo=None,
) -> PrepResult:
    """OCR every photo, cut it into bands, and write one job per band."""
    work_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = work_dir / "crops"
    store_map = load_store_map(store_config)

    metas: dict[str, PhotoMeta] = {}
    errors: list[dict] = []
    usable: list[Path] = []
    shas: list[str] = []
    for path in photos:
        try:
            meta = read_meta(path)
        except Exception as exc:
            errors.append({"photo": str(path), "stage": "read", "detail": f"{type(exc).__name__}: {exc}"})
            continue
        metas[str(path)] = meta
        usable.append(path)
        shas.append(meta.sha256)

    jobs: list[Job] = []
    bands_total = 0

    for path, page in zip(usable, read_pages(usable, shas, cache, max_tile=max_tile)):
        meta = metas[str(path)]
        if page.error:
            errors.append({"photo": str(path), "stage": "ocr", "detail": page.error})
            continue

        store = resolve_store(path, root, store_override, store_map, None)
        try:
            img = load_oriented(path)
        except Exception as exc:
            errors.append({"photo": str(path), "stage": "decode", "detail": f"{type(exc).__name__}: {exc}"})
            continue

        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)

        bands = build_bands(page)
        bands_total += len(bands)
        for index, band in enumerate(bands):
            jid = job_id(meta.sha256, band, index)
            crop_path = crops_dir / f"{jid}.jpg"
            if not crop_path.exists():
                size = _write_crop(img, band, crop_path)
            else:
                with Image.open(crop_path) as existing:
                    size = existing.size
            jobs.append(
                Job(
                    job_id=jid,
                    photo=rel,
                    photo_abs=str(path),
                    photo_sha256=meta.sha256,
                    photo_size=[meta.width, meta.height],
                    store=store.store,
                    store_source=store.source,
                    band_index=index,
                    band_label=band.label,
                    crop_path=str(crop_path),
                    crop_box=[round(band.x0, 4), round(band.y0, 4),
                              round(band.x1, 4), round(band.y1, 4)],
                    crop_size=list(size),
                    ocr_prices=[round(p, 2) for p in band.prices],
                    ocr_digest=digest(band),
                    photo_taken_at=meta.taken_at,
                    warnings=_band_warnings(page, band),
                )
            )
        img.close()
        if on_photo:
            on_photo(path, len(bands))

    return PrepResult(jobs, len(usable), bands_total, errors, metas)


def brief_path(work_dir: Path, photo_sha: str) -> Path:
    return work_dir / "briefs" / f"{photo_sha[:16]}.md"


def write_briefs(work_dir: Path, jobs: Sequence[Job]) -> list[Path]:
    """One self-contained instruction file per photo.

    The reading pass is done by agents, which have a filesystem but no way to be
    handed a large payload. A brief is therefore the whole contract in one file:
    the standing instructions, the schema, and for each band of that photo the
    crop to look at and the path to write the answer to.

    Grouping by photo rather than by band is what keeps the run tractable — 448
    photos is a manageable number of agent turns, ~2500 bands is not — and it
    lets one reader see every shelf of a fixture at once, which is exactly the
    context needed to notice that two rails hold the same wine.
    """
    from .prompts import BAND_SYSTEM, band_user_prompt
    from .schema import json_schema_for
    from .models import BandExtraction

    schema = json.dumps(json_schema_for(BandExtraction), ensure_ascii=False, indent=1)

    by_photo: dict[str, list[Job]] = {}
    for job in jobs:
        by_photo.setdefault(job.photo_sha256, []).append(job)

    written: list[Path] = []
    for sha, group in by_photo.items():
        group.sort(key=lambda j: j.band_index)
        first = group[0]
        out = [
            f"# Read the wines in `{first.photo}`",
            "",
            f"Store: **{first.store}**"
            + (f"  ·  captured {first.photo_taken_at}" if first.photo_taken_at else ""),
            "",
            f"This photo was cut into {len(group)} shelf crop(s). Work through them in "
            "order. For each one: open the image with the Read tool, read it against "
            "the instructions below, then write the resulting JSON object to the "
            "answer path given for that crop, using the Write tool.",
            "",
            "Write one file per crop. Write nothing else, and do not modify any "
            "other file. If a crop turns out to hold no wine at all, still write an "
            "answer for it, with an empty `wines` list and a note saying why.",
            "",
            "---",
            "",
            "## Standing instructions",
            "",
            BAND_SYSTEM,
            "",
            "## The exact JSON shape to write",
            "",
            "Every field is required; use `null` for anything you cannot determine.",
            "",
            "```json",
            schema,
            "```",
        ]
        for n, job in enumerate(group, start=1):
            out += [
                "",
                "---",
                "",
                f"## Crop {n} of {len(group)} — {job.band_label}",
                "",
                f"- Image to read: `{job.crop_path}`",
                f"- Write the answer to: `{answer_path_for(work_dir, job.job_id)}`",
                "",
                band_user_prompt(
                    job.band_label, Path(job.photo).name, job.store,
                    job.photo_taken_at, job.ocr_digest, job.warnings,
                ),
            ]

        path = brief_path(work_dir, sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(out), encoding="utf-8")
        written.append(path)
    return written


def answer_path_for(work_dir: Path, job_id: str) -> Path:
    """Where a job's answer belongs. Mirrors ``collect.answer_path``."""
    return work_dir / "answers" / f"{job_id}.json"


def write_manifest(path: Path, jobs: Iterable[Job]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for job in jobs:
            fh.write(job.to_json() + "\n")
            count += 1
    return count


def read_manifest(path: Path) -> list[Job]:
    jobs: list[Job] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                jobs.append(Job(**json.loads(line)))
    return jobs
