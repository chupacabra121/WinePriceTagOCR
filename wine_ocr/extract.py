"""Claude vision calls: layout pass, extraction pass, and Batch API submission."""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import anthropic
from PIL import Image
from pydantic import BaseModel, ValidationError

from .cache import ResponseCache
from .images import (
    MAX_EDGE_HIRES,
    MAX_EDGE_STANDARD,
    Region,
    band_regions,
    crop_region,
    encode_jpeg,
    grid_regions,
    image_block,
)
from .models import BandExtraction, PhotoLayout
from .prompts import (
    EXTRACT_SYSTEM,
    LAYOUT_SYSTEM,
    PROMPT_VERSION,
    extract_user_prompt,
    layout_user_prompt,
)
from .schema import output_config

DEFAULT_MODEL = "claude-opus-5"


class ExtractionError(RuntimeError):
    """A call that failed in a way the caller should record but survive."""


@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    effort: str = "high"
    max_tokens: int = 32000
    max_edge: int = MAX_EDGE_HIRES
    layout_max_edge: int = MAX_EDGE_STANDARD
    tiling: str = "auto"  # auto | grid | whole
    workers: int = 4
    jpeg_quality: int = 90


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    calls: int = 0

    def add(self, usage) -> None:
        self.calls += 1
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write += getattr(usage, "cache_creation_input_tokens", 0) or 0


@dataclass
class RegionResult:
    region: Region
    extraction: Optional[BandExtraction]
    error: Optional[str] = None
    from_cache: bool = False


@dataclass
class PhotoResult:
    path: Path
    layout: Optional[PhotoLayout]
    regions: list[RegionResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def build_client(max_retries: int = 5) -> anthropic.Anthropic:
    """Client with generous retries — 429/529 are routine on long runs."""
    return anthropic.Anthropic(max_retries=max_retries)


def _system_blocks(text: str) -> list[dict]:
    """System prompt with a cache breakpoint.

    Render order is tools -> system -> messages, so a breakpoint on the last
    system block caches the whole prefix. Every call in a run shares this exact
    prefix, and the per-photo image and context sit after it in the user turn.
    """
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _parse_into(model_cls: type[BaseModel], text: str) -> BaseModel:
    try:
        return model_cls.model_validate_json(text)
    except ValidationError as exc:
        raise ExtractionError(f"response did not match schema: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"response was not valid JSON: {exc}") from exc


def _text_of(message) -> str:
    parts = [b.text for b in message.content if getattr(b, "type", None) == "text"]
    if not parts:
        raise ExtractionError("response contained no text block")
    return "".join(parts)


def _call(
    client: anthropic.Anthropic,
    settings: Settings,
    system: str,
    user_text: str,
    b64: str,
    schema_model: type[BaseModel],
    usage: Usage,
) -> BaseModel:
    """One streamed vision call returning a validated model instance.

    Streaming is used throughout: with a large ``max_tokens`` a non-streaming
    request risks an HTTP timeout, and thinking tokens count against the same
    budget as the answer.
    """
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            with client.messages.stream(
                model=settings.model,
                max_tokens=settings.max_tokens,
                system=_system_blocks(system),
                output_config=output_config(schema_model, settings.effort),
                messages=[
                    {
                        "role": "user",
                        "content": [image_block(b64), {"type": "text", "text": user_text}],
                    }
                ],
            ) as stream:
                message = stream.get_final_message()

            usage.add(message.usage)

            if message.stop_reason == "refusal":
                detail = getattr(message, "stop_details", None)
                category = getattr(detail, "category", None) if detail else None
                raise ExtractionError(f"model declined this image (category={category})")
            if message.stop_reason == "max_tokens":
                raise ExtractionError(
                    "hit max_tokens before finishing — raise --max-tokens or "
                    "lower --effort for this photo"
                )
            return _parse_into(schema_model, _text_of(message))

        except (anthropic.APIConnectionError, anthropic.InternalServerError) as exc:
            last_error = exc
            time.sleep(min(2**attempt + random.random(), 12))
        except anthropic.RateLimitError as exc:
            last_error = exc
            time.sleep(min(4 * (attempt + 1) + random.random() * 2, 30))

    raise ExtractionError(f"call failed after retries: {last_error}")


# --------------------------------------------------------------------------
# Pass 1
# --------------------------------------------------------------------------


def get_layout(
    client: anthropic.Anthropic,
    settings: Settings,
    img: Image.Image,
    photo_path: Path,
    sha: str,
    store_hint: Optional[str],
    taken_at: Optional[str],
    cache: ResponseCache,
    usage: Usage,
) -> PhotoLayout:
    key = cache.key("layout", PROMPT_VERSION, settings.model, settings.layout_max_edge, sha)
    if cached := cache.get(key):
        return PhotoLayout.model_validate(cached)

    b64, _, _ = encode_jpeg(img, settings.layout_max_edge, settings.jpeg_quality)
    layout = _call(
        client,
        settings,
        LAYOUT_SYSTEM,
        layout_user_prompt(photo_path.name, store_hint, taken_at),
        b64,
        PhotoLayout,
        usage,
    )
    cache.put(key, layout.model_dump())
    return layout  # type: ignore[return-value]


def plan_regions(
    settings: Settings, width: int, height: int, layout: Optional[PhotoLayout]
) -> list[Region]:
    if settings.tiling == "whole":
        return [Region("whole photo", 0, 0, width, height, 0, 0, 1)]
    if settings.tiling == "grid" or layout is None or not layout.bands:
        return grid_regions(width, height, settings.max_edge)
    regions = band_regions(width, height, layout.bands, settings.max_edge)
    # Every band was marked non-wine, or the geometry was degenerate.
    return regions or []


# --------------------------------------------------------------------------
# Pass 2
# --------------------------------------------------------------------------


def extract_region(
    client: anthropic.Anthropic,
    settings: Settings,
    img: Image.Image,
    region: Region,
    photo_path: Path,
    sha: str,
    store_hint: Optional[str],
    currency_hint: Optional[str],
    taken_at: Optional[str],
    cache: ResponseCache,
    usage: Usage,
) -> RegionResult:
    key = cache.key(
        "extract", PROMPT_VERSION, settings.model, settings.effort, settings.max_edge,
        sha, region.x0, region.y0, region.x1, region.y1, region.contains_tag_rail,
    )
    if cached := cache.get(key):
        try:
            return RegionResult(region, BandExtraction.model_validate(cached), from_cache=True)
        except ValidationError:
            pass  # cached under an older shape; re-extract

    try:
        crop = crop_region(img, region)
        b64, _, _ = encode_jpeg(crop, settings.max_edge, settings.jpeg_quality)
        extraction = _call(
            client,
            settings,
            EXTRACT_SYSTEM,
            extract_user_prompt(
                photo_path.name, region.label, store_hint, currency_hint,
                taken_at, region.tile_count > 1, region.contains_tag_rail,
            ),
            b64,
            BandExtraction,
            usage,
        )
        cache.put(key, extraction.model_dump())
        return RegionResult(region, extraction)  # type: ignore[arg-type]
    except ExtractionError as exc:
        return RegionResult(region, None, error=str(exc))
    except Exception as exc:  # never let one bad crop kill the run
        return RegionResult(region, None, error=f"{type(exc).__name__}: {exc}")


def extract_photo(
    client: anthropic.Anthropic,
    settings: Settings,
    img: Image.Image,
    photo_path: Path,
    sha: str,
    store_hint: Optional[str],
    taken_at: Optional[str],
    cache: ResponseCache,
    usage: Usage,
    on_region: Optional[Callable[[RegionResult], None]] = None,
) -> PhotoResult:
    layout: Optional[PhotoLayout] = None
    errors: list[str] = []

    if settings.tiling == "auto":
        try:
            layout = get_layout(
                client, settings, img, photo_path, sha, store_hint, taken_at, cache, usage
            )
        except ExtractionError as exc:
            errors.append(f"layout pass failed, falling back to grid: {exc}")

    regions = plan_regions(settings, img.width, img.height, layout)
    if not regions:
        return PhotoResult(photo_path, layout, [], errors + ["no wine shelves found"])

    currency_hint = layout.currency_guess if layout else None

    def run(region: Region) -> RegionResult:
        result = extract_region(
            client, settings, img, region, photo_path, sha,
            store_hint, currency_hint, taken_at, cache, usage,
        )
        if on_region:
            on_region(result)
        return result

    # The first call warms the shared system-prompt cache; a cache entry only
    # becomes readable once its response starts streaming, so firing the whole
    # fan-out at once would have every worker pay the full prefix.
    results = [run(regions[0])]
    remaining = regions[1:]
    if remaining:
        if settings.workers > 1:
            with ThreadPoolExecutor(max_workers=settings.workers) as pool:
                results.extend(pool.map(run, remaining))
        else:
            results.extend(run(r) for r in remaining)

    return PhotoResult(photo_path, layout, results, errors)


# --------------------------------------------------------------------------
# Batch API
# --------------------------------------------------------------------------

BATCH_MAX_REQUESTS = 500
BATCH_MAX_BYTES = 180 * 1024 * 1024  # well under the 256 MB request cap


@dataclass
class BatchItem:
    custom_id: str
    photo_path: Path
    region: Region
    cache_key: str
    params: dict


def build_batch_item(
    settings: Settings,
    img: Image.Image,
    region: Region,
    photo_path: Path,
    sha: str,
    index: int,
    store_hint: Optional[str],
    currency_hint: Optional[str],
    taken_at: Optional[str],
    cache: ResponseCache,
) -> BatchItem:
    key = cache.key(
        "extract", PROMPT_VERSION, settings.model, settings.effort, settings.max_edge,
        sha, region.x0, region.y0, region.x1, region.y1, region.contains_tag_rail,
    )
    crop = crop_region(img, region)
    b64, _, _ = encode_jpeg(crop, settings.max_edge, settings.jpeg_quality)
    params = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": _system_blocks(EXTRACT_SYSTEM),
        "output_config": output_config(BandExtraction, settings.effort),
        "messages": [
            {
                "role": "user",
                "content": [
                    image_block(b64),
                    {
                        "type": "text",
                        "text": extract_user_prompt(
                            photo_path.name, region.label, store_hint,
                            currency_hint, taken_at, region.tile_count > 1,
                            region.contains_tag_rail,
                        ),
                    },
                ],
            }
        ],
    }
    return BatchItem(f"{sha[:24]}-{index:04d}", photo_path, region, key, params)


def chunk_batch(items: list[BatchItem]) -> Iterable[list[BatchItem]]:
    chunk: list[BatchItem] = []
    size = 0
    for item in items:
        approx = len(json.dumps(item.params))
        if chunk and (len(chunk) >= BATCH_MAX_REQUESTS or size + approx > BATCH_MAX_BYTES):
            yield chunk
            chunk, size = [], 0
        chunk.append(item)
        size += approx
    if chunk:
        yield chunk


def submit_and_wait(
    client: anthropic.Anthropic,
    items: list[BatchItem],
    cache: ResponseCache,
    usage: Usage,
    poll_seconds: int = 30,
    on_status: Optional[Callable[[str], None]] = None,
) -> dict[str, RegionResult]:
    """Run items through the Batch API (half price) and return results by custom_id."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    out: dict[str, RegionResult] = {}
    by_id = {item.custom_id: item for item in items}

    for chunk_index, chunk in enumerate(chunk_batch(items), start=1):
        batch = client.messages.batches.create(
            requests=[
                Request(
                    custom_id=item.custom_id,
                    params=MessageCreateParamsNonStreaming(**item.params),
                )
                for item in chunk
            ]
        )
        if on_status:
            on_status(f"batch {chunk_index}: submitted {len(chunk)} requests ({batch.id})")

        while True:
            current = client.messages.batches.retrieve(batch.id)
            if current.processing_status == "ended":
                break
            if on_status:
                counts = current.request_counts
                on_status(
                    f"batch {chunk_index}: {counts.succeeded} done, "
                    f"{counts.processing} running, {counts.errored} errored"
                )
            time.sleep(poll_seconds)

        for entry in client.messages.batches.results(batch.id):
            item = by_id.get(entry.custom_id)
            if item is None:
                continue
            if entry.result.type != "succeeded":
                out[entry.custom_id] = RegionResult(
                    item.region, None, error=f"batch result: {entry.result.type}"
                )
                continue
            message = entry.result.message
            usage.add(message.usage)
            if message.stop_reason == "refusal":
                out[entry.custom_id] = RegionResult(
                    item.region, None, error="model declined this image"
                )
                continue
            try:
                extraction = _parse_into(BandExtraction, _text_of(message))
                cache.put(item.cache_key, extraction.model_dump())
                out[entry.custom_id] = RegionResult(item.region, extraction)  # type: ignore[arg-type]
            except ExtractionError as exc:
                out[entry.custom_id] = RegionResult(item.region, None, error=str(exc))

    return out
