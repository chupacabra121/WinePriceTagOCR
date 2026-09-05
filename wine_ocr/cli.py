"""Command line interface."""

from __future__ import annotations

import collections
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)
from rich.table import Table

from . import extract as ex
from .cache import ResponseCache
from .images import (
    MAX_EDGE_HIRES, MAX_EDGE_STANDARD, crop_bbox, estimate_image_tokens,
    iter_photos, load_oriented, read_meta,
)
from .output import (
    COLUMNS, UNMATCHED_COLUMNS, deduplicate, rows_from_photo, write_csv,
    write_jsonl, write_xlsx,
)
from .stores import load_store_map, resolve_store
from .brandapply import enrich, load_reference, load_verdicts
from .collect import assemble, pending
from .layout import build_bands, digest, flipped_lines
from .mirror import write_mirror
from .prep import brief_path, prepare, read_manifest, write_briefs, write_manifest
from .vision import DEFAULT_MAX_TILE, read_page

app = typer.Typer(
    add_completion=False,
    help="Read wine names and prices off shop shelf photos and build a table.",
)
console = Console()

# Approximate list prices, USD per million tokens, for the estimate command.
PRICING = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-fable-5": (10.0, 50.0),
}


def _require_key() -> None:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("ANTHROPIC_AUTH_TOKEN"):
        console.print(
            "[red]No API key found.[/red] Set ANTHROPIC_API_KEY in your environment "
            "or in a .env file (see .env.example)."
        )
        raise typer.Exit(2)


@app.command()
def extract(
    path: Path = typer.Argument(..., help="Photo, or folder of photos."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
    store: Optional[str] = typer.Option(
        None, "--store", "-s", help="Force a store name for every photo in this run."
    ),
    stores_config: Path = typer.Option(
        Path("config/stores.yaml"), "--stores-config", help="Folder-to-store alias map."
    ),
    model: str = typer.Option(ex.DEFAULT_MODEL, "--model", "-m"),
    effort: str = typer.Option(
        "high", "--effort", help="low | medium | high | xhigh | max."
    ),
    tiling: str = typer.Option(
        "auto", "--tiling",
        help="auto = find shelves then read each at full resolution (accurate); "
             "grid = fixed overlapping tiles; whole = one call per photo (cheapest, "
             "loses small tag text).",
    ),
    max_edge: int = typer.Option(MAX_EDGE_HIRES, "--max-edge"),
    max_tokens: int = typer.Option(32000, "--max-tokens"),
    workers: int = typer.Option(4, "--workers", "-w"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only the first N photos."),
    batch: bool = typer.Option(
        False, "--batch", help="Use the Batch API: half price, minutes-to-hours latency."
    ),
    crops: bool = typer.Option(
        False, "--crops", help="Write review crops for rows that need checking."
    ),
    no_cache: bool = typer.Option(False, "--no-cache"),
    append: bool = typer.Option(
        False, "--append", help="Merge into an existing wines.csv in the output folder."
    ),
) -> None:
    """Extract wines and prices from photos into a table."""
    _require_key()

    photos = iter_photos(path)
    if limit:
        photos = photos[:limit]
    if not photos:
        console.print(f"[red]No images found at {path}[/red]")
        raise typer.Exit(1)

    root = path if path.is_dir() else path.parent
    settings = ex.Settings(
        model=model, effort=effort, max_tokens=max_tokens, max_edge=max_edge,
        tiling=tiling, workers=workers,
    )
    cache = ResponseCache(Path(".cache/responses"), enabled=not no_cache)
    store_map = load_store_map(stores_config)
    client = ex.build_client()
    usage = ex.Usage()

    console.print(
        f"[bold]{len(photos)}[/bold] photo(s) · model [cyan]{model}[/cyan] · "
        f"effort [cyan]{effort}[/cyan] · tiling [cyan]{tiling}[/cyan]"
        + (" · [yellow]batch mode[/yellow]" if batch else "")
    )

    all_rows: list[dict] = []
    all_unmatched: list[dict] = []
    all_errors: list[dict] = []
    raw_records: list[dict] = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Reading photos", total=len(photos))

        for photo in photos:
            progress.update(task, description=f"Reading {photo.name}")
            try:
                meta = read_meta(photo)
                img = load_oriented(photo)
            except Exception as exc:
                all_errors.append(
                    {"photo": str(photo), "stage": "load", "detail": f"{type(exc).__name__}: {exc}"}
                )
                progress.advance(task)
                continue

            hint = store or None
            result = ex.extract_photo(
                client, settings, img, photo, meta.sha256, hint, meta.taken_at,
                cache, usage,
            )

            resolution = resolve_store(
                photo, root, store, store_map,
                result.layout.store_name_visible if result.layout else None,
            )

            for message in result.errors:
                all_errors.append({"photo": str(photo), "stage": "layout", "detail": message})
            for region_result in result.regions:
                if region_result.error:
                    all_errors.append(
                        {
                            "photo": str(photo),
                            "stage": f"extract [{region_result.region.label}]",
                            "detail": region_result.error,
                        }
                    )

            rows, unmatched = rows_from_photo(result, meta, resolution, model, root)
            all_rows.extend(rows)
            all_unmatched.extend(unmatched)
            raw_records.append(
                {
                    "photo": str(photo),
                    "sha256": meta.sha256,
                    "store": resolution.store,
                    "layout": result.layout.model_dump() if result.layout else None,
                    "regions": [
                        {
                            "label": r.region.label,
                            "box": [r.region.x0, r.region.y0, r.region.x1, r.region.y1],
                            "from_cache": r.from_cache,
                            "error": r.error,
                            "extraction": r.extraction.model_dump() if r.extraction else None,
                        }
                        for r in result.regions
                    ],
                }
            )

            if crops:
                _write_crops(out / "crops", img, photo, rows)

            progress.advance(task)

    if append and (out / "wines.csv").exists():
        import csv as _csv

        with (out / "wines.csv").open("r", encoding="utf-8-sig") as fh:
            existing = [r for r in _csv.DictReader(fh) if not r.get("duplicate_of")]
        for row in existing:
            for key in ("price", "price_per_litre", "volume_ml", "original_price"):
                if row.get(key):
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
        all_rows = existing + all_rows
        console.print(f"Merged {len(existing)} existing row(s).")

    final_rows = deduplicate(all_rows)
    primary = [r for r in final_rows if not r.get("duplicate_of")]

    write_csv(out / "wines.csv", final_rows, COLUMNS)
    write_csv(out / "unmatched_tags.csv", all_unmatched, UNMATCHED_COLUMNS)
    if all_errors:
        write_csv(out / "errors.csv", all_errors, ["photo", "stage", "detail"])
    write_jsonl(out / "extractions.jsonl", raw_records)
    write_xlsx(out / "wines.xlsx", final_rows, all_unmatched, all_errors)

    _summary(primary, final_rows, all_unmatched, all_errors, usage, cache, model, out)


def _write_crops(crop_dir: Path, img, photo: Path, rows: list[dict]) -> None:
    crop_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        if not row.get("needs_review"):
            continue
        box = row.get("bottle_bbox") or row.get("tag_bbox")
        if not box:
            continue
        try:
            coords = tuple(float(v) for v in box.split(","))
            crop_bbox(img, coords).save(
                crop_dir / f"{photo.stem}_{i:03d}.jpg", quality=88
            )
        except Exception:
            continue


def _summary(primary, final_rows, unmatched, errors, usage, cache, model, out) -> None:
    review = [r for r in primary if r.get("needs_review") == "yes"]
    priced = [r for r in primary if r.get("price") is not None]

    table = Table(title="Extraction summary", show_header=False, title_style="bold")
    table.add_row("Wines found", str(len(primary)))
    table.add_row("With a price", f"{len(priced)} ({_pct(len(priced), len(primary))})")
    table.add_row("Needs review", f"{len(review)} ({_pct(len(review), len(primary))})")
    table.add_row("Duplicate sightings", str(len(final_rows) - len(primary)))
    table.add_row("Unresolved tags", str(len(unmatched)))
    table.add_row("Errors", str(len(errors)))
    table.add_row("API calls", str(usage.calls))
    table.add_row("Cache hits", f"{cache.hits} hit / {cache.misses} miss")

    if usage.calls:
        rate_in, rate_out = PRICING.get(model, (0.0, 0.0))
        billed_in = usage.input_tokens + usage.cache_write * 0.25 + usage.cache_read * 0.1
        cost = (billed_in * rate_in + usage.output_tokens * rate_out) / 1_000_000
        table.add_row(
            "Tokens",
            f"{usage.input_tokens:,} in ({usage.cache_read:,} cached) / "
            f"{usage.output_tokens:,} out",
        )
        if rate_in:
            table.add_row("Approx. cost", f"${cost:,.2f}")

    console.print(table)
    console.print(f"\n[green]Wrote[/green] {out / 'wines.xlsx'} and {out / 'wines.csv'}")
    if review:
        console.print(
            f"[yellow]{len(review)} row(s) flagged[/yellow] — see the "
            f"'Needs review' sheet."
        )


def _pct(part: int, whole: int) -> str:
    return f"{(100.0 * part / whole):.0f}%" if whole else "0%"


@app.command()
def estimate(
    path: Path = typer.Argument(..., help="Photo, or folder of photos."),
    model: str = typer.Option(ex.DEFAULT_MODEL, "--model", "-m"),
    tiling: str = typer.Option("auto", "--tiling"),
    max_edge: int = typer.Option(MAX_EDGE_HIRES, "--max-edge"),
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    """Estimate tokens and cost for a run without calling the API."""
    photos = iter_photos(path)
    if limit:
        photos = photos[:limit]
    if not photos:
        console.print(f"[red]No images found at {path}[/red]")
        raise typer.Exit(1)

    from .images import grid_regions

    total_in = 0
    total_regions = 0
    for photo in photos:
        try:
            meta = read_meta(photo)
        except Exception:
            continue
        if tiling == "whole":
            regions = [None]
            total_in += estimate_image_tokens(meta.width, meta.height, max_edge)
        else:
            # The layout pass sees the whole photo at the standard tier.
            if tiling == "auto":
                total_in += estimate_image_tokens(
                    meta.width, meta.height, MAX_EDGE_STANDARD
                )
            regions = grid_regions(meta.width, meta.height, max_edge)
            for region in regions:
                total_in += estimate_image_tokens(region.width, region.height, max_edge)
        total_regions += len(regions)
        total_in += 1400 * (len(regions) + (1 if tiling == "auto" else 0))  # prompts

    est_out = 900 * total_regions
    rate_in, rate_out = PRICING.get(model, (0.0, 0.0))
    cost = (total_in * rate_in + est_out * rate_out) / 1_000_000

    table = Table(title="Cost estimate", show_header=False, title_style="bold")
    table.add_row("Photos", str(len(photos)))
    table.add_row("Model calls", str(total_regions + (len(photos) if tiling == "auto" else 0)))
    table.add_row("Input tokens (approx.)", f"{total_in:,}")
    table.add_row("Output tokens (approx.)", f"{est_out:,}")
    if rate_in:
        table.add_row("Approx. cost", f"${cost:,.2f}")
        table.add_row("With --batch (50% off)", f"${cost / 2:,.2f}")
    console.print(table)
    console.print(
        "\n[dim]Upper bound: it assumes a full grid and ignores prompt-cache and "
        "response-cache savings. 'auto' tiling usually reads fewer regions because "
        "it skips shelves with no wine.[/dim]"
    )


@app.command()
def plan(
    path: Path = typer.Argument(..., help="A single photo."),
    max_edge: int = typer.Option(MAX_EDGE_HIRES, "--max-edge"),
    out: Optional[Path] = typer.Option(None, "--out", help="Write the tile crops here."),
) -> None:
    """Show how a photo would be tiled, without calling the API.

    Useful for sanity-checking geometry and resolution before spending anything.
    """
    from .images import grid_regions

    meta = read_meta(path)
    regions = grid_regions(meta.width, meta.height, max_edge)

    table = Table(title=f"{path.name} — {meta.width}x{meta.height}")
    table.add_column("Tile")
    table.add_column("Box (px)")
    table.add_column("Size")
    table.add_column("Downscale")
    table.add_column("~tokens", justify="right")
    for region in regions:
        scale = max(region.width, region.height) / max_edge
        table.add_row(
            region.label,
            f"{region.x0},{region.y0} → {region.x1},{region.y1}",
            f"{region.width}x{region.height}",
            f"{scale:.2f}x" if scale > 1 else "none",
            f"{estimate_image_tokens(region.width, region.height, max_edge):,}",
        )
    console.print(table)
    console.print(
        f"Whole photo at {max_edge}px would be a "
        f"{max(meta.width, meta.height) / max_edge:.2f}x downscale "
        f"(~{estimate_image_tokens(meta.width, meta.height, max_edge):,} tokens)."
    )

    if out:
        img = load_oriented(path)
        out.mkdir(parents=True, exist_ok=True)
        from .images import crop_region

        for i, region in enumerate(regions):
            crop_region(img, region).save(out / f"{path.stem}_tile{i:02d}.jpg", quality=88)
        console.print(f"[green]Wrote {len(regions)} tile(s) to {out}[/green]")


@app.command()
def init(
    stores: Optional[str] = typer.Argument(
        None, help='Comma-separated store names, e.g. "Annabella,Kaufland,Lidl".'
    ),
    photos: Path = typer.Option(Path("data/photos"), "--photos", "-p"),
    force: bool = typer.Option(False, "--force", help="Overwrite config/stores.yaml."),
) -> None:
    """Create the local folder structure and config files. No API calls.

    Photos stay on your machine — they are gitignored deliberately. Run this
    once after cloning, then drop each shop's photos into its own folder.
    """
    created: list[str] = []
    existed: list[str] = []

    names = [s.strip() for s in (stores or "").split(",") if s.strip()]
    photos.mkdir(parents=True, exist_ok=True)
    (photos / ".gitkeep").touch()

    for name in names:
        # Folder names feed store attribution, so keep them tidy and lowercase.
        slug = re.sub(r"[^\w\-]+", "-", name.strip().lower()).strip("-")
        folder = photos / slug
        (existed if folder.exists() else created).append(str(folder))
        folder.mkdir(parents=True, exist_ok=True)

    for target, template in (
        (Path("config/stores.yaml"), Path("config/stores.example.yaml")),
        (Path(".env"), Path(".env.example")),
    ):
        if target.exists() and not force:
            existed.append(str(target))
            continue
        if template.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
            created.append(str(target))

    Path("out").mkdir(exist_ok=True)

    if created:
        console.print("[green]Created[/green]\n  " + "\n  ".join(created))
    if existed:
        console.print("[dim]Already there\n  " + "\n  ".join(existed) + "[/dim]")

    console.print(
        "\n[bold]Next[/bold]\n"
        "  1. Put your API key in [cyan].env[/cyan]\n"
        f"  2. Copy each shop's photos into [cyan]{photos}/<store>/[/cyan]\n"
        "  3. [cyan]wine-ocr estimate data/photos[/cyan]   — see the cost first\n"
        "  4. [cyan]wine-ocr extract data/photos[/cyan]\n\n"
        "[dim]Photos and .env are gitignored: they stay on this machine.[/dim]"
    )


@app.command()
def review(
    extracted: Path = typer.Argument(
        Path("out/wines.csv"), help="A wines.csv produced by `extract`."
    ),
    photos: Path = typer.Option(
        Path("data/photos"), "--photos", "-p",
        help="Folder the `photo` column is relative to.",
    ),
    out: Path = typer.Option(Path("out/review.html"), "--out", "-o"),
    all_rows: bool = typer.Option(
        False, "--all", help="Include every row, not just flagged ones."
    ),
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    """Build a self-contained HTML sheet pairing each row with its crop.

    Makes no API calls. Open the file in a browser to check rows against what
    the model actually saw.
    """
    import csv as _csv

    from .review import build_report

    if not extracted.exists():
        console.print(f"[red]Not found:[/red] {extracted}")
        raise typer.Exit(1)

    with extracted.open("r", encoding="utf-8-sig") as fh:
        rows = [r for r in _csv.DictReader(fh) if not (r.get("duplicate_of") or "").strip()]

    if not all_rows:
        rows = [r for r in rows if r.get("needs_review") == "yes"]
        if not rows:
            console.print(
                "[green]Nothing flagged for review.[/green] "
                "Use --all to inspect every row anyway."
            )
            raise typer.Exit(0)
    if limit:
        rows = rows[:limit]

    document, rendered, crops = build_report(rows, photos)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")

    size_kb = out.stat().st_size / 1024
    console.print(
        f"[green]Wrote[/green] {out} — {rendered} row(s), {crops} crop(s), "
        f"{size_kb:,.0f} KB"
    )
    if rendered and not crops:
        console.print(
            "[yellow]No crops embedded.[/yellow] Either the photos are not under "
            f"{photos} (pass --photos), or the model returned no bounding boxes."
        )


@app.command()
def verify(
    extracted: Path = typer.Argument(
        Path("out/wines.csv"), help="A wines.csv produced by `extract`."
    ),
    truth: Path = typer.Option(
        Path("data/samples/annabella/ground_truth.csv"), "--truth", "-t",
        help="Hand-read ground-truth CSV.",
    ),
    show: int = typer.Option(30, "--show", help="Rows of detail to print."),
) -> None:
    """Score an extraction against hand-read ground truth. Makes no API calls.

    Ground truth covers a subset of the photos, so this reports recall over
    that subset; rows outside it are listed but never counted as errors.
    """
    from .verify import load_extracted, load_truth, verify as run_verify

    for path in (extracted, truth):
        if not path.exists():
            console.print(f"[red]Not found:[/red] {path}")
            raise typer.Exit(1)

    expected = load_truth(truth)
    rows = load_extracted(extracted)
    if not expected:
        console.print(f"[red]No usable rows in {truth}[/red]")
        raise typer.Exit(1)

    report = run_verify(expected, rows)

    detail = Table(title=f"{truth.name} vs {extracted.name}")
    detail.add_column("#", justify="right")
    detail.add_column("Expected price", justify="right")
    detail.add_column("Expected name")
    detail.add_column("Found")
    detail.add_column("Extracted name")
    for outcome in report.outcomes[:show]:
        exp = outcome.expected
        if not outcome.found:
            found, name = "[red]missing[/red]", "—"
        else:
            marks = {True: "[green]yes[/green]", False: "[red]no[/red]", None: "[dim]n/a[/dim]"}
            found = f"yes · name {marks[outcome.name_ok]}"
            name = (outcome.matched.get("wine_name") or "")[:44]
        detail.add_row(
            f"{exp.rail}.{exp.position}", f"{exp.price:.2f}",
            exp.name_contains or "[dim]—[/dim]", found, name,
        )
    console.print(detail)

    total = len(report.outcomes)
    summary = Table(title="Score", show_header=False, title_style="bold")
    summary.add_row("Prices found", f"{report.found}/{total} ({_pct(report.found, total)})")
    if report.name_checked:
        summary.add_row(
            "Names correct",
            f"{report.name_correct}/{report.name_checked} "
            f"({_pct(report.name_correct, report.name_checked)}) of tags found",
        )
    volumes = sum(1 for o in report.outcomes if o.expected.volume_ml is not None)
    if volumes:
        summary.add_row(
            "Volumes correct",
            f"{report.volume_correct}/{volumes} ({_pct(report.volume_correct, volumes)})",
        )
    summary.add_row("Rows outside ground truth", f"{len(report.extra)} (not scored)")
    summary.add_row("Total rows in table", str(report.total_rows))
    console.print(summary)

    if report.found < total:
        console.print(
            "\n[yellow]Missing prices usually mean the tag was never read.[/yellow] "
            "Check --tiling (whole loses small tag text), then the 'Errors' sheet."
        )



# --------------------------------------------------------------------------
# Local-first pipeline: prep -> read (by agents, elsewhere) -> collect
# --------------------------------------------------------------------------


@app.command()
def prep(
    path: Path = typer.Argument(..., help="Photo, or folder of photos."),
    work: Path = typer.Option(
        Path("out/work"), "--work", help="Where crops, the manifest and answers live."
    ),
    store: Optional[str] = typer.Option(
        None, "--store", "-s", help="Force a store name for every photo in this run."
    ),
    stores_config: Path = typer.Option(
        Path("config/stores.yaml"), "--stores-config", help="Folder-to-store alias map."
    ),
    max_tile: int = typer.Option(
        DEFAULT_MAX_TILE, "--max-tile",
        help="Tile size handed to local OCR. Smaller sees smaller print, slower.",
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only the first N photos."),
    no_cache: bool = typer.Option(False, "--no-cache"),
) -> None:
    """Read every photo locally and write one job per shelf. No API calls."""
    photos = iter_photos(path)
    if limit:
        photos = photos[:limit]
    if not photos:
        console.print(f"[red]No images found at {path}[/red]")
        raise typer.Exit(1)

    root = path if path.is_dir() else path.parent
    cache = ResponseCache(Path(".cache/vision"), enabled=not no_cache)

    console.print(
        f"[bold]{len(photos)}[/bold] photo(s) · local OCR at "
        f"[cyan]{max_tile}px[/cyan] tiles · no API calls"
    )
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Reading photos", total=len(photos))
        result = prepare(
            photos, root, work, cache,
            store_override=store,
            store_config=stores_config,
            max_tile=max_tile,
            on_photo=lambda p, n: progress.advance(task),
        )

    manifest = work / "manifest.jsonl"
    write_manifest(manifest, result.jobs)
    briefs = write_briefs(work, result.jobs)
    if result.errors:
        write_csv(work / "prep_errors.csv", result.errors, ["photo", "stage", "detail"])

    table = Table(show_header=False, box=None)
    table.add_row("Photos read", str(result.photos))
    table.add_row("Shelves found", str(result.bands))
    table.add_row("Prices found", str(sum(len(j.ocr_prices) for j in result.jobs)))
    table.add_row("Jobs written", f"{len(result.jobs)} -> {manifest}")
    table.add_row("Crops written", str(work / "crops"))
    table.add_row("Briefs written", f"{len(briefs)} -> {work / 'briefs'}")
    if result.errors:
        table.add_row("[yellow]Errors[/yellow]", f"{len(result.errors)} -> prep_errors.csv")
    console.print(table)
    console.print(
        f"\nNext: have an agent answer each job, writing "
        f"[cyan]{work}/answers/<job_id>.json[/cyan], then run "
        f"[cyan]wine-ocr collect --work {work}[/cyan]."
    )


@app.command()
def status(
    work: Path = typer.Option(Path("out/work"), "--work"),
) -> None:
    """How much of a prepared run has been answered."""
    manifest = work / "manifest.jsonl"
    if not manifest.exists():
        console.print(f"[red]No manifest at {manifest}[/red] — run `wine-ocr prep` first.")
        raise typer.Exit(1)
    jobs = read_manifest(manifest)
    outstanding = pending(jobs, work)
    done = len(jobs) - len(outstanding)

    table = Table(show_header=False, box=None)
    table.add_row("Jobs", str(len(jobs)))
    table.add_row("Answered", f"{done} ({_pct(done, len(jobs))})")
    table.add_row("Pending", str(len(outstanding)))
    console.print(table)

    by_store: dict[str, list] = {}
    for job in outstanding:
        by_store.setdefault(job.store, []).append(job)
    if by_store:
        remaining = Table(title="Still to read", title_style="bold")
        remaining.add_column("Store")
        remaining.add_column("Jobs", justify="right")
        for name in sorted(by_store):
            remaining.add_row(name, str(len(by_store[name])))
        console.print(remaining)


@app.command()
def collect(
    work: Path = typer.Option(Path("out/work"), "--work"),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
    root: Optional[Path] = typer.Option(
        None, "--root", help="Photo root, for the mirrored folder layout."
    ),
    model: str = typer.Option("claude-code-agent", "--model"),
    excel: bool = typer.Option(True, "--excel/--no-excel"),
) -> None:
    """Assemble answered jobs into CSVs mirroring the photo folders. No API calls."""
    manifest = work / "manifest.jsonl"
    if not manifest.exists():
        console.print(f"[red]No manifest at {manifest}[/red] — run `wine-ocr prep` first.")
        raise typer.Exit(1)

    jobs = read_manifest(manifest)
    outstanding = pending(jobs, work)
    if outstanding:
        console.print(
            f"[yellow]{len(outstanding)} of {len(jobs)} jobs have no answer yet[/yellow]"
            " — collecting what there is."
        )

    photo_root = root or Path(".")
    rows, unmatched, errors = assemble(jobs, work, photo_root, model)
    final_rows = deduplicate(rows)
    primary = [r for r in final_rows if not r.get("duplicate_of")]

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "wines.csv", final_rows, COLUMNS)
    write_csv(out / "unmatched_tags.csv", unmatched, UNMATCHED_COLUMNS)
    if errors:
        write_csv(out / "errors.csv", errors, ["photo", "stage", "detail"])
    index = write_mirror(out, final_rows, unmatched)
    if excel:
        write_xlsx(out / "wines.xlsx", final_rows, unmatched, errors)

    table = Table(title="Per folder", title_style="bold")
    for column in ("Folder", "Wines", "Priced", "Review", "Median"):
        table.add_column(column, justify="right" if column != "Folder" else "left")
    for entry in index:
        table.add_row(
            entry["folder"], str(entry["wines"]), str(entry["priced"]),
            str(entry["needs_review"]),
            f"{entry['median_price']:.2f}" if entry["median_price"] else "-",
        )
    console.print(table)

    summary = Table(show_header=False, box=None)
    summary.add_row("Wines", str(len(primary)))
    summary.add_row("Duplicates linked", str(len(final_rows) - len(primary)))
    summary.add_row("Needs review", str(sum(1 for r in primary if r.get("needs_review") == "yes")))
    summary.add_row("Unmatched tags", str(len(unmatched)))
    summary.add_row("Written to", str(out))
    console.print(summary)


@app.command()
def ocr(
    photo: Path = typer.Argument(..., help="One photo to read locally."),
    max_tile: int = typer.Option(DEFAULT_MAX_TILE, "--max-tile"),
    show_lines: bool = typer.Option(False, "--lines", help="Print every OCR line."),
) -> None:
    """Show what local OCR reads on one photo, and the shelves derived from it."""
    cache = ResponseCache(Path(".cache/vision"))
    meta = read_meta(photo)
    page = read_page(photo, meta.sha256, cache, max_tile=max_tile)
    if page.error:
        console.print(f"[red]{page.error}[/red]")
        raise typer.Exit(1)

    console.print(
        f"[bold]{photo.name}[/bold] {page.width}x{page.height} · "
        f"{len(page.lines)} text lines"
    )
    flipped = flipped_lines(page)
    if flipped:
        console.print(
            f"[yellow]{len(flipped)} line(s) read upside down[/yellow] — "
            "prices on this photo need a second opinion"
        )
    if show_lines:
        for line in page.lines:
            console.print(
                f"  [dim]y={line.y0:.3f} x={line.x0:.3f} h={line.height_px:>3} "
                f"c={line.conf:.1f}[/dim] {line.text!r}"
            )

    for band in build_bands(page):
        console.print(f"\n[bold cyan]{band.label}[/bold cyan] "
                      f"[dim]crop ({band.x0:.3f},{band.y0:.3f})-({band.x1:.3f},{band.y1:.3f})[/dim]")
        console.print(digest(band))


@app.command()
def briefs(
    work: Path = typer.Option(Path("out/work"), "--work"),
    store: Optional[str] = typer.Option(None, "--store", help="Only this store."),
    limit: Optional[int] = typer.Option(None, "--limit", help="At most N photos."),
    all_photos: bool = typer.Option(
        False, "--all", help="Include photos that are already fully answered."
    ),
    compact: bool = typer.Option(
        False, "--compact",
        help="Emit the shortest form the reading pass accepts. At several "
             "hundred photos the argument list is itself worth shrinking.",
    ),
) -> None:
    """Print the reading pass's work list as JSON, ready to hand to an agent run.

    One entry per photo, smallest first, so a run that is cut short has still
    finished whole photos rather than leaving many half-read.
    """
    manifest = work / "manifest.jsonl"
    if not manifest.exists():
        console.print(f"[red]No manifest at {manifest}[/red] — run `wine-ocr prep` first.")
        raise typer.Exit(1)

    jobs = read_manifest(manifest)
    if store:
        needle = store.lower()
        jobs = [j for j in jobs if needle in j.store.lower()]
    if not all_photos:
        outstanding = {j.job_id for j in pending(jobs, work)}
        jobs = [j for j in jobs if j.job_id in outstanding]

    by_photo: dict[str, list] = {}
    for job in jobs:
        by_photo.setdefault(job.photo_sha256, []).append(job)

    entries = []
    for sha, group in by_photo.items():
        group.sort(key=lambda j: j.band_index)
        entries.append({
            "sha": sha[:16],
            "photo": group[0].photo,
            "store": group[0].store,
            "bands": len(group),
        })
    entries.sort(key=lambda e: (e["bands"], e["photo"]))
    if limit:
        entries = entries[:limit]

    # Shaped for the reading workflow's `args`: the work directory once, then
    # one entry per photo.
    if compact:
        payload = {
            "label": "resume",
            "work": str(work.resolve()),
            "shas": [f"{e['sha']}|{Path(e['photo']).name[:24]}" for e in entries],
        }
    else:
        payload = {"work": str(work.resolve()), "briefs": entries}
    print(json.dumps(payload, ensure_ascii=False))



@app.command()
def read(
    work: Path = typer.Option(Path("out/work"), "--work"),
    model: str = typer.Option("claude-sonnet-5", "--model", "-m"),
    effort: str = typer.Option("high", "--effort", help="low | medium | high | xhigh | max."),
    max_tokens: int = typer.Option(16000, "--max-tokens"),
    workers: int = typer.Option(4, "--workers", "-w"),
    cache_ttl: str = typer.Option(
        "5m", "--cache-ttl",
        help="Prompt-cache lifetime: 5m (default, cheapest for a bulk run) or 1h.",
    ),
    store: Optional[str] = typer.Option(None, "--store", help="Only this store."),
    limit: Optional[int] = typer.Option(None, "--limit", help="At most N bands."),
    all_jobs: bool = typer.Option(False, "--all", help="Re-read bands already answered."),
    no_cache: bool = typer.Option(False, "--no-cache"),
) -> None:
    """Read prepared bands through the Anthropic API. Needs a key.

    The same crops and digests the agent pass uses, sent straight to the model.
    Answers land in the same place, so the two paths are interchangeable and can
    even be mixed across one run.
    """
    _require_key()

    manifest = work / "manifest.jsonl"
    if not manifest.exists():
        console.print(f"[red]No manifest at {manifest}[/red] — run `wine-ocr prep` first.")
        raise typer.Exit(1)

    jobs = read_manifest(manifest)
    if store:
        needle = store.lower()
        jobs = [j for j in jobs if needle in j.store.lower()]
    if not all_jobs:
        jobs = pending(jobs, work)
    if limit:
        jobs = jobs[:limit]
    if not jobs:
        console.print("[green]Nothing to read — every band already has an answer.[/green]")
        return

    settings = ex.Settings(
        model=model, effort=effort, max_tokens=max_tokens, workers=workers,
        cache_ttl=cache_ttl,
    )
    cache = ResponseCache(Path(".cache/responses"), enabled=not no_cache)
    client = ex.build_client()
    usage = ex.Usage()

    console.print(
        f"[bold]{len(jobs)}[/bold] band(s) · model [cyan]{model}[/cyan] · "
        f"effort [cyan]{effort}[/cyan] · {workers} workers"
    )
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Reading shelves", total=len(jobs))
        written, errors = ex.read_jobs(
            client, settings, jobs, work, cache, usage,
            on_done=lambda j, e: progress.advance(task),
        )

    table = Table(show_header=False, box=None)
    table.add_row("Bands read", str(written))
    if errors:
        table.add_row("[yellow]Failed[/yellow]", str(len(errors)))
    table.add_row("API calls", str(usage.calls))
    table.add_row("Input tokens", f"{usage.input_tokens:,}")
    table.add_row("Output tokens", f"{usage.output_tokens:,}")
    table.add_row("Cache read", f"{usage.cache_read:,}")
    table.add_row("Cache write", f"{usage.cache_write:,}")
    price = PRICING.get(model)
    if price:
        billed_in = usage.input_tokens + usage.cache_write * 1.25 + usage.cache_read * 0.1
        cost = billed_in / 1e6 * price[0] + usage.output_tokens / 1e6 * price[1]
        table.add_row("Approx cost", f"${cost:,.2f}")
    console.print(table)
    for err in errors[:5]:
        console.print(f"  [yellow]{err['photo']}[/yellow]: {err['detail'][:110]}")


@app.command()
def brands(
    sheet: Path = typer.Argument(..., help="Standardized brand list (tab-separated)."),
    table: Path = typer.Option(Path("out/table"), "--table", help="Table to enrich in place."),
    work: Path = typer.Option(Path("out/brandmatch"), "--work", help="Batches and reviewed answers."),
    root: Optional[Path] = typer.Option(None, "--root", help="Photo root, for the mirrored layout."),
    excel: bool = typer.Option(True, "--excel/--no-excel"),
) -> None:
    """Add standardized Group / Winery / Label columns to the wine table.

    Reviewed verdicts win where they exist; anything the reviewers never saw
    falls back to the index's own pick and is labelled `auto` so the two are
    never confused in the sheet. Makes no API calls.
    """
    csv_path = table / "wines.csv"
    if not csv_path.exists():
        console.print(f"[red]No table at {csv_path}[/red] — run `wine-ocr collect` first.")
        raise typer.Exit(1)

    index = load_reference(sheet)
    # Later passes overwrite earlier ones: a second review exists precisely
    # because the first was working from a faulty shortlist.
    verdicts = load_verdicts(work / "answers", work / "batches")
    for extra in sorted(work.glob("pass*")):
        if (extra / "answers").is_dir():
            verdicts.update(load_verdicts(extra / "answers", extra / "batches"))
    with csv_path.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    stats = enrich(rows, index, verdicts)

    write_csv(csv_path, rows, COLUMNS)
    unmatched_path = table / "unmatched_tags.csv"
    stray: list[dict] = []
    if unmatched_path.exists():
        with unmatched_path.open(encoding="utf-8-sig") as fh:
            stray = list(csv.DictReader(fh))
    write_mirror(table, rows, stray)
    if excel:
        errs: list[dict] = []
        errors_path = table / "errors.csv"
        if errors_path.exists():
            with errors_path.open(encoding="utf-8-sig") as fh:
                errs = list(csv.DictReader(fh))
        write_xlsx(table / "wines.xlsx", rows, stray, errs)

    named = sum(1 for r in rows if r.get("brand_label"))
    summary = Table(show_header=False, box=None)
    summary.add_row("Rows", str(stats["rows"]))
    summary.add_row("With a brand", f"{named} ({_pct(named, stats['rows'])})")
    summary.add_row("  reviewed", str(stats["reviewed"]))
    summary.add_row("  auto (unreviewed)", str(stats["auto"]))
    summary.add_row("No brand in the sheet", str(stats["none"]))
    summary.add_row("Written to", str(table))
    console.print(summary)

    top = collections.Counter(
        r["brand_group"] for r in rows if r.get("brand_group")
    ).most_common(10)
    if top:
        t = Table(title="Biggest groups", title_style="bold")
        t.add_column("Group"); t.add_column("Rows", justify="right")
        for name, n in top:
            t.add_row(name, str(n))
        console.print(t)


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
