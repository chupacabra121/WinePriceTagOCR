"""A self-contained HTML review sheet: the crop beside what was read from it.

The spreadsheet is the deliverable, but it cannot show you the bottle. When a
row looks wrong the question is always "what did the model actually see", and
answering it by hunting for the photo and zooming to the right shelf is slow
enough that people skip it. This puts the crop next to the values, so a rail of
flagged rows can be checked in one pass.

Crops are embedded as base64, so the file works offline and can be mailed to
whoever is doing the checking.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

from .images import crop_bbox, load_oriented

THUMB_WIDTH = 300
# Bottle boxes are tall and narrow; without a height cap a single row can be a
# thousand pixels high and the sheet stops being scannable.
THUMB_HEIGHT = 230

_CSS = """
:root { --bg:#faf8f6; --card:#fff; --ink:#221c1c; --muted:#6d6460;
        --line:#e6dfda; --flag:#8c2f39; --ok:#2d6a4f; }
* { box-sizing: border-box; }
body { margin:0; padding:24px; background:var(--bg); color:var(--ink);
       font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
h1 { font-size:22px; margin:0 0 4px; }
.sub { color:var(--muted); margin-bottom:20px; }
.stats { display:flex; gap:20px; flex-wrap:wrap; margin-bottom:24px; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:8px;
        padding:10px 16px; min-width:110px; }
.stat b { display:block; font-size:20px; }
.stat span { color:var(--muted); font-size:12px; text-transform:uppercase;
             letter-spacing:.04em; }
h2 { font-size:15px; margin:28px 0 10px; padding-bottom:6px;
     border-bottom:1px solid var(--line); color:var(--muted); font-weight:600; }
.row { display:flex; gap:16px; background:var(--card); border:1px solid var(--line);
       border-left:4px solid var(--line); border-radius:8px; padding:12px;
       margin-bottom:10px; align-items:flex-start; }
.row.flagged { border-left-color:var(--flag); }
.row img { width:auto; max-width:300px; max-height:230px; border-radius:4px;
           background:#eee; flex:none; }
.row .nocrop { width:300px; height:90px; border-radius:4px; background:#f1ece8;
               color:var(--muted); font-size:12px; display:flex; flex:none;
               align-items:center; justify-content:center; }
.meta { flex:1; min-width:0; }
.name { font-weight:600; font-size:16px; margin-bottom:2px; }
.price { font-size:20px; font-weight:700; }
.price small { font-size:12px; color:var(--muted); font-weight:400; }
.fields { display:flex; flex-wrap:wrap; gap:4px 14px; margin:8px 0; font-size:13px;
          color:var(--muted); }
.reasons { display:inline-block; background:#fdf2f4; color:var(--flag);
           border-radius:4px; padding:3px 8px; font-size:12px; margin-top:4px; }
.raw { font:12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
       background:#f7f3f0; border-radius:4px; padding:7px 9px; margin-top:8px;
       color:#584f4b; white-space:pre-wrap; word-break:break-word; }
.ok { color:var(--ok); }
"""


def _thumb(img: Image.Image, box: str) -> Optional[str]:
    try:
        coords = tuple(float(v) for v in box.split(","))
        if len(coords) != 4:
            return None
        crop = crop_bbox(img, coords)  # type: ignore[arg-type]
        if crop.width < 8 or crop.height < 8:
            return None
        scale = min(THUMB_WIDTH / crop.width, THUMB_HEIGHT / crop.height, 1.0)
        if scale < 1.0:
            crop = crop.resize(
                (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                Image.LANCZOS,
            )
        buf = io.BytesIO()
        crop.convert("RGB").save(buf, format="JPEG", quality=78, optimize=True)
        return base64.standard_b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _esc(value) -> str:
    return html.escape(str(value)) if value not in (None, "") else ""


def _row_html(row: dict, thumb: Optional[str]) -> str:
    flagged = row.get("needs_review") == "yes"
    price = row.get("price")
    price_html = (
        f'{_esc(price)} <small>{_esc(row.get("currency"))}</small>'
        if price not in (None, "")
        else '<span style="color:#8c2f39">no price</span>'
    )

    fields = []
    for label, key in (
        ("vintage", "vintage"), ("volume", "volume_ml"), ("per litre", "price_per_litre"),
        ("type", "wine_type"), ("producer", "producer"), ("confidence", "pairing_confidence"),
    ):
        if value := row.get(key):
            fields.append(f"{label} <b>{_esc(value)}</b>")

    checks = []
    for key in ("price_check", "unit_price_check"):
        value = row.get(key) or ""
        if value == "ok":
            checks.append(f'<span class="ok">{key.replace("_", " ")} ok</span>')
        elif value.startswith("mismatch"):
            checks.append(f'<b style="color:#8c2f39">{_esc(value)}</b>')

    img_html = (
        f'<img src="data:image/jpeg;base64,{thumb}" alt="crop">'
        if thumb
        else '<div class="nocrop">no crop available</div>'
    )

    parts = [
        f'<div class="row{" flagged" if flagged else ""}">',
        img_html,
        '<div class="meta">',
        f'<div class="name">{_esc(row.get("wine_name")) or "(no name)"}</div>',
        f'<div class="price">{price_html}</div>',
        f'<div class="fields">{" · ".join(fields + checks)}</div>',
    ]
    if reasons := row.get("review_reasons"):
        parts.append(f'<div class="reasons">{_esc(reasons)}</div>')
    if raw := row.get("raw_tag_text"):
        parts.append(f'<div class="raw">{_esc(raw)}</div>')
    parts.append(f'<div class="fields">{_esc(row.get("shelf"))} · row {_esc(row.get("row_id"))}</div>')
    parts.append("</div></div>")
    return "".join(parts)


def build_report(
    rows: Iterable[dict],
    photo_root: Path,
    title: str = "Wine extraction review",
) -> tuple[str, int, int]:
    """Return (html, rows rendered, crops embedded)."""
    rows = list(rows)  # iterated twice: once to group, once for the totals
    by_photo: dict[str, list[dict]] = {}
    for row in rows:
        by_photo.setdefault(row.get("photo") or "(unknown photo)", []).append(row)

    body: list[str] = []
    rendered = crops = 0

    for photo, photo_rows in sorted(by_photo.items()):
        img: Optional[Image.Image] = None
        path = photo_root / photo
        if path.exists():
            try:
                img = load_oriented(path)
            except Exception:
                img = None

        note = "" if img else " <span style='color:#8c2f39'>(photo not found)</span>"
        body.append(f"<h2>{_esc(photo)} — {len(photo_rows)} row(s){note}</h2>")

        for row in photo_rows:
            thumb = None
            if img is not None:
                box = row.get("bottle_bbox") or row.get("tag_bbox")
                if box:
                    thumb = _thumb(img, box)
                    if thumb:
                        crops += 1
            body.append(_row_html(row, thumb))
            rendered += 1

    flagged = sum(1 for r in rows if r.get("needs_review") == "yes")
    priced = sum(1 for r in rows if r.get("price") not in (None, ""))
    stats = "".join(
        f'<div class="stat"><b>{value}</b><span>{label}</span></div>'
        for label, value in (
            ("rows", rendered), ("with a price", priced),
            ("flagged", flagged), ("crops", crops), ("photos", len(by_photo)),
        )
    )

    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        "<div class='sub'>Each row shows the crop the values were read from. "
        "Flagged rows carry a red edge.</div>"
        f"<div class='stats'>{stats}</div>"
        f"{''.join(body)}</body></html>"
    )
    return document, rendered, crops
