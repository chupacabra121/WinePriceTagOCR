"""Drive the CLI commands against a stubbed client.

`extract_photo` is covered in test_end_to_end; this covers the command wrapper
around it — argument handling, --append merging, --crops, and the offline
commands that must never touch the network.
"""

import csv
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wine_ocr import cli
from wine_ocr import extract as ex
from tests.test_end_to_end import SAMPLE, _Stub

runner = CliRunner()
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample photo not present")


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the API client and satisfy the key check."""
    stub = _Stub()
    monkeypatch.setattr(ex, "build_client", lambda *a, **k: stub)
    monkeypatch.setattr(cli.ex, "build_client", lambda *a, **k: stub)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    return stub


def _read(path):
    with path.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _invoke(*args):
    result = runner.invoke(cli.app, list(args))
    assert result.exit_code == 0, result.output + str(result.exception)
    return result


def test_extract_writes_every_output(tmp_path, stubbed):
    out = tmp_path / "out"
    _invoke("extract", str(SAMPLE.parent), "--out", str(out),
            "--no-cache", "--workers", "1", "--limit", "1")

    for name in ("wines.csv", "wines.xlsx", "unmatched_tags.csv", "extractions.jsonl"):
        assert (out / name).exists(), name

    rows = _read(out / "wines.csv")
    assert rows
    primary = [r for r in rows if not r["duplicate_of"]]
    assert len(primary) == 1
    assert primary[0]["price"] == "54.59"
    assert primary[0]["store"] == "Annabella"


def test_store_override_beats_the_folder(tmp_path, stubbed):
    out = tmp_path / "out"
    _invoke("extract", str(SAMPLE.parent), "--out", str(out), "--store", "Kaufland",
            "--no-cache", "--workers", "1", "--limit", "1")
    row = [r for r in _read(out / "wines.csv") if not r["duplicate_of"]][0]
    assert row["store"] == "Kaufland"
    assert row["store_source"] == "cli"
    # The photo's own reading is preserved rather than overwritten.
    assert row["store_read_from_photo"] == "Annabella"


def test_append_merges_instead_of_replacing(tmp_path, stubbed):
    out = tmp_path / "out"
    _invoke("extract", str(SAMPLE), "--out", str(out), "--no-cache",
            "--workers", "1", "--store", "Annabella")
    first = len([r for r in _read(out / "wines.csv") if not r["duplicate_of"]])
    assert first == 1

    # Same wine again from a different store: appends rather than dedupes away.
    _invoke("extract", str(SAMPLE), "--out", str(out), "--no-cache",
            "--workers", "1", "--store", "Lidl", "--append")
    rows = [r for r in _read(out / "wines.csv") if not r["duplicate_of"]]
    assert {r["store"] for r in rows} == {"Annabella", "Lidl"}
    assert len(rows) == 2

    # And the merged table still round-trips its numbers.
    assert all(float(r["price"]) == 54.59 for r in rows)


def test_crops_are_written_only_for_flagged_rows(tmp_path, stubbed, monkeypatch):
    out = tmp_path / "out"
    # Force a review flag by removing the price from the stubbed answer.
    import tests.test_end_to_end as e2e

    monkeypatch.setitem(e2e.WINE, "price", None)
    try:
        _invoke("extract", str(SAMPLE), "--out", str(out), "--crops",
                "--no-cache", "--workers", "1")
        crops = list((out / "crops").glob("*.jpg"))
        assert crops, "a flagged row with a bbox should produce a crop"
    finally:
        e2e.WINE["price"] = 54.59


def test_missing_key_exits_cleanly(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    result = runner.invoke(cli.app, ["extract", str(SAMPLE)])
    assert result.exit_code == 2
    assert "No API key" in result.output


def test_no_images_found_is_an_error(tmp_path, stubbed):
    result = runner.invoke(cli.app, ["extract", str(tmp_path)])
    assert result.exit_code == 1
    assert "No images found" in result.output


# --------------------------------------------------------------------------
# Offline commands must not need a key or a network
# --------------------------------------------------------------------------


def test_plan_needs_no_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = _invoke("plan", str(SAMPLE), "--out", str(tmp_path / "tiles"))
    assert "4284x5712" in result.output
    assert list((tmp_path / "tiles").glob("*.jpg"))


def test_estimate_needs_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    whole = _invoke("estimate", str(SAMPLE.parent), "--tiling", "whole").output
    auto = _invoke("estimate", str(SAMPLE.parent), "--tiling", "auto").output
    assert "Cost estimate" in whole

    def calls(text):
        line = next(ln for ln in text.splitlines() if "Model calls" in ln)
        return int("".join(c for c in line if c.isdigit()))

    assert calls(auto) > calls(whole), "tiling should cost more calls than one-shot"


def test_verify_needs_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    truth = Path("data/samples/annabella/ground_truth.csv")
    extracted = tmp_path / "wines.csv"
    from wine_ocr.output import COLUMNS
    from wine_ocr.verify import load_truth

    with extracted.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for t in load_truth(truth):
            w.writerow({c: "" for c in COLUMNS} | {
                "store": "Annabella", "wine_name": f"Wine {t.name_contains}",
                "price": t.price, "volume_ml": 750, "raw_tag_text": t.name_contains,
            })

    output = _invoke("verify", str(extracted), "--truth", str(truth)).output
    assert "11/11" in output
