"""Tests for ground-truth scoring."""

import csv
from pathlib import Path

import pytest

from wine_ocr.output import COLUMNS
from wine_ocr.verify import load_extracted, load_truth, verify

TRUTH = Path("data/samples/annabella/ground_truth.csv")


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    return path


def _row(price, name, **kw):
    base = dict(
        store="Annabella", wine_name=name, price=price, volume_ml=750,
        raw_tag_text=name, duplicate_of="",
    )
    base.update(kw)
    return base


@pytest.mark.skipif(not TRUTH.exists(), reason="fixture missing")
def test_bundled_ground_truth_parses():
    truth = load_truth(TRUTH)
    assert len(truth) == 11
    assert truth[0].price == 58.99
    assert truth[0].name_contains == "TERRA ROMANA"
    assert truth[0].currency == "RON"
    # The last tag is cut off, so no name is asserted for it.
    assert truth[-1].name_contains == ""


def test_comment_lines_are_ignored(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text(
        "# provenance note\n# another\n"
        "photo,rail,position,price,currency,volume_ml,name_contains,notes\n"
        "a.jpg,1,1,10.00,RON,750,MERLOT,\n",
        encoding="utf-8",
    )
    assert len(load_truth(path)) == 1


def test_perfect_extraction_scores_full_marks(tmp_path):
    truth = load_truth(TRUTH)
    rows = [_row(t.price, f"Wine {t.name_contains or 'x'}") for t in truth]
    report = verify(truth, load_extracted(_write(tmp_path / "e.csv", rows)))
    assert report.found == 11
    assert report.name_correct == report.name_checked == 10  # one asserts no name
    assert report.volume_correct == 11
    assert report.extra == []


def test_uncovered_rows_are_reported_but_not_penalised(tmp_path):
    truth = load_truth(TRUTH)
    rows = [_row(t.price, f"Wine {t.name_contains or 'x'}") for t in truth]
    rows += [_row(9.0 + i, f"Other {i}") for i in range(20)]  # other rails
    report = verify(truth, load_extracted(_write(tmp_path / "e.csv", rows)))
    assert report.found == 11
    assert len(report.extra) == 20


def test_right_price_wrong_product_is_caught(tmp_path):
    truth = load_truth(TRUTH)
    rows = [_row(t.price, "Something Else Entirely") for t in truth]
    report = verify(truth, load_extracted(_write(tmp_path / "e.csv", rows)))
    assert report.found == 11          # prices all matched
    assert report.name_correct == 0    # but nothing is the right wine


def test_repeated_price_matches_two_distinct_rows(tmp_path):
    """Two Castel Huniade tags both read 27.99; each needs its own row."""
    truth = [t for t in load_truth(TRUTH) if t.price == 27.99]
    assert len(truth) == 2
    rows = [_row(27.99, "Castel Huniade Merlot"), _row(27.99, "Castel Huniade Merlot")]
    report = verify(truth, load_extracted(_write(tmp_path / "e.csv", rows)))
    assert report.found == 2

    # With only one extracted row, the second must be reported as missing.
    report = verify(truth, load_extracted(_write(tmp_path / "e2.csv", rows[:1])))
    assert report.found == 1


def test_token_match_is_preferred_when_prices_tie(tmp_path):
    """A tie on price must not steal the row that carries the right name."""
    truth = [t for t in load_truth(TRUTH) if t.name_contains == "MERLOT"][:1]
    rows = [_row(27.99, "Unrelated Wine"), _row(27.99, "Castel Huniade Merlot")]
    report = verify(truth, load_extracted(_write(tmp_path / "e.csv", rows)))
    assert report.outcomes[0].name_ok is True


def test_duplicate_rows_are_not_scored(tmp_path):
    truth = [load_truth(TRUTH)[0]]
    rows = [
        _row(58.99, "Serve Terra Romana", row_id="W1"),
        _row(58.99, "Serve Terra Romana", row_id="W1d1", duplicate_of="W1"),
    ]
    report = verify(truth, load_extracted(_write(tmp_path / "e.csv", rows)))
    assert report.total_rows == 1  # the duplicate never enters scoring
    assert report.found == 1
    assert report.extra == []


def test_diacritics_and_case_do_not_break_matching(tmp_path):
    truth = [t for t in load_truth(TRUTH) if t.name_contains == "SAMBURESTI"]
    rows = [_row(t.price, "domeniile sâmburești fetească neagră") for t in truth]
    report = verify(truth, load_extracted(_write(tmp_path / "e.csv", rows)))
    assert report.name_correct == 1


def test_wrong_volume_is_flagged(tmp_path):
    truth = [load_truth(TRUTH)[0]]
    rows = [_row(58.99, "Serve Terra Romana", volume_ml=1500)]
    report = verify(truth, load_extracted(_write(tmp_path / "e.csv", rows)))
    assert report.found == 1
    assert report.outcomes[0].volume_ok is False


def test_unparseable_price_does_not_crash(tmp_path):
    truth = [load_truth(TRUTH)[0]]
    rows = [_row("", "No price at all")]
    report = verify(truth, load_extracted(_write(tmp_path / "e.csv", rows)))
    assert report.found == 0


def test_nan_price_does_not_match_everything(tmp_path):
    """Regression: float('nan') compares False to everything and matched all."""
    truth = load_truth(TRUTH)
    rows = [_row("nan", "Junk"), _row("NaN", "More junk")]
    report = verify(truth, load_extracted(_write(tmp_path / "e.csv", rows)))
    assert report.found == 0
