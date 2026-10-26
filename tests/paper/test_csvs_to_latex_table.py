"""Requested evaluations must never disappear silently from a paper table."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/paper/csvs_to_latex_table.py"
METHODS = ("opensky", "opensky-nn", "algorithm-nearest", "algorithm-nn")
REFERENCES = ("bts", "sfdps", "eurocontrol", "bts-verified", "sfdps-verified", "eurocontrol-verified")


def evaluation_rows():
    return [dict(test_date="2024-06-01" if reference.startswith("eurocontrol") else "2026-03-01",
                 gold_src=reference, test_src=method, target_day_adsb="false",
                 adsb_src="" if method.startswith("opensky") else "opensky",
                 true_positive=8, false_positive=2, false_negative=2,
                 gold_icao_count=5, precision=.8, recall=.8, f1=.8)
            for reference in REFERENCES for method in METHODS]


def run_export(tmp_path, rows, *extra):
    source = tmp_path / "evaluations.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evaluation_rows()[0]))
        writer.writeheader()
        writer.writerows(rows)
    return subprocess.run([
        sys.executable, str(SCRIPT), str(source), "--output-dir", str(tmp_path / "tables"),
        "--prefix", "table_0", "--test-source", *METHODS, "--gold-source", *REFERENCES,
        "--adsb-source", "opensky", "none", "--reference-date", "bts=2026-03-01",
        "--reference-date", "sfdps=2026-03-01", "--reference-date", "eurocontrol=2024-06-01",
        "--metrics", "precision", "recall", "f1", *extra,
    ], capture_output=True, text=True)


def test_complete_selection_accepts_provider_independent_opensky_nn(tmp_path):
    result = run_export(tmp_path, evaluation_rows())
    assert result.returncode == 0, result.stderr
    folder = tmp_path / "tables/table_0"
    manifest = json.loads((folder / "table_0_manifest.json").read_text())
    assert manifest["daily_rows"] == 24
    assert "OpenSky-NN" in (folder / "table_0_reference_summary.tex").read_text()


@pytest.mark.parametrize("change, message", [
    ("absent_method", "Missing 6 requested evaluations"),
    ("missing_pair", "Missing 1 requested evaluations"),
    ("absent_reference", "Missing evaluations for requested reference sources: bts-verified"),
    ("missing_provider", "Missing 12 requested evaluations"),
    ("missing_metric", "Unavailable metrics"),
    ("missing_denominator", "gold_icao_count"),
])
def test_incomplete_selection_fails_by_default_and_preserves_outputs(tmp_path, change, message):
    rows = evaluation_rows()
    extra = []
    if change == "absent_method":
        rows = [row for row in rows if row["test_src"] != "opensky-nn"]
    elif change == "missing_pair":
        rows.pop()
    elif change == "absent_reference":
        rows = [row for row in rows if row["gold_src"] != "bts-verified"]
    elif change == "missing_provider":
        extra = ["--adsb-source", "adsbx"]
    elif change == "missing_denominator":
        rows[0]["gold_icao_count"] = None
    else:
        rows[0]["precision"] = None
    folder = tmp_path / "tables/table_0"
    folder.mkdir(parents=True)
    sentinel = folder / "table_0_reference_summary.tex"
    sentinel.write_text("previous table")
    result = run_export(tmp_path, rows, *extra)
    assert result.returncode != 0
    assert message in result.stderr
    assert sentinel.read_text() == "previous table"
    assert list(folder.iterdir()) == [sentinel]


def test_explicit_missing_date_in_detailed_export_fails(tmp_path):
    sys.path.insert(0, str(SCRIPT.parent))
    from csvs_to_latex_table import validate_completeness
    with pytest.raises(ValueError, match="2026-03-02"):
        validate_completeness([evaluation_rows()[0]], {
            "test_src": {"opensky"}, "gold_src": {"bts"},
            "test_date": {"2026-03-01", "2026-03-02"}, "target_day_adsb": {"false"},
        })
