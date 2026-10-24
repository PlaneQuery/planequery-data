"""Verify the paper preset cannot silently export an incomplete comparison."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/paper/ads-b_source_comparision_table.py"


def evaluation_rows():
    return [dict(test_date="2024-06-01" if source.startswith("eurocontrol") else "2026-03-01",
                 gold_src=source, adsb_src=provider, target_day_adsb="True", test_src=method,
                 true_positive=8, false_positive=2, false_negative=2, gold_icao_count=5,
                 precision=.8, recall=.8, f1=.8,
                 takeoff_airport_id_match_pct=.75 if method == "algorithm-nn" else .5,
                 landing_airport_id_match_pct=.625)
            for source in ("bts", "sfdps", "eurocontrol", "bts-verified", "sfdps-verified", "eurocontrol-verified")
            for provider in ("adsblol", "opensky", "adsbx")
            for method in ("algorithm-nearest", "algorithm-nn")]


def run_export(tmp_path, rows, *options):
    source = tmp_path / "evaluations.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return subprocess.run([sys.executable, str(SCRIPT), str(source),
                           "--output-dir", str(tmp_path / "tables"), *options],
                          capture_output=True, text=True)


@pytest.mark.parametrize("nn_only", [False, True])
def test_complete_export_and_filtering(tmp_path, nn_only):
    rows = evaluation_rows()
    # Unselected input windows/dates and average rows must never enter the table.
    rows += [{**rows[0], "target_day_adsb": "false", "precision": .1},
             {**rows[0], "test_date": "2024-06-01", "precision": .2},
             {**rows[0], "test_date": "average"}]
    options = ("--test-source", "algorithm-nn") if nn_only else ()
    result = run_export(tmp_path, rows, *options)
    assert result.returncode == 0, result.stderr
    folder = tmp_path / "tables/table_1_adsb_sources"
    with (folder / "table_1_adsb_sources.csv").open() as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == (18 if nn_only else 36)
    assert {row["precision_percent"] for row in exported} == {"80.0"}
    assert {row["target_day_adsb"] for row in exported} == {"true"}
    assert {row["gold_flight_count"] for row in exported} == {"10"}
    assert {row["true_positive"] for row in exported} == {"8"}
    latex = (folder / "table_1_adsb_sources.latex").read_text()
    assert r"\label{tab:table-1-adsb-sources}" in latex
    assert "Target-day-only ADS-B input: true" in latex
    assert "75.00 / 62.50" in latex
    assert "Time bias" not in latex
    assert (r"\multirow{2}{*}{80.00 / 80.00 / 80.00}" in latex) != nn_only
    manifest = json.loads((folder / "table_1_adsb_sources_manifest.json").read_text())
    assert manifest["daily_rows"] == len(exported)
    assert manifest["inputs"][0]["sha256"]


@pytest.mark.parametrize("change, message", [
    ("missing_provider", "Missing 12 of 36"),
    ("wrong_date", "Missing 1 of 36"),
    ("missing_metric", "Unavailable metrics"),
    ("conflict", "Conflicting daily rows"),
])
def test_invalid_input_writes_nothing(tmp_path, change, message):
    rows = evaluation_rows()
    if change == "missing_provider":
        rows = [row for row in rows if row["adsb_src"] != "adsbx"]
    elif change == "wrong_date":
        rows[0]["test_date"] = "2024-06-01"
    elif change == "missing_metric":
        rows[0]["precision"] = None
    else:
        rows.append({**rows[0], "precision": .1})
    result = run_export(tmp_path, rows)
    assert result.returncode != 0
    assert message in result.stderr
    assert not (tmp_path / "tables").exists()
