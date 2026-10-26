"""Check exported percentages, empty denominators, and comparison integrity."""
import csv
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/paper/training_vs_non_training_table.py"


def exposure_rows():
    rows = []
    for method in ("algorithm-nearest", "algorithm-nn"):
        for group in ("training", "nontraining"):
            for endpoint in ("takeoff", "landing"):
                count = 4 if group == "training" else 0
                correct = (3 if method == "algorithm-nn" else 2) if count else 0
                rows.append(dict(test_date="2026-03-01", test_src=method,
                                 gold_src="bts", adsb_src="opensky", target_day_adsb="false",
                                 airport_group=group, endpoint=endpoint,
                                 matched_endpoint_count=count, correct_count=correct,
                                 airport_accuracy=correct/count if count else None,
                                 training_manifest_sha256="shared_training_hash"))
    return rows


def run_export(tmp_path, rows, *options):
    source = tmp_path / "exposure.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return subprocess.run([sys.executable, str(SCRIPT), str(source),
                           "--output-dir", str(tmp_path / "tables"), "--prefix", "test",
                           "--require-complete", *options], capture_output=True, text=True)


def test_export_percentages_and_zero_denominators(tmp_path):
    result = run_export(tmp_path, exposure_rows())
    assert result.returncode == 0, result.stderr
    folder = tmp_path / "tables/test"
    with (folder / "test_training_vs_non_training.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert {row["airport_accuracy_percent"] for row in rows} == {"50.0", "75.0", ""}
    assert all(row["matched_endpoint_count"] == "0" for row in rows if row["airport_group"] == "nontraining")
    assert all(row["accuracy_drop_pp"] == "" for row in rows)
    latex = (folder / "test_training_vs_non_training.tex").read_text()
    assert "75.00 / 75.00" in latex
    assert "--- / ---" in latex
    assert "OpenSky ADS-B" in latex
    assert "NN / OpenSky" not in latex


def test_drop_is_paired_by_method_endpoint_and_provider(tmp_path):
    rows = exposure_rows()
    for row in rows:
        if row["airport_group"] == "nontraining":
            # Nearest drops from 50% to 25%/75%; NN from 75% to 25%/75%.
            correct = 1 if row["endpoint"] == "takeoff" else 3
            row.update(matched_endpoint_count=4, correct_count=correct, airport_accuracy=correct/4)
    other_provider = [{**row, "adsb_src": "adsblol", "correct_count": 4,
                       "airport_accuracy": 1} for row in rows]
    result = run_export(tmp_path, rows + other_provider)
    assert result.returncode == 0, result.stderr
    folder = tmp_path / "tables/test"
    with (folder / "test_training_vs_non_training.csv").open() as stream:
        exported = list(csv.DictReader(stream))
    expected = {("algorithm-nearest", "takeoff"): 25, ("algorithm-nearest", "landing"): -25,
                ("algorithm-nn", "takeoff"): 50, ("algorithm-nn", "landing"): 0}
    for row in exported:
        assert float(row["accuracy_drop_pp"]) == (0 if row["adsb_src"] == "adsblol" else
                                                  expected[row["test_src"], row["endpoint"]])
    latex = (folder / "test_training_vs_non_training.tex").read_text()
    assert "Drop (pp)" in latex
    assert "25.00 / -25.00" in latex
    assert "50.00 / 0.00" in latex


@pytest.mark.parametrize("change, message", [
    ("missing_method", "Incomplete selection"),
    ("mixed_training", "one training manifest"),
    ("conflict", "Conflicting exposure rows"),
    ("wrong_scale", "0–1 fraction"),
])
def test_invalid_comparisons_fail_before_writing(tmp_path, change, message):
    rows = exposure_rows()
    if change == "missing_method":
        rows = rows[:4]
    elif change == "mixed_training":
        rows[-1]["training_manifest_sha256"] = "different_hash"
    elif change == "conflict":
        rows.append({**rows[0], "correct_count": 1})
    else:
        rows[0]["airport_accuracy"] = 50
    result = run_export(tmp_path, rows)
    assert result.returncode != 0
    assert message in result.stderr
    assert not (tmp_path / "tables").exists()
