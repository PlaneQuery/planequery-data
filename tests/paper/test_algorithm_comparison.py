"""Verify the algorithm-comparison preset excludes OpenAP."""
import csv
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/paper/algoirthm_comparision.py"


def evaluation_rows():
    return [
        dict(
            test_date="2024-06-01" if reference == "eurocontrol" else "2026-03-01",
            gold_src=reference,
            adsb_src="opensky",
            target_day_adsb="false",
            test_src=method,
            true_positive=8,
            false_positive=2,
            false_negative=2,
            gold_icao_count=5,
            precision=.8,
            recall=.8,
            f1=.8,
            takeoff_airport_id_match_pct=.75,
            landing_airport_id_match_pct=.625,
        )
        for reference in ("bts", "sfdps", "eurocontrol")
        for method in ("sun", "plane-notify", "traffic", "openap")
    ]


def test_export_excludes_openap(tmp_path):
    source = tmp_path / "evaluations.csv"
    rows = evaluation_rows()
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), "--output-dir", str(tmp_path / "tables")],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    folder = tmp_path / "tables/algorithm_comparison"
    with (folder / "algorithm_comparison.csv").open() as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 9
    assert {row["test_src"] for row in exported} == {"sun", "plane-notify", "traffic"}
    assert "openap" not in (folder / "algorithm_comparison.tex").read_text().lower()
    manifest = json.loads((folder / "algorithm_comparison_manifest.json").read_text())
    assert "openap" not in manifest["method_labels"]
