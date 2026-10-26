"""Requested evaluations must never disappear silently from a paper table."""
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/paper/csvs_to_latex_table.py"
METHODS = ("opensky", "opensky-nearest", "opensky-nn", "algorithm-nearest", "algorithm-nn")
REFERENCES = ("bts", "sfdps", "eurocontrol")


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
    assert manifest["daily_rows"] == 15
    assert "OpenSky-NN" in (folder / "table_0_reference_summary.tex").read_text()


def test_complete_selection_accepts_provider_independent_opensky_nearest(tmp_path):
    rows = evaluation_rows()
    result = run_export(tmp_path, rows)
    assert result.returncode == 0, result.stderr
    folder = tmp_path / "tables/table_0"
    manifest = json.loads((folder / "table_0_manifest.json").read_text())
    assert manifest["daily_rows"] == 15
    latex = (folder / "table_0_reference_summary.tex").read_text()
    assert "OpenSky-Nearest" in latex
    assert latex.count(r"\multirow{3}{*}{80.00 / 80.00 / 80.00}") == 3
    assert latex.count(r"\multirow{2}{*}{80.00 / 80.00 / 80.00}") == 3
    assert latex.index("OpenSky &") < latex.index("OpenSky-Nearest &") < latex.index("OpenSky-NN &")


@pytest.mark.parametrize("method", ["opensky-nearest", "opensky-nn"])
@pytest.mark.parametrize("metric", ["true_positive", "false_positive", "false_negative", "precision", "recall", "f1", "test_flight_count", "gold_flight_count"])
def test_opensky_segmentation_requires_exact_equality(method, metric):
    sys.path.insert(0, str(SCRIPT.parent))
    from csvs_to_latex_table import validate_opensky_segmentation
    rows = evaluation_rows()
    # Both absent derived counts are allowed; unequal present counts are not.
    row = next(row for row in rows if row["test_src"] == method)
    row[metric] = row.get(metric, 10) + 1e-10
    with pytest.raises(ValueError, match=f"OpenSky segmentation metrics differ.*{metric}"):
        validate_opensky_segmentation(rows)


def test_segmentation_mismatch_preserves_outputs_even_when_metric_not_selected(tmp_path):
    rows = evaluation_rows()
    rows[1]["precision"] += 1e-10  # Still renders as 80.00, but differs exactly.
    folder = tmp_path / "tables/table_0"
    folder.mkdir(parents=True)
    sentinel = folder / "table_0_reference_summary.tex"
    sentinel.write_text("previous table")
    source = tmp_path / "input.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = subprocess.run([
        sys.executable, str(SCRIPT), str(source), "--output-dir", str(tmp_path / "tables"),
        "--prefix", "table_0", "--metrics", "recall",
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "OpenSky segmentation metrics differ" in result.stderr
    assert sentinel.read_text() == "previous table"
    assert list(folder.iterdir()) == [sentinel]


def test_opensky_shared_cells_keep_airport_metrics_separate():
    sys.path.insert(0, str(SCRIPT.parent))
    from csvs_to_latex_table import TABLES, summary_latex, validate_opensky_segmentation
    rows = evaluation_rows()[:3]
    for index, row in enumerate(rows):
        row.update(test_flight_count=10, gold_flight_count=10,
                   takeoff_airport_id_match_pct=.5 + index / 10)
    validate_opensky_segmentation(rows)
    metrics = tuple(metric for metric in TABLES["reference_summary"].metrics
                    if metric.key in {"test_flight_count", "precision", "takeoff_airport_id_match_pct"})
    latex = summary_latex(rows, metrics, "table_0", 2)
    assert r"\multirow{3}{*}{10}" in latex
    assert r"\multirow{3}{*}{80.00}" in latex
    assert "OpenSky & \\multirow{3}{*}{10} & \\multirow{3}{*}{80.00} & 50.00" in latex
    assert "OpenSky-Nearest &  &  & 60.00 \\\\*" in latex
    assert "OpenSky-NN &  &  & 70.00 \\\\" in latex


@pytest.mark.parametrize("change, message", [
    ("absent_method", "Missing 3 requested evaluations"),
    ("missing_pair", "Missing 1 requested evaluations"),
    ("absent_reference", "No rows for requested reference datasets: ['bts']"),
    ("missing_provider", "Missing 6 requested evaluations"),
    ("missing_metric", "Unavailable metrics"),
    ("missing_denominator", "gold_icao_count"),
    ("different_gold_count", "Gold flight totals differ"),
])
def test_incomplete_selection_fails_by_default_and_preserves_outputs(tmp_path, change, message):
    rows = evaluation_rows()
    extra = []
    if change == "absent_method":
        rows = [row for row in rows if row["test_src"] != "opensky-nn"]
    elif change == "missing_pair":
        rows.pop()
    elif change == "absent_reference":
        rows = [row for row in rows if row["gold_src"] != "bts"]
    elif change == "missing_provider":
        extra = ["--adsb-source", "adsbx"]
    elif change == "missing_denominator":
        rows[0]["gold_icao_count"] = None
    elif change == "different_gold_count":
        rows[0]["false_negative"] = 3
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


def test_gold_count_in_heading_and_test_counts_in_rows(tmp_path):
    rows = evaluation_rows()
    rows[3]["false_positive"] = 4
    result = run_export(tmp_path, rows, "--metrics", "gold_flight_count", "test_flight_count")
    assert result.returncode == 0, result.stderr
    folder = tmp_path / "tables/table_0"
    latex = (folder / "table_0_reference_summary.tex").read_text()
    assert latex.count("Gold flights: 10") == len(REFERENCES)
    assert "G / T" not in latex
    assert r"\shortstack{Test flights\\T}" in latex
    assert "Nearest & 12 &" in latex
    with (folder / "table_0_reference_summary.csv").open() as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 15
    assert {row["gold_flight_count"] for row in exported} == {"10"}


def test_detailed_table_checks_shared_gold_count():
    sys.path.insert(0, str(SCRIPT.parent))
    from csvs_to_latex_table import TABLES, latex_table, prepare_rows
    rows = evaluation_rows()[:2]
    table = TABLES["segmentation"]
    prepare_rows(rows, {"segmentation": table.metrics}, True)
    latex = latex_table(rows, "segmentation", table, table.metrics, "test", 2)
    assert latex.count("Gold flights: 10") == 1
    rows[1]["gold_flight_count"] = 11
    with pytest.raises(ValueError, match="Gold flight totals differ"):
        latex_table(rows, "segmentation", table, table.metrics, "test", 2)


@pytest.mark.parametrize("mode", ["csv", "cache"])
def test_verified_sources_excluded_without_explicit_filters(tmp_path, mode):
    sys.path.insert(0, str(SCRIPT.parent))
    from csvs_to_latex_table import load_cache_rows, load_rows
    rows = evaluation_rows()
    rows += [{**row, key: "bts-verified"} for key in ("gold_src", "test_src")
             for row in evaluation_rows()]
    if mode == "csv":
        path = tmp_path / "evaluations.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        selected = load_rows([path])
    else:
        entries = [SimpleNamespace(stats=SimpleNamespace(to_row=lambda row=row: dict(row)))
                   for row in rows]
        cache = SimpleNamespace(find_stats=lambda **kwargs: entries)
        selected, _ = load_cache_rows(cache, {})
    assert len(selected) == 15
    assert all(not row[key].endswith("-verified") for row in selected
               for key in ("gold_src", "test_src"))


@pytest.mark.parametrize("mode", ["csv", "cache"])
@pytest.mark.parametrize("matched, takeoff_non_null, landing_non_null", [
    (8, 6, 7), (8, 8, 0), (0, 0, 0),
])
def test_airport_null_tables_use_only_matched_flights(
        tmp_path, mode, matched, takeoff_non_null, landing_non_null):
    sys.path.insert(0, str(SCRIPT.parent))
    from csvs_to_latex_table import TABLES, export_tables, load_cache_rows, load_rows, prepare_rows
    row = dict(evaluation_rows()[0], true_positive=matched,
               takeoff_airport_id_non_null_count=takeoff_non_null,
               landing_airport_id_non_null_count=landing_non_null,
               # Both unmatched test flights have null airports. These old
               # counts/rates must not leak into either table or CSV export.
               takeoff_airport_id_null_count=matched - takeoff_non_null + 2,
               landing_airport_id_null_count=matched - landing_non_null + 2,
               takeoff_airport_id_null_pct=(matched - takeoff_non_null + 2) / (matched + 2),
               landing_airport_id_null_pct=(matched - landing_non_null + 2) / (matched + 2))
    if mode == "csv":
        source = tmp_path / "evaluations.csv"
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        rows = load_rows([source])
    else:
        entry = SimpleNamespace(stats=SimpleNamespace(to_row=lambda: dict(row)))
        rows, _ = load_cache_rows(SimpleNamespace(find_stats=lambda **kwargs: [entry]), {})
    selected = {name: tuple(metric for metric in TABLES[name].metrics
                            if "_null_" in metric.key)
                for name in ("reference_summary", "airport_accuracy")}
    assert prepare_rows(rows, selected, True) == []
    # Reusing prepared rows must not change the derived metrics.
    assert prepare_rows(rows, selected, True) == []
    export_tables(rows, tmp_path, "nulls", selected, 2)
    for name in selected:
        with (tmp_path / f"nulls_{name}.csv").open() as stream:
            exported = next(csv.DictReader(stream))
        for endpoint, non_null in (("takeoff", takeoff_non_null), ("landing", landing_non_null)):
            rate = exported[f"{endpoint}_airport_id_null_percent"]
            if matched:
                assert float(rate) == pytest.approx(100 * (matched - non_null) / matched)
            else:
                assert rate == ""
            if name == "airport_accuracy":
                assert int(exported[f"{endpoint}_airport_id_null_count"]) == matched - non_null
    summary = (tmp_path / "nulls_reference_summary.tex").read_text()
    assert "Airport null rate" in summary
    assert "null airports divide by T" not in summary
    if not matched:
        assert "--- / ---" in summary


@pytest.mark.parametrize("non_null", [None, 9, -1, 1.5])
def test_airport_null_tables_reject_missing_or_invalid_matched_counts(non_null):
    sys.path.insert(0, str(SCRIPT.parent))
    from csvs_to_latex_table import TABLES, prepare_rows
    row = dict(evaluation_rows()[0], takeoff_airport_id_non_null_count=non_null,
               takeoff_airport_id_null_count=3, takeoff_airport_id_null_pct=.3)
    selected = {"reference_summary": tuple(metric for metric in TABLES["reference_summary"].metrics
                                            if metric.key == "takeoff_airport_id_null_pct")}
    with pytest.raises(ValueError, match="takeoff_airport_id_non_null_count"):
        prepare_rows([row], selected, True)
