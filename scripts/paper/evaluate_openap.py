"""Report the OpenAP segmentation baseline through the standard evaluator.

PYTHONPATH=src python scripts/paper/evaluate_openap.py
"""

import argparse
import hashlib
import inspect
import json
from dataclasses import asdict
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

import polars as pl

from airports.runtime import airport_catalog_for_date
from data_engineering.adsb.canonical import ADSB_PARQUET_VERSIONS, ADSBSource
from flights.evaluation.cache import EVALUATION_CACHE_VERSION
from flights.evaluation.evaluation import evaluate, write_evaluation_csv
from flights.sources import FLIGHT_SOURCES_BY_NAME
from planequery_data.utils import current_commit_hash


def timing_metrics(comparison: pl.DataFrame) -> dict:
    """Endpoint errors conditional on matches; these do not change matching."""
    matched = comparison.filter(pl.col("match_status") == "both")
    metrics = {}
    errors = []
    for endpoint in ("takeoff", "landing"):
        error = (
            matched[f"{endpoint}_time"] - matched[f"{endpoint}_time_df1"]
        ).dt.total_milliseconds() / 60000
        errors.append(error.abs())
        metrics[f"{endpoint}_signed_median_minutes"] = error.median()
        metrics[f"{endpoint}_absolute_median_minutes"] = error.abs().median()
        metrics[f"{endpoint}_absolute_p95_minutes"] = error.abs().quantile(0.95)
    metrics["matched_both_endpoints_within_5min_pct"] = (
        ((errors[0] <= 5) & (errors[1] <= 5)).mean()
    )
    unmatched = comparison.filter(pl.col("match_status") == "df0_only")
    duration = (unmatched["landing_time"] - unmatched["takeoff_time"]).dt.total_seconds()
    metrics["unmatched_test_under_5min_count"] = (duration < 300).sum()
    metrics["unmatched_test_over_12h_count"] = (duration > 43200).sum()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dates", nargs="+", type=date.fromisoformat,
                        default=[date(2026, 3, 1)])
    parser.add_argument("--adsb-source", type=ADSBSource, choices=tuple(ADSBSource),
                        default=ADSBSource.ADSBLOL)
    parser.add_argument("--output-stem", type=Path,
                        default=Path("reports/openap_segmentation_2026-03-01"))
    args = parser.parse_args()
    test_sources = ["openap", "algorithm-nn"]
    gold_sources = ["bts", "sfdps", "bts-verified", "sfdps-verified"]
    results = evaluate(
        test_source=test_sources, gold_source=gold_sources,
        test_dates=args.test_dates, adsb_source=args.adsb_source,
    )
    rows = [
        {**entry.stats.to_row(), **timing_metrics(entry.comparison)}
        for entry in results.entries
    ]
    frame = pl.DataFrame(rows)
    write_evaluation_csv(frame, args.output_stem.with_suffix(".csv"))

    from openap.phase import FlightPhase
    from flights.openap import segmentation as adapter

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "base_commit": current_commit_hash(),
        "openap_version": version("openap"),
        "openap_phase_sha256": hashlib.sha256(
            Path(inspect.getfile(FlightPhase)).read_bytes()
        ).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
        "source_versions": {
            name: FLIGHT_SOURCES_BY_NAME[name].version
            for name in [*test_sources, *gold_sources]
        },
        "test_dates": [d.isoformat() for d in args.test_dates],
        "adsb_source": args.adsb_source.value,
        "adsb_parquet_version": ADSB_PARQUET_VERSIONS[args.adsb_source],
        "evaluation_cache_version": EVALUATION_CACHE_VERSION,
        "target_day_adsb": False,
        "phase_window_seconds": 60,
        "split_gap_seconds": adapter.MAX_GAP_SECONDS,
        "matching_endpoint_tolerance_minutes": 60,
        "scoring_departure_window_utc": ["01:00", "23:00"],
        "scoring_window_inclusive": True,
        "airport_catalogs": [
            asdict(airport_catalog_for_date(d).provenance) for d in args.test_dates
        ],
        "limitations": [
            "Flight interval accuracy, not accuracy against phase-label ground truth.",
            "Endpoint error metrics are conditional on successful one-hour matches.",
            "Each gold source defines its own aircraft cohort; other aircraft are excluded.",
            "Verified references are ADS-B-filtered subsets, not independent datasets.",
            "OpenAP's final NA time window is preserved and excluded from flights.",
            "Nearest-airport guesses are postprocessing and do not affect segmentation.",
        ],
        "results": rows,
    }
    args.output_stem.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8",
    )
    print(frame.select("test_src", "gold_src", "true_positive", "false_positive",
                       "false_negative", "precision", "recall", "f1"))
    print(f"Reports saved to {args.output_stem}.csv and .json")


if __name__ == "__main__":
    main()
