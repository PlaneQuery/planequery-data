import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from data_engineering.adsb.read_adsb import ADSBSource
from flights.evaluation.evaluation import evaluate as run_evaluation
from flights.evaluation.models import EvaluationStats
from flights.flight_source import (
    ADSBFlightSource,
    flight_path,
    get_flights,
    write_flights,
)
from flights.sources import FLIGHT_SOURCES_BY_NAME


BASELINES_DIR = Path(__file__).parent / "baselines"
BASELINE_PATHS = tuple(sorted(BASELINES_DIR.glob("*.csv")))
MINIMUM_METRICS = (
    "precision",
    "recall",
    "f1",
    "takeoff_airport_id_match_pct",
    "landing_airport_id_match_pct",
    "airport_id_match_pct",
)
# Allow up to a 0.1 percentage-point drop to absorb small run-to-run variation.
MAX_METRIC_REGRESSION = 0.001
MIN_METRIC_INCREASE_TO_REPORT = 0.01
CONFIG_COLUMNS = (
    "test_date",
    "adsb_src",
    "test_src",
)
REQUIRED_COLUMNS = set(CONFIG_COLUMNS) | {"gold_src"} | set(MINIMUM_METRICS)


@dataclass(frozen=True)
class EvaluationConfig:
    test_date: date
    test_source_name: str
    adsb_source: ADSBSource | None


@dataclass(frozen=True)
class Baseline:
    config: EvaluationConfig
    gold_source_name: str
    minimum_metrics: tuple[tuple[str, float], ...]


def parse_optional_adsb_source(value: str) -> ADSBSource | None:
    value = value.strip()
    return ADSBSource(value) if value else None


def load_baselines(path: Path) -> list[Baseline]:
    with path.open(newline="") as baseline_file:
        reader = csv.DictReader(baseline_file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing_columns)}"
            )

        baselines = []
        for row in reader:
            # Evaluation CSVs can contain a derived average row. Daily rows are
            # sufficient because this test evaluates each date independently.
            if row["test_date"] == "average":
                continue
            config = EvaluationConfig(
                test_date=date.fromisoformat(row["test_date"]),
                test_source_name=row["test_src"],
                adsb_source=parse_optional_adsb_source(row["adsb_src"]),
            )
            baselines.append(
                Baseline(
                    config=config,
                    gold_source_name=row["gold_src"],
                    minimum_metrics=tuple(
                        (metric, float(row[metric]))
                        for metric in MINIMUM_METRICS
                    ),
                )
            )
    return baselines


def source_for(config: EvaluationConfig):
    try:
        source = FLIGHT_SOURCES_BY_NAME[config.test_source_name]
    except KeyError:
        raise ValueError(
            f"Unknown test source in baseline: {config.test_source_name}"
        ) from None

    if isinstance(source, ADSBFlightSource):
        if config.adsb_source is None:
            raise ValueError(
                f"{config.test_source_name} requires a non-empty adsb_src"
            )
    elif config.adsb_source is not None:
        raise ValueError(
            f"{config.test_source_name} is not an ADS-B source, so adsb_src "
            "must be empty"
        )
    return source


def prepare_flights(
    config: EvaluationConfig,
    refresh_cache: bool,
) -> None:
    source = source_for(config)
    output_path = flight_path(source, config.test_date, config.adsb_source)

    if not refresh_cache:
        get_flights(source, config.test_date, config.adsb_source)
        assert output_path.is_file(), (
            f"Flight output was not created or reused: {output_path}"
        )
        return

    previous_modified_ns = (
        output_path.stat().st_mtime_ns if output_path.is_file() else None
    )

    write_flights(source, config.test_date, config.adsb_source)

    assert output_path.is_file(), f"Flight output was not created: {output_path}"
    if previous_modified_ns is not None:
        assert output_path.stat().st_mtime_ns > previous_modified_ns, (
            f"Flight output was not refreshed: {output_path}"
        )


def evaluate(
    config: EvaluationConfig,
    gold_sources: list[str],
    adsb_source: ADSBSource = ADSBSource.ADSBLOL,
    refresh_cache: bool = True,
):
    results = run_evaluation(
        test_source=source_for(config),
        gold_source=gold_sources,
        test_dates=config.test_date,
        adsb_source=config.adsb_source or adsb_source,
        refresh_cache=refresh_cache,
    )
    return {
        stats.gold_source: stats
        for stats in results.stats
    }


def assert_meets_baseline(
    baseline: Baseline,
    actual: EvaluationStats,
    baseline_path: Path,
) -> None:
    case = (
        f"{baseline_path.name}: {baseline.config.test_date} "
        f"{baseline.config.test_source_name} vs {baseline.gold_source_name}"
    )
    for metric, minimum in baseline.minimum_metrics:
        measured = float(getattr(actual, metric))
        minimum_with_margin = minimum - MAX_METRIC_REGRESSION
        assert measured >= minimum_with_margin, (
            f"{case}: {metric}={measured:.6f} is below the "
            f"allowed minimum {minimum_with_margin:.6f} "
            f"(baseline {minimum:.6f}, margin {MAX_METRIC_REGRESSION:.6f})"
        )
        if measured >= minimum + MIN_METRIC_INCREASE_TO_REPORT:
            print(
                f"{case}: {metric} increased by {measured - minimum:.6f} "
                f"(baseline {minimum:.6f}, measured {measured:.6f})"
            )


def requested_source_names(
    request: pytest.FixtureRequest,
    option_name: str,
) -> set[str] | None:
    values = request.config.getoption(option_name)
    if values is None:
        return None

    unknown_sources = set(values) - FLIGHT_SOURCES_BY_NAME.keys()
    if unknown_sources:
        raise pytest.UsageError(
            f"Unknown {option_name.replace('_', ' ')}: "
            f"{sorted(unknown_sources)}"
        )
    return set(values)


@pytest.mark.parametrize(
    "baseline_path",
    BASELINE_PATHS,
)
def test_quality_meets_baseline(
    baseline_path: Path,
    request: pytest.FixtureRequest,
):
    baselines = load_baselines(baseline_path)
    assert baselines, f"No daily quality baselines found in {baseline_path}"
    refresh_cache = request.config.getoption("refresh_cache")

    selected_test_sources = requested_source_names(request, "test_source")
    selected_gold_sources = requested_source_names(request, "gold_source")
    requested_adsb_sources = request.config.getoption("adsb_source")
    selected_adsb_sources = (
        tuple(ADSBSource(value) for value in requested_adsb_sources)
        if requested_adsb_sources is not None
        else None
    )
    selected_dates = request.config.getoption("test_dates")

    baselines = [
        baseline
        for baseline in baselines
        if (
            selected_test_sources is None
            or baseline.config.test_source_name in selected_test_sources
        )
        and (
            selected_gold_sources is None
            or baseline.gold_source_name in selected_gold_sources
        )
        and (
            selected_adsb_sources is None
            or baseline.config.adsb_source is None
            or baseline.config.adsb_source in selected_adsb_sources
        )
        and (
            selected_dates is None
            or baseline.config.test_date in selected_dates
        )
    ]
    if not baselines:
        pytest.skip("No baselines match the requested evaluation filters")

    baselines_by_config = defaultdict(list)
    for baseline in baselines:
        baselines_by_config[baseline.config].append(baseline)

    prepared = set()
    for config, configured_baselines in baselines_by_config.items():
        artifact_key = (
            config.test_source_name,
            config.test_date,
            config.adsb_source,
        )
        if artifact_key not in prepared:
            prepare_flights(
                config,
                refresh_cache=refresh_cache,
            )
            prepared.add(artifact_key)

        gold_sources = list(
            dict.fromkeys(
                baseline.gold_source_name for baseline in configured_baselines
            )
        )
        actual_by_gold_source = evaluate(
            config,
            gold_sources,
            adsb_source=(
                selected_adsb_sources[0]
                if selected_adsb_sources
                else ADSBSource.ADSBLOL
            ),
            refresh_cache=refresh_cache,
        )
        for baseline in configured_baselines:
            assert baseline.gold_source_name in actual_by_gold_source, (
                f"Evaluation did not produce stats for "
                f"{baseline.gold_source_name!r}"
            )
            assert_meets_baseline(
                baseline,
                actual_by_gold_source[baseline.gold_source_name],
                baseline_path,
            )
