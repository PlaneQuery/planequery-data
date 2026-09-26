import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from data_engineering.adsb.read_adsb import ADSBSource
from flights.evaluation.evaluation import main
from flights.flight_source import ADSBFlightSource, flight_path, write_flights
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
CONFIG_COLUMNS = (
    "test_date",
    "adsb_src",
    "matching_adsb_src",
    "test_src",
    "gold_src",
    "use_all_icaos",
    "filter_rotorcraft",
    "pia_or_american_ladd_only",
)
REQUIRED_COLUMNS = set(CONFIG_COLUMNS) | set(MINIMUM_METRICS)


@dataclass(frozen=True)
class EvaluationConfig:
    test_date: date
    test_source_name: str
    adsb_source: ADSBSource | None
    matching_adsb_source: ADSBSource
    use_all_icaos: bool
    filter_rotorcraft: bool
    pia_or_american_ladd_only: bool


@dataclass(frozen=True)
class Baseline:
    config: EvaluationConfig
    gold_source_name: str
    minimum_metrics: tuple[tuple[str, float], ...]


def parse_bool(value: str, *, column: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{column} must be 'true' or 'false', got {value!r}")
    return normalized == "true"


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
                matching_adsb_source=ADSBSource(row["matching_adsb_src"]),
                use_all_icaos=parse_bool(
                    row["use_all_icaos"], column="use_all_icaos"
                ),
                filter_rotorcraft=parse_bool(
                    row["filter_rotorcraft"], column="filter_rotorcraft"
                ),
                pia_or_american_ladd_only=parse_bool(
                    row["pia_or_american_ladd_only"],
                    column="pia_or_american_ladd_only",
                ),
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


def regenerate_flights(config: EvaluationConfig) -> None:
    source = source_for(config)
    output_path = flight_path(source, config.test_date, config.adsb_source)
    previous_modified_ns = (
        output_path.stat().st_mtime_ns if output_path.is_file() else None
    )

    write_flights(source, config.test_date, config.adsb_source)

    assert output_path.is_file(), f"Flight output was not created: {output_path}"
    if previous_modified_ns is not None:
        assert output_path.stat().st_mtime_ns > previous_modified_ns, (
            f"Flight output was not refreshed: {output_path}"
        )


def evaluate(config: EvaluationConfig, gold_sources: list[str]):
    _, stats = main(
        test_src=source_for(config),
        gold_src=gold_sources,
        test_dates=config.test_date,
        adsb_src=config.adsb_source or ADSBSource.ADSBLOL,
        matching_adsb_src=config.matching_adsb_source,
        use_all_icaos=config.use_all_icaos,
        filter_rotorcraft=config.filter_rotorcraft,
        pia_or_american_ladd_only=config.pia_or_american_ladd_only,
        write_outputs=False,
    )
    return {
        row["gold_src"]: row
        for row in stats.to_dicts()
        if row["test_date"] != "average"
    }


def assert_meets_baseline(
    baseline: Baseline,
    actual: dict[str, int | float | str | bool | None],
    baseline_path: Path,
) -> None:
    case = (
        f"{baseline_path.name}: {baseline.config.test_date} "
        f"{baseline.config.test_source_name} vs {baseline.gold_source_name}"
    )
    for metric, minimum in baseline.minimum_metrics:
        measured = float(actual[metric])
        minimum_with_margin = minimum - MAX_METRIC_REGRESSION
        assert measured >= minimum_with_margin, (
            f"{case}: {metric}={measured:.6f} is below the "
            f"allowed minimum {minimum_with_margin:.6f} "
            f"(baseline {minimum:.6f}, margin {MAX_METRIC_REGRESSION:.6f})"
        )


@pytest.mark.parametrize(
    "baseline_path",
    BASELINE_PATHS,
    ids=lambda path: path.stem,
)
def test_quality_meets_baseline(baseline_path: Path):
    baselines = load_baselines(baseline_path)
    assert baselines, f"No daily quality baselines found in {baseline_path}"

    baselines_by_config = defaultdict(list)
    for baseline in baselines:
        baselines_by_config[baseline.config].append(baseline)

    regenerated = set()
    for config, configured_baselines in baselines_by_config.items():
        artifact_key = (
            config.test_source_name,
            config.test_date,
            config.adsb_source,
        )
        if artifact_key not in regenerated:
            regenerate_flights(config)
            regenerated.add(artifact_key)

        gold_sources = list(
            dict.fromkeys(
                baseline.gold_source_name for baseline in configured_baselines
            )
        )
        actual_by_gold_source = evaluate(config, gold_sources)
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
