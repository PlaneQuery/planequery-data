from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from statistics import fmean

from data_engineering.adsb.read_adsb import ADSBSource
from flights.evaluation.cache import EvaluationCache
from flights.evaluation.evaluation import evaluate
from flights.evaluation.models import EvaluationStats
from flights.sources import FLIGHT_SOURCES_BY_NAME


DEFAULT_TEST_SOURCE = "algorithm-nn"
DEFAULT_GOLD_SOURCES = ("bts", "sfdps", "eurocontrol")
DEFAULT_ADSB_SOURCES = (ADSBSource.ADSBLOL, ADSBSource.ADSBX)


def get_evaluation_stats(
    *,
    cache: EvaluationCache,
    test_source: str,
    gold_sources: Sequence[str],
    adsb_sources: Sequence[ADSBSource],
    test_dates: Sequence[date] | None = None,
    airport_nn_model_path: Path | None = None,
    refresh_cache: bool = False,
) -> tuple[EvaluationStats, ...]:
    """Get cached stats, creating missing requested dates through evaluation."""
    if test_dates is not None:
        return evaluate(
            test_source=test_source,
            gold_source=gold_sources,
            test_dates=test_dates,
            adsb_source=adsb_sources,
            write_outputs=False,
            airport_nn_model_path=airport_nn_model_path,
            refresh_cache=refresh_cache,
            evaluation_cache=cache,
        ).stats

    if refresh_cache:
        raise ValueError("--refresh-cache requires --test-dates")

    return tuple(
        entry.stats
        for entry in cache.find_stats(
            test_source=test_source,
            gold_source=gold_sources,
            adsb_source=adsb_sources,
        )
    )


def paired_stats(
    stats: Sequence[EvaluationStats],
    left_source: ADSBSource,
    right_source: ADSBSource,
) -> dict[str, list[tuple[EvaluationStats, EvaluationStats]]]:
    """Pair provider stats by gold source and date."""
    by_source = {
        (item.gold_source, item.test_date, item.adsb_source): item
        for item in stats
    }
    pairs: dict[str, list[tuple[EvaluationStats, EvaluationStats]]] = {}
    days = sorted(
        {(item.gold_source, item.test_date) for item in stats}
    )
    for gold_source, test_date in days:
        try:
            left = by_source[(gold_source, test_date, left_source)]
            right = by_source[(gold_source, test_date, right_source)]
        except KeyError as error:
            missing_source = error.args[0][2]
            raise ValueError(
                "Missing cached provider evaluation for "
                f"{gold_source} {test_date}: {missing_source.value}"
            ) from None
        pairs.setdefault(gold_source, []).append(
            (left, right)
        )
    return pairs


def print_report(
    stats: Sequence[EvaluationStats],
    left_source: ADSBSource,
    right_source: ADSBSource,
) -> None:
    pairs_by_gold_source = paired_stats(stats, left_source, right_source)
    if not pairs_by_gold_source:
        raise ValueError("No cached evaluations matched the requested sources")

    comparison_name = f"{left_source.value} - {right_source.value}"
    for gold_source, pairs in pairs_by_gold_source.items():
        print(gold_source.upper())
        print(f"Days: {len(pairs)}")
        print(
            f"Avg Precision Diff ({comparison_name}): "
            f"{fmean(left.precision - right.precision for left, right in pairs)}"
        )
        print(
            f"Avg Recall Diff ({comparison_name}): "
            f"{fmean(left.recall - right.recall for left, right in pairs)}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare cached evaluation precision and recall between two ADS-B "
            "providers. Supplying dates creates any missing cache entries."
        )
    )
    parser.add_argument(
        "--test-source",
        choices=FLIGHT_SOURCES_BY_NAME,
        default=DEFAULT_TEST_SOURCE,
    )
    parser.add_argument(
        "--gold-source",
        nargs="+",
        choices=FLIGHT_SOURCES_BY_NAME,
        default=list(DEFAULT_GOLD_SOURCES),
    )
    parser.add_argument(
        "--adsb-source",
        nargs=2,
        type=ADSBSource,
        choices=tuple(ADSBSource),
        default=list(DEFAULT_ADSB_SOURCES),
        metavar=("LEFT", "RIGHT"),
    )
    parser.add_argument(
        "--test-dates",
        nargs="+",
        type=date.fromisoformat,
        help="Dates to get from cache or evaluate when missing.",
    )
    parser.add_argument(
        "--airport-nn-model",
        type=Path,
        help="Checkpoint to use when creating algorithm-nn evaluations.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recalculate requested dates instead of accepting cache hits.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Evaluation cache directory (default: OUTPUT_DIR cache).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.adsb_source[0] == args.adsb_source[1]:
        raise ValueError("--adsb-source values must be different")

    cache = EvaluationCache(args.cache_dir)
    stats = get_evaluation_stats(
        cache=cache,
        test_source=args.test_source,
        gold_sources=args.gold_source,
        adsb_sources=args.adsb_source,
        test_dates=args.test_dates,
        airport_nn_model_path=args.airport_nn_model,
        refresh_cache=args.refresh_cache,
    )
    print_report(stats, args.adsb_source[0], args.adsb_source[1])


if __name__ == "__main__":
    main()
