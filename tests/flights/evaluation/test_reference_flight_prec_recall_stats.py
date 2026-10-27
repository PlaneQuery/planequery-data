from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from data_engineering.adsb.read_adsb import ADSBSource
from flights.evaluation.models import EvaluationStats
from scripts.paper import refernce_flight_prec_recall_stats as report


def _stats(gold_source: str, adsb_source: ADSBSource) -> EvaluationStats:
    true_positive = 8 if adsb_source == ADSBSource.ADSBLOL else 9
    return EvaluationStats(
        test_date=date(2026, 2, 1),
        adsb_source=adsb_source,
        test_source="algorithm-nn",
        gold_source=gold_source,
        gold_icao_count=10,
        gold_icao_missing_from_test_count=0,
        true_positive=true_positive,
        false_positive=10 - true_positive,
        false_negative=10 - true_positive,
        takeoff_airport_id_match_count=true_positive,
        landing_airport_id_match_count=true_positive,
        airport_id_match_count=true_positive,
        takeoff_airport_id_non_null_count=true_positive,
        landing_airport_id_non_null_count=true_positive,
    )


def _cache_entries(*, include_adsbx: bool = True) -> list[SimpleNamespace]:
    sources = [ADSBSource.ADSBLOL]
    if include_adsbx:
        sources.append(ADSBSource.ADSBX)
    return [
        SimpleNamespace(stats=_stats(gold_source, adsb_source))
        for gold_source in report.GOLD_SOURCES
        for adsb_source in sources
    ]


def test_report_reads_evaluation_stats_from_cache(capsys):
    cache = Mock()
    cache.find_stats.return_value = _cache_entries()

    report.main(cache)

    cache.find_stats.assert_called_once()
    output = capsys.readouterr().out
    assert "BTS" in output
    assert "SFDPS" in output
    assert "EUROCONTROL" in output
    assert "Avg Precision Diff:" in output


def test_missing_cached_provider_fails():
    cache = Mock()
    cache.find_stats.return_value = _cache_entries(include_adsbx=False)

    with pytest.raises(KeyError):
        report.main(cache)
