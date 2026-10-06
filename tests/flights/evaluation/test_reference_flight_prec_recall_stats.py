from datetime import date
from types import SimpleNamespace

import pytest

from data_engineering.adsb.read_adsb import ADSBSource
from flights.evaluation.models import EvaluationStats
from scripts.paper import refernce_flight_prec_recall_stats as report


def _stats(
    *,
    test_date: date,
    adsb_source: ADSBSource,
    gold_source: str = "bts",
    true_positive: int = 8,
    false_positive: int = 2,
    false_negative: int = 2,
) -> EvaluationStats:
    return EvaluationStats(
        test_date=test_date,
        adsb_source=adsb_source,
        test_source="algorithm-nn",
        gold_source=gold_source,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        takeoff_airport_id_match_count=true_positive,
        landing_airport_id_match_count=true_positive,
        airport_id_match_count=true_positive,
    )


def test_report_uses_evaluation_stats_objects(capsys):
    stats = (
        _stats(
            test_date=date(2026, 2, 1),
            adsb_source=ADSBSource.ADSBLOL,
        ),
        _stats(
            test_date=date(2026, 2, 1),
            adsb_source=ADSBSource.ADSBX,
            true_positive=9,
            false_positive=1,
            false_negative=1,
        ),
    )

    report.print_report(stats, ADSBSource.ADSBLOL, ADSBSource.ADSBX)

    output = capsys.readouterr().out
    assert "BTS" in output
    assert "Days: 1" in output
    assert "Avg Precision Diff (adsblol - adsbx):" in output


def test_pairing_rejects_a_missing_provider():
    stats = (
        _stats(
            test_date=date(2026, 2, 1),
            adsb_source=ADSBSource.ADSBLOL,
        ),
    )

    with pytest.raises(ValueError, match="adsbx"):
        report.paired_stats(stats, ADSBSource.ADSBLOL, ADSBSource.ADSBX)


def test_script_defaults_to_cache_discovery_without_dates():
    args = report.parse_args([])

    assert args.test_source == "algorithm-nn"
    assert args.gold_source == ["bts", "sfdps", "eurocontrol"]
    assert args.adsb_source == [ADSBSource.ADSBLOL, ADSBSource.ADSBX]
    assert args.test_dates is None


def test_requested_dates_use_evaluation_to_create_cache_entries(
    monkeypatch,
    tmp_path,
):
    expected = (
        _stats(
            test_date=date(2026, 2, 1),
            adsb_source=ADSBSource.ADSBLOL,
        ),
    )
    received = {}

    def fake_evaluate(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(stats=expected)

    monkeypatch.setattr(report, "evaluate", fake_evaluate)

    actual = report.get_evaluation_stats(
        cache=report.EvaluationCache(tmp_path),
        test_source="algorithm-nn",
        gold_sources=["bts"],
        adsb_sources=[ADSBSource.ADSBLOL],
        test_dates=[date(2026, 2, 1)],
    )

    assert actual == expected
    assert received["write_outputs"] is False
    assert received["test_dates"] == [date(2026, 2, 1)]
