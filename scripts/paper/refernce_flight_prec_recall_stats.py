from __future__ import annotations

from statistics import fmean
from datetime import date

from data_engineering.adsb.read_adsb import ADSBSource
from flights.evaluation.cache import EvaluationCache


TEST_SOURCE = "algorithm-nn"
GOLD_SOURCES = ("bts", "sfdps", "eurocontrol")
LEFT_ADSB_SOURCE = ADSBSource.ADSBLOL
RIGHT_ADSB_SOURCE = ADSBSource.ADSBX


def main(cache: EvaluationCache | None = None) -> None:
    cache = cache or EvaluationCache()
    stats = {
        (entry.stats.gold_source, entry.stats.test_date, entry.stats.adsb_source):
        entry.stats
        for entry in cache.find_stats(
            test_source=TEST_SOURCE,
            gold_source=GOLD_SOURCES,
            adsb_source=(LEFT_ADSB_SOURCE, RIGHT_ADSB_SOURCE),
        )
    }

    for gold_source in GOLD_SOURCES:
        dates = sorted(
            test_date
            for source, test_date, adsb_source in stats
            if source == gold_source and adsb_source == LEFT_ADSB_SOURCE
        )
        pairs = [
            (
                stats[(gold_source, test_date, LEFT_ADSB_SOURCE)],
                stats[(gold_source, test_date, RIGHT_ADSB_SOURCE)],
            )
            for test_date in dates
        ]

        print(gold_source.upper())
        print(
            "Avg Precision Diff:",
            fmean(left.precision - right.precision for left, right in pairs),
        )
        print(
            "Avg Recall Diff:",
            fmean(left.recall - right.recall for left, right in pairs),
        )


if __name__ == "__main__":
    main()

# ---- desired days
SFPDS_DAYS = [date(2026,2,1), date(2026,3,1)]
BTS_DAYS =  SFPDS_DAYS + [date(2025, 6,1)]
EUROCONTROL_DAYS = [date(2025,6,1), date(2025,7,1), date(2025,8,1)]