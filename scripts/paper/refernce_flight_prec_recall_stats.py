from dataclasses import dataclass
from datetime import date

import polars as pl

from data_engineering.adsb.read_adsb import ADSBSource
from flights.evaluation.models import EvaluationStats


# Data Analysis

bts_csv_path = "/Volumes/T2-SSD/planequery/data/experiments/comparison/2026-10-03_20-56-20-039148_18884de_flights_evaluation.csv" 
sfpds_csv_path = "/Volumes/T2-SSD/planequery/data/experiments/comparison/2026-10-04_19-02-38-491861_18884de_flights_evaluation.csv"
eurocontrol_csv_path = "/Volumes/T2-SSD/planequery/data/experiments/comparison/2026-10-04_19-05-37-057977_bbf4675_flights_evaluation.csv"  # needs OpenSky and actually make it general for 3 adsb sources. 

def main(path):
    df = pl.read_csv(path)
    stats = [
        EvaluationStats.from_row(row)
        for row in df.filter(
            pl.col("test_date") != "average"
        ).to_dicts()
    ]
    stats_by_day: dict[date, list] = dict()
    for stat in stats:
        stats_by_day.setdefault(stat.test_date, []).append(stat)

    metrics_by_day = dict()

    @dataclass
    class DayMetrics:
        date: date
        adsblol_stats: EvaluationStats # | None? 
        adsbx_stats: EvaluationStats

    for date_key, stats in stats_by_day.items():
        dm = DayMetrics(
            date=date_key,
            adsblol_stats=next(
                stat for stat in stats if stat.adsb_source == ADSBSource.ADSBLOL
            ),
            adsbx_stats=next(
                stat for stat in stats if stat.adsb_source == ADSBSource.ADSBX
            ),
        )
        metrics_by_day[date_key] = dm
    precision_diffs = []
    recall_diffs = []
    for dm in metrics_by_day.values():
        precision_diff = dm.adsblol_stats.precision - dm.adsbx_stats.precision
        recall_diff = dm.adsblol_stats.recall - dm.adsbx_stats.recall
        precision_diffs.append(precision_diff)
        recall_diffs.append(recall_diff)

    print("Avg Precision Diff:", sum(precision_diffs) / len(precision_diffs))
    print("Avg Recall Diff:", sum(recall_diffs) / len(recall_diffs))

print("BTS")        
main(bts_csv_path)
print("SFPDS")
main(sfpds_csv_path)
print("Eurocontrol")
main(eurocontrol_csv_path)

# TODO: Write down all dates in this file. I shoudl have 2 days for both.
