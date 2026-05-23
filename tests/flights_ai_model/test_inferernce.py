import os
import unittest
from datetime import date, datetime, timedelta
from data_engineering.adsb.read_adsb import read_adsb
from flights_ai_model.inference import run_inference, get_latest_model_path
from flights_ai_model.validation import flights_df_comparison, get_validation_flights
from pathlib import Path
import polars as pl
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])

import math
from statistics import NormalDist

def required_sample_size(
    population_size=None,   # total number of days (x). None = very large population
    confidence_level=0.95,  # e.g. 0.95 for 95%
    margin_of_error=0.05,   # e.g. 0.05 for ±5%
    p=0.5                   # estimated accuracy (use 0.5 for worst-case)
):
    """
    Returns required sample size (number of days to test).
    Applies finite population correction if population_size is provided.
    """

    # Z-score from confidence level
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2)

    # Base sample size (infinite population)
    n0 = (z**2 * p * (1 - p)) / (margin_of_error**2)

    if population_size is None:
        return math.ceil(n0)

    # Finite population correction
    n = n0 / (1 + (n0 - 1) / population_size)
    return math.ceil(n)


class TestInference(unittest.TestCase):
    def test_inference(self):
        target_date = date(2025,8,1) # maybe 2024 can only expect 96% accuracy
        df_adsb = read_adsb(target_date)
        if target_date.year == 2024:
             percent = 0.96
        else:
            percent = 0.98
        df_val_flights = get_validation_flights(datetime(target_date.year, target_date.month, target_date.day), df_adsb)
        icaos = df_val_flights["icao"].unique().to_list()
        model_path = Path("/Volumes/T2-SSD/planequery/data/models/2026-04-21/2026-04-21_20-08_1771d1f_30days_32epochs/model.pth")
        df_flights = run_inference(target_date, model_path, icaos = icaos)
        df_val_flights = df_val_flights.filter(pl.col("icao").is_in(df_flights["icao"].unique().implode()))
        df, stats = flights_df_comparison(df_flights, df_val_flights)
        self.assertGreater(stats["precision"], percent)
        self.assertGreater(stats["recall"], percent)
        self.assertGreater(stats["f1"], percent)

    def test_flights_data(self):
        # if I have x days of data and I test each day and find the accuracy percentage how many do I need to test to find average accuracy with confidence of 5%
        from data_engineering.flights.flight_type import get_flights
        MAX_DATE = date(2024,1,31)
        MIN_DATE = date(2024,1,1)
        population_size = (MAX_DATE - MIN_DATE).days + 1
        req_sample_size = required_sample_size(population_size = 1, p = 0.95)
        # choice req_sample_size random days between MIN_DATE and MAX_DATE
        import random
        random.seed(42)
        random_days = [MIN_DATE + timedelta(days=offset) for offset in random.sample(range(population_size), req_sample_size)]
        print(random_days)
        stats_list = []
        for day in random_days[0:5]:
            print(day)
            df_flights = get_flights(day)
            df_val_flights = get_validation_flights(datetime(day.year, day.month, day.day), read_adsb(day))
            df_flights = df_flights.filter(pl.col("icao").is_in(df_val_flights["icao"].unique().implode()))
            df, stats = flights_df_comparison(df_flights, df_val_flights)
            stats_list.append(stats)
            print(f"{day}: {len(df_flights)} flights")
        avg_precision = sum(s["precision"] for s in stats_list) / len(stats_list)
        avg_recall = sum(s["recall"] for s in stats_list) / len(stats_list)
        avg_f1 = sum(s["f1"] for s in stats_list) / len(stats_list)
        print(f"Average precision: {avg_precision:.3f}")
        print(f"Average recall: {avg_recall:.3f}")
        print(f"Average f1: {avg_f1:.3f}")
        self.assertGreater(avg_precision, 0.95)
        self.assertGreater(avg_recall, 0.95)
        self.assertGreater(avg_f1, 0.95)
        min_precision = min(s["precision"] for s in stats_list)
        min_recall = min(s["recall"] for s in stats_list)
        min_f1 = min(s["f1"] for s in stats_list)
        print(f"Min precision: {min_precision:.3f}")
        print(f"Min recall: {min_recall:.3f}")
        print(f"Min f1: {min_f1:.3f}")
        self.assertGreater(min_precision, 0.90)
        self.assertGreater(min_recall, 0.90)
        self.assertGreater(min_f1, 0.90)
    # i should make the statsitical calcualtion for how many I need to test and I hsoudl view what flights it was getting wrong for the 2024-01-15. 

#  python -m unittest tests.flights_ai_model.test_inferernce.TestInference.test_inference