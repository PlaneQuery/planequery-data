from data_engineering.adsb.canonical import ADSBSource
from flights.evaluation.evaluation import evaluate
from flights.sources import ALGORITHM_NN_SOURCE, BTS_SOURCE, OPENSKY_SOURCE
from datetime import date

results = evaluate(
    test_source=[ALGORITHM_NN_SOURCE, OPENSKY_SOURCE],
    gold_source=BTS_SOURCE,
    test_dates=[date(2026,3,1), date(2024,6,1)],
    adsb_source=[ADSBSource.ADSBLOL, ADSBSource.OPENSKY]
)
