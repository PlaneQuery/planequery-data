import polars as pl

from data_engineering.utils import OUTPUT_DIR
from pia.main import run_pia_algorithm
from datetime import date
from datetime import date
from pathlib import Path
from data_engineering.flights.flight_type import get_flights
from data_engineering.openairframes.read import scan_latest_faa_registration_icao_for_join
from data_engineering.utils import OUTPUT_DIR
from pia.evaluation.main import evaluation, visualize_graph
from data_engineering.flights.flight_type import flights_df_to_flights

from src.pia.evaluation.main import evaluation, visualize_graph

def test_0():

    mappings_path = Path("/Users/jonahgoode/Documents/PlaneQuery/r-piaicao jacksweeney mappings.csv")

    df_mappings = pl.read_csv(mappings_path).rename(
        {
            "PIA ICAO": "pia_icao",
            "Tail Number": "tailnumber",
            "Date": "date",
        }
    ).with_columns(
        pl.col("pia_icao").str.strip_chars().str.to_lowercase(),
        pl.col("tailnumber").str.strip_chars().str.to_uppercase(),
        pl.col("date").str.strptime(pl.Date, strict=False),
    )
    df_real_icaos = scan_latest_faa_registration_icao_for_join().collect()

    df_mappings = df_mappings.join(
        df_real_icaos,
        left_on="tailnumber",
        right_on="registration",
        how="left",
    )

    df_mappings = df_mappings.filter(pl.col("real_icao").is_not_null())
    df_mappings = df_mappings.filter(pl.col("date") > date(2026,6,1))
    df_flights = get_flights(date(2026, 5, 8), date(2026, 6, 9))
    real_icaos_avaliable = df_flights.filter(pl.col("icao").is_in(df_mappings.get_column("real_icao").unique().to_list())).get_column("icao").unique().to_list()
    df_mappings = df_mappings.filter(pl.col("real_icao").is_in(real_icaos_avaliable))
    pia_icaos = df_mappings.get_column("pia_icao").unique().to_list()
    icaos = set(real_icaos_avaliable) | set(pia_icaos)
    df_flights = df_flights.filter(pl.col("icao").is_in(icaos))

    pia_to_real_icao_map = dict(zip(df_mappings["pia_icao"], df_mappings["real_icao"]))
    gold_dict = dict()
    pia_flights = flights_df_to_flights(df_flights.filter(pl.col("pia") & pl.col("icao").is_in(pia_icaos)))
    for flight in pia_flights:
        gold_dict[flight] = pia_to_real_icao_map[flight.icao]

    icaos=list(gold_dict.values()) + pia_icaos 
    mappings, g = run_pia_algorithm(df_flights)
    stats = evaluation(mappings, gold_dict)
    assert stats == {'expected_mappings': 12, 'predicted_mappings': 12, 'correct_mappings': 12, 'missing_predictions': 0, 'extra_predictions': 0, 'accuracy': 1.0}

def test_1():

    mappings_path = Path("/Users/jonahgoode/Documents/PlaneQuery/r-piaicao jacksweeney mappings.csv")

    df_mappings = pl.read_csv(mappings_path).rename(
        {
            "PIA ICAO": "pia_icao",
            "Tail Number": "tailnumber",
            "Date": "date",
        }
    ).with_columns(
        pl.col("pia_icao").str.strip_chars().str.to_lowercase(),
        pl.col("tailnumber").str.strip_chars().str.to_uppercase(),
        pl.col("date").str.strptime(pl.Date, strict=False),
    )
    df_real_icaos = scan_latest_faa_registration_icao_for_join().collect()

    df_mappings = df_mappings.join(
        df_real_icaos,
        left_on="tailnumber",
        right_on="registration",
        how="left",
    )

    df_mappings = df_mappings.filter(pl.col("real_icao").is_not_null())
    df_mappings = df_mappings.filter(pl.col("date") > date(2026,6,1))
    df_flights = get_flights(date(2026, 5, 8), date(2026, 6, 9))
    real_icaos_avaliable = df_flights.filter(pl.col("icao").is_in(df_mappings.get_column("real_icao").unique().to_list())).get_column("icao").unique().to_list()
    df_mappings = df_mappings.filter(pl.col("real_icao").is_in(real_icaos_avaliable))
    pia_icaos = df_mappings.get_column("pia_icao").unique().to_list()
    extra_pia_icaos = sorted(df_flights.filter(pl.col("pia")).get_column("icao").unique().to_list())[5:10]
    icaos = set(real_icaos_avaliable) | set(pia_icaos) | set(extra_pia_icaos)
    df_flights = df_flights.filter(pl.col("icao").is_in(icaos))

    pia_to_real_icao_map = dict(zip(df_mappings["pia_icao"], df_mappings["real_icao"]))
    gold_dict = dict()
    pia_flights = flights_df_to_flights(df_flights.filter(pl.col("pia") & pl.col("icao").is_in(pia_icaos)))
    for flight in pia_flights:
        gold_dict[flight] = pia_to_real_icao_map[flight.icao]

    icaos=list(gold_dict.values()) + pia_icaos 
    mappings, g = run_pia_algorithm(df_flights)
    stats = evaluation(mappings, gold_dict)
    print(stats)
    assert stats['expected_mappings'] == 12
    assert stats['correct_mappings'] == 12
    assert stats['accuracy'] == 1.0
    # assert stats == {'expected_mappings': 12, 'predicted_mappings': 65, 'correct_mappings': 12, 'missing_predictions': 0, 'extra_predictions': 53, 'accuracy': 1.0}
