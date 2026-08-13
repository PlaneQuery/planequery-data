from datetime import date, datetime
import os
from pathlib import Path
import polars as pl
import pytest

from data_engineering.utils import OUTPUT_DIR
from pia.ground_truth import construct_ground_truth_mappings

from data_engineering.flights.flight_type import FlightIndex, create_pia_evaluation_df, flights_df_to_flights, flights_to_flights_df, get_flights, pia_or_american_ladd_icao_filter, create_pia_evaluation_df
from pia.resolver import build_candidates, check_candidate_mapping_paths, resolve_pia_mappings
from pia.evaluation.main import evaluation
from pia.evaluation.pia_graph_0 import flights_0, flights_2, flights_3, test_mapping_dict_3, test_mapping_dict_0
from pia.main import main
from pia.evaluation.real_flights_0 import a039a2_flights

def test_simulator_flights_0():
    # TODO: What is the random assignment percentage
    from pia.evaluation.simulator_flights import get_simulated_flights_0

    df_flights_sim, gold_mapping = get_simulated_flights_0()
    mapping_dict = resolve_pia_mappings(flights_df_to_flights(df_flights_sim))
    correct = 0
    total = 0
    for flight, real_icao in gold_mapping.items():
        if flight in mapping_dict:
            total += 1
            if mapping_dict[flight] == real_icao:
                correct += 1
    print(f"Resolved accuracy: {correct}/{total} = {correct/total:.2%}")

def test_real_flights_0():
    start_date = date(2025,4,1)
    end_date = date(2025,7,16)
    correct_number = 35
    stats = pia_tester(start_date, end_date, correct_number)
    print_real_flight_stats(stats)

def test_real_flights_1():
    start_date = date(2025,4,1)
    end_date = date(2025,7,16)
    correct_number = 35
    stats = pia_tester(start_date, end_date, correct_number, filter=False)
    print_real_flight_stats(stats)

def test_real_flights_2():
    start_date = date(2025,4,1)
    end_date = date(2025,7,16)
    stats = pia_tester(start_date, end_date, correct_number=0)
    print_real_flight_stats(stats)


def test_real_flights_3():
    start_date = date(2025,1,1)
    end_date = date(2026,7,5)
    stats = pia_tester(start_date, end_date, correct_number=0, filter=False)
    print_real_flight_stats(stats)


def test_real_flights_4(request):
    start_date = date(2026,1,1)
    end_date = date(2026,7,1)
    stats = pia_tester(
        start_date,
        end_date,
        correct_number=0,
        filter=False,
        test_name=request.node.name,
    )
    print_real_flight_stats(stats)
    assert stats['flight_overall_accuracy'] >= 0.3633
    assert stats['traceability_index'] >= 0.243

@pytest.mark.parametrize(
    ("start_date", "expected_stats"),
    [
        pytest.param(
            date(2026, 5, 1),
            {
                "correct_pia_icaos": 6,
                "resolved_pia_icaos": 23,
                "expected_pia_icaos": 30,
                "pia_icao_resolved_accuracy": 6 / 23,
                "pia_icao_coverage": 23 / 30,
                "pia_icao_overall_accuracy": 6 / 30,
                "unknown_label_pia_icaos": 93,
                "resolved_expected_flight_mappings": 391,
                "expected_flight_mappings": 454,
                "correct_flight_mappings": 122,
                "flight_coverage": 391 / 454,
                "flight_resolved_accuracy": 122 / 391,
                "flight_overall_accuracy": 122 / 454,
                "traceable_aircraft": 5,
                "expected_aircraft": 28,
                "traceability_index": 5 / 28,
            },
            id="may",
        ),
        pytest.param(
            date(2026, 6, 1),
            {
                "correct_pia_icaos": 4,
                "resolved_pia_icaos": 17,
                "expected_pia_icaos": 21,
                "pia_icao_resolved_accuracy": 4 / 17,
                "pia_icao_coverage": 17 / 21,
                "pia_icao_overall_accuracy": 4 / 21,
                "unknown_label_pia_icaos": 77,
                "resolved_expected_flight_mappings": 153,
                "expected_flight_mappings": 182,
                "correct_flight_mappings": 63,
                "flight_coverage": 153 / 182,
                "flight_resolved_accuracy": 63 / 153,
                "flight_overall_accuracy": 63 / 182,
                "traceable_aircraft": 4,
                "expected_aircraft": 20,
                "traceability_index": 4 / 20,
            },
            id="june",
        ),
    ],
)
def test_real_flights_5(request, start_date, expected_stats):
    end_date = date(2026,7,1)
    stats = pia_tester(
        start_date,
        end_date,
        correct_number=0,
        filter=False,
        test_name=request.node.name,
    )
    print_real_flight_stats(stats)
    minimum_metrics = (
        "pia_icao_resolved_accuracy",
        "pia_icao_overall_accuracy",
        "flight_resolved_accuracy",
        "flight_overall_accuracy",
        "traceability_index",
    )
    assert all(stats[key] >= expected_stats[key] for key in minimum_metrics)

def print_real_flight_stats(stats):
    print(
        "PIA ICAO resolved accuracy: "
        f"{stats['correct_pia_icaos']}/{stats['resolved_pia_icaos']} "
        f"= {stats['pia_icao_resolved_accuracy']:.2%}"
    )
    print(
        "PIA ICAO coverage: "
        f"{stats['resolved_pia_icaos']}/{stats['expected_pia_icaos']} "
        f"= {stats['pia_icao_coverage']:.2%}"
    )
    print(
        "PIA ICAO overall accuracy: "
        f"{stats['correct_pia_icaos']}/{stats['expected_pia_icaos']} "
        f"= {stats['pia_icao_overall_accuracy']:.2%}"
    )
    print(
        "Observed PIA ICAOs assigned UNKNOWN_LABEL: "
        f"{stats['unknown_label_pia_icaos']}"
    )
    print(
        "Flight coverage: "
        f"{stats['resolved_expected_flight_mappings']}/"
        f"{stats['expected_flight_mappings']} "
        f"= {stats['flight_coverage']:.2%}"
    )
    print(
        "Flight resolved accuracy: "
        f"{stats['correct_flight_mappings']}/"
        f"{stats['resolved_expected_flight_mappings']} "
        f"= {stats['flight_resolved_accuracy']:.2%}"
    )
    print(
        "Flight overall accuracy: "
        f"{stats['correct_flight_mappings']}/{stats['expected_flight_mappings']} "
        f"= {stats['flight_overall_accuracy']:.2%}"
    )
    print(
        "Traceability: "
        f"{stats['traceable_aircraft']}/{stats['expected_aircraft']} aircraft "
        f"= {stats['traceability_index']:.2%}"
    )

def save_pia_mapping_audit(flights, gold_mapping, mappings, pia_to_real_icao, test_name):
    """Write the chronological per-flight PIA mapping audit for one test run."""
    real_icaos = set(pia_to_real_icao.values())
    audit_rows = []
    for flight in flights:
        expected_icao = gold_mapping.get(flight)
        predicted_icao = mappings.get(flight)
        if expected_icao is None:
            mapping_status = "not_evaluated"
        elif predicted_icao is None:
            mapping_status = "missing_prediction"
        elif predicted_icao == expected_icao:
            mapping_status = "correct"
        else:
            mapping_status = "incorrect"

        audit_rows.append({
            "flight_id": flight.flight_id,
            "takeoff_time": flight.takeoff_time,
            "landing_time": flight.landing_time,
            "observed_icao": flight.icao,
            "pia_icao": flight.icao if flight.pia else None,
            # Visible real-ICAO flights are included as history anchors.
            "real_icao": expected_icao or (
                flight.icao if flight.icao in real_icaos else None
            ),
            "predicted_icao": predicted_icao,
            "correct": (
                predicted_icao == expected_icao
                if expected_icao is not None
                else None
            ),
            "mapping_status": mapping_status,
            "pia": flight.pia,
            "callsign": flight.callsign,
            "takeoff_airport_ident": flight.takeoff_airport_ident,
            "landing_airport_ident": flight.landing_airport_ident,
        })

    safe_test_name = test_name or "pia_tester"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        OUTPUT_DIR
        / "data"
        / "tests"
        / "test_pia"
        / safe_test_name
        / f"test_pia_{safe_test_name}_{current_time}.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_schema = {
        "flight_id": pl.String,
        "takeoff_time": pl.Datetime,
        "landing_time": pl.Datetime,
        "observed_icao": pl.String,
        "pia_icao": pl.String,
        "real_icao": pl.String,
        "predicted_icao": pl.String,
        "correct": pl.Boolean,
        "mapping_status": pl.String,
        "pia": pl.Boolean,
        "callsign": pl.String,
        "takeoff_airport_ident": pl.String,
        "landing_airport_ident": pl.String,
    }
    (
        pl.DataFrame(audit_rows, schema=audit_schema)
        .sort("takeoff_time")
        .write_parquet(output_path)
    )
    print(f"Wrote PIA mapping audit: {output_path}")
    return output_path


def pia_tester(
    start_date: date,
    end_date: date,
    correct_number: int,
    filter=True,
    test_name=None,
    pia_location_radius_km: float = 10,
):
    """Run the PIA evaluation and save a per-flight mapping audit as Parquet.

    ``test_name`` controls the audit directory; pytest callers can pass
    ``request.node.name``.
    """
    df_mappings, icaos_0, icaos_1, both = construct_ground_truth_mappings()
    pia_to_real_icao = dict(zip(df_mappings["pia_icao"], df_mappings["real_icao"]))
    df_flights = get_flights(start_date, end_date)
    for pia_icao, real_icao in list(pia_to_real_icao.items()):
        n_pia_flights = len(df_flights.filter(pl.col("icao") == pia_icao))
        n_real_flights = len(df_flights.filter(pl.col("icao") == real_icao))
        if n_pia_flights == 0 or n_real_flights == 0:
            # print(f"Warning: No flights found for pia_icao {pia_icao} or real_icao {real_icao}")
            pia_to_real_icao.pop(pia_icao) 
    print(f"len(pia_to_real_icao): {len(pia_to_real_icao)}")
    surviving_icaos = set(pia_to_real_icao)
    only_ground_control = surviving_icaos & (icaos_0 - icaos_1)
    only_broadcast = surviving_icaos & (icaos_1 - icaos_0)
    in_both = surviving_icaos & both
    assert only_ground_control | only_broadcast | in_both == surviving_icaos
    print(f"len(only_ground_control after filter): {len(only_ground_control)}")
    print(f"len(only_broadcast after filter): {len(only_broadcast)}")
    print(f"len(in both after filter): {len(in_both)}")

    print(pia_to_real_icao)
    if filter:
        df_flights = df_flights.filter(pl.col("icao").is_in(pia_to_real_icao.keys() | pia_to_real_icao.values()))
    else:
        df_flights = df_flights.filter(pl.col("icao").str.len_chars() == 6)
        df_flights = df_flights.filter(pia_or_american_ladd_icao_filter())
    print(f"Number of unique pia icaos in flights: {df_flights.filter(pl.col('pia')).select('icao').n_unique()}")
    flights = flights_df_to_flights(df_flights)
    flight_index = FlightIndex(flights, pia_location_radius_km)
    candidates = build_candidates(flight_index)
    candidate_path_results = check_candidate_mapping_paths(
        candidates,
        flight_index,
        pia_to_real_icao,
    )
    possible_mappings = sum(candidate_path_results.values())
    print(
        "Ground-control mappings with a path in the candidate graph: "
        f"{possible_mappings}/{len(candidate_path_results)}"
    )
    for pia_icao, path_exists in candidate_path_results.items():
        if not path_exists:
            print(
                "No candidate path for ground-control mapping: "
                f"{pia_icao} -> {pia_to_real_icao[pia_icao]}"
            )

    mappings = resolve_pia_mappings(flights, pia_location_radius_km)
    gold_mapping = {
        flight: pia_to_real_icao[flight.icao]
        for flight in flights
        if flight.icao in pia_to_real_icao
    }
    known_mappings = {
        flight: real_icao
        for flight, real_icao in mappings.items()
        if flight in gold_mapping
    }
    stats = evaluation(known_mappings, gold_mapping)
    pia_icaos = {flight.icao for flight in flight_index.get_pia_flights()}
    resolved_pia_icaos = {flight.icao for flight in mappings}
    stats["unknown_label_pia_icaos"] = len(pia_icaos - resolved_pia_icaos)

    save_pia_mapping_audit(
        flights,
        gold_mapping,
        mappings,
        pia_to_real_icao,
        test_name,
    )
    evaluation_df = create_pia_evaluation_df(flights, known_mappings, gold_mapping)
    output_path = OUTPUT_DIR / "data" / "tmp" / "pia_tester_flights_df.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_df.write_parquet(output_path)
    print(f"Wrote flights DataFrame with real_icao column: {output_path}")
    return stats
