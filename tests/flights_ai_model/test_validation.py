from datetime import date, datetime

import polars as pl

import flights_ai_model.validation as validation_module
from flights_ai_model.validation import (
    ADSB_VERIFICATION_STATUS_COL,
    ADSB_VERIFIED_COL,
    VALIDATION_TIER_ADSB_VERIFIED,
    VALIDATION_TIER_ALL,
    VALIDATION_TIER_UNVERIFIED_OR_HARD,
    add_adsb_verification_status,
    flights_df_comparison,
    flights_df_comparison_by_validation_tier,
    get_validation_flights,
    validation_flight_tiers,
)


def test_flights_df_comparison_matches_k_prefixed_us_airports_to_iata_codes():
    df1 = pl.DataFrame(
        [
            {
                "icao": "abc123",
                "takeoff_time": datetime(2026, 1, 1, 12, 0),
                "landing_time": datetime(2026, 1, 1, 13, 0),
                "takeoff_airport_ident": "KTUS",
                "landing_airport_ident": "TUS",
            }
        ]
    )
    df2 = pl.DataFrame(
        [
            {
                "icao": "abc123",
                "takeoff_time": datetime(2026, 1, 1, 12, 0),
                "landing_time": datetime(2026, 1, 1, 13, 0),
                "takeoff_airport_ident": "TUS",
                "landing_airport_ident": "KTUS",
            }
        ]
    )

    _df, stats = flights_df_comparison(df1, df2)

    assert stats["airport_ident_pct_match"] == 1.0


def test_add_adsb_verification_status_keeps_hard_gold_flights(monkeypatch):
    df_gold = pl.DataFrame(
        [
            {
                "icao": "verified",
                "takeoff_time": datetime(2026, 1, 1, 12, 0),
                "landing_time": datetime(2026, 1, 1, 13, 0),
            },
            {
                "icao": "hard",
                "takeoff_time": datetime(2026, 1, 1, 14, 0),
                "landing_time": datetime(2026, 1, 1, 15, 0),
            },
        ]
    )
    df_adsb = pl.DataFrame()
    monkeypatch.setattr(
        validation_module,
        "get_icaos_with_perfect_match",
        lambda _df_gold, _df_adsb: {"verified"},
    )

    df = add_adsb_verification_status(df_gold, df_adsb)

    assert df.height == 2
    status_by_icao = dict(zip(df["icao"].to_list(), df[ADSB_VERIFICATION_STATUS_COL].to_list()))
    verified_by_icao = dict(zip(df["icao"].to_list(), df[ADSB_VERIFIED_COL].to_list()))
    assert status_by_icao == {
        "verified": VALIDATION_TIER_ADSB_VERIFIED,
        "hard": VALIDATION_TIER_UNVERIFIED_OR_HARD,
    }
    assert verified_by_icao == {"verified": True, "hard": False}


def test_validation_flight_tiers_returns_all_verified_and_hard(monkeypatch):
    df_gold = pl.DataFrame(
        [
            {
                "icao": "verified",
                "takeoff_time": datetime(2026, 1, 1, 12, 0),
                "landing_time": datetime(2026, 1, 1, 13, 0),
            },
            {
                "icao": "hard",
                "takeoff_time": datetime(2026, 1, 1, 14, 0),
                "landing_time": datetime(2026, 1, 1, 15, 0),
            },
        ]
    )
    monkeypatch.setattr(
        validation_module,
        "get_icaos_with_perfect_match",
        lambda _df_gold, _df_adsb: {"verified"},
    )

    tiers = validation_flight_tiers(df_gold, pl.DataFrame())

    assert tiers[VALIDATION_TIER_ALL]["icao"].to_list() == ["verified", "hard"]
    assert tiers[VALIDATION_TIER_ADSB_VERIFIED]["icao"].to_list() == ["verified"]
    assert tiers[VALIDATION_TIER_UNVERIFIED_OR_HARD]["icao"].to_list() == ["hard"]


def test_get_validation_flights_defaults_to_all_and_can_filter_tiers(monkeypatch):
    df_source = pl.DataFrame(
        [
            {
                "icao": "verified",
                "takeoff_time": datetime(2025, 1, 1, 12, 0),
                "landing_time": datetime(2025, 1, 1, 13, 0),
            },
            {
                "icao": "hard",
                "takeoff_time": datetime(2025, 1, 1, 14, 0),
                "landing_time": datetime(2025, 1, 1, 15, 0),
            },
        ]
    )
    monkeypatch.setattr(validation_module, "get_bts_flights_for_day", lambda _date, use_cache=True: df_source)
    monkeypatch.setattr(
        validation_module,
        "get_icaos_with_perfect_match",
        lambda _df_gold, _df_adsb: {"verified"},
    )

    df_all = get_validation_flights(date(2025, 1, 1), pl.DataFrame())
    df_verified = get_validation_flights(
        datetime(2025, 1, 1),
        pl.DataFrame(),
        validation_tier=VALIDATION_TIER_ADSB_VERIFIED,
    )
    df_hard = get_validation_flights(
        date(2025, 1, 1),
        pl.DataFrame(),
        validation_tier=VALIDATION_TIER_UNVERIFIED_OR_HARD,
    )

    assert df_all["icao"].to_list() == ["verified", "hard"]
    assert ADSB_VERIFICATION_STATUS_COL in df_all.columns
    assert df_verified["icao"].to_list() == ["verified"]
    assert df_hard["icao"].to_list() == ["hard"]


def test_flights_df_comparison_by_validation_tier_reports_hard_misses(monkeypatch):
    df_pred = pl.DataFrame(
        [
            {
                "icao": "verified",
                "takeoff_time": datetime(2026, 1, 1, 12, 0),
                "landing_time": datetime(2026, 1, 1, 13, 0),
            },
        ]
    )
    df_gold = pl.DataFrame(
        [
            {
                "icao": "verified",
                "takeoff_time": datetime(2026, 1, 1, 12, 0),
                "landing_time": datetime(2026, 1, 1, 13, 0),
            },
            {
                "icao": "hard",
                "takeoff_time": datetime(2026, 1, 1, 14, 0),
                "landing_time": datetime(2026, 1, 1, 15, 0),
            },
        ]
    )
    monkeypatch.setattr(
        validation_module,
        "get_icaos_with_perfect_match",
        lambda _df_gold, _df_adsb: {"verified"},
    )

    outputs = flights_df_comparison_by_validation_tier(df_pred, df_gold, pl.DataFrame())

    assert outputs[VALIDATION_TIER_ALL][1]["recall"] == 0.5
    assert outputs[VALIDATION_TIER_ADSB_VERIFIED][1]["f1"] == 1.0
    assert outputs[VALIDATION_TIER_UNVERIFIED_OR_HARD][1]["recall"] == 0.0
