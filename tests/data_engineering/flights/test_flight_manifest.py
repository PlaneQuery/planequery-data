from datetime import date

from flights.flight_manifest import FlightManifest, flight_manifest_path


def test_flight_manifest_json_round_trip(tmp_path):
    manifest = FlightManifest(
        flight_date=date(2024, 1, 1),
        observed_commit_sha="flight-state-sha",
        knowledge_cutoff_date=date(2024, 2, 1),
        knowledge_commit_sha="knowledge-sha",
        catalog_available_date=date(2026, 9, 24),
        policy_version="1",
        adsb_source="adsblol",
        adsb_parquet_version=6,
    )
    path = flight_manifest_path(tmp_path / "data.parquet")

    manifest.write(path)

    assert path == tmp_path / "flight_manifest.json"
    assert FlightManifest.read(path) == manifest
