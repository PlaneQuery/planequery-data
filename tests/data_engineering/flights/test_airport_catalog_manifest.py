from datetime import date

from flights.airport_catalog_manifest import AirportCatalogManifest


def test_airport_catalog_manifest_json_round_trip(tmp_path):
    manifest = AirportCatalogManifest(
        flight_date=date(2024, 1, 1),
        observed_commit_sha="flight-state-sha",
        knowledge_cutoff_date=date(2024, 2, 1),
        knowledge_commit_sha="knowledge-sha",
        catalog_available_date=date(2026, 9, 24),
        policy_version="1",
    )
    path = tmp_path / "airport_catalog.json"

    manifest.write(path)

    assert AirportCatalogManifest.read(path) == manifest
