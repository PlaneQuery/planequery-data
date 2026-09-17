from flights_airport_nn.reference_labels import resolve_faa_point

CATALOG = {
    "KCMA": {
        "type": "small_airport",
        "iso_country": "US",
        "gps_code": "KCMA",
        "local_code": "CMA",
    },
    "MUVT": {
        "type": "medium_airport",
        "iso_country": "CU",
        "gps_code": "MUVT",
        "iata_code": "VTU",
    },
    "K16": {"type": "small_airport", "iso_country": "US", "local_code": "K16"},
    "OTHER": {"type": "small_airport", "iso_country": "PG", "local_code": "BEN"},
}


def test_route_points_do_not_become_foreign_iata_or_local_airports():
    for code in ("VTU", "BEN", "CMA", "RZS", None):
        assert resolve_faa_point(code, CATALOG)[0] == ""


def test_explicit_airport_codes_and_domestic_digit_identifiers():
    assert resolve_faa_point("KCMA", CATALOG)[0] == "KCMA"
    assert resolve_faa_point("MUVT", CATALOG)[0] == "MUVT"
    assert resolve_faa_point("K16", CATALOG)[0] == "K16"


def test_ambiguous_airport_codes_abstain():
    catalog = {**CATALOG, "DUPLICATE": dict(CATALOG["KCMA"])}
    assert resolve_faa_point("KCMA", catalog) == ("", "ambiguous_airport_code")
