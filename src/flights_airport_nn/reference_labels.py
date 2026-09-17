"""Conservative reference labels, separate from airport predictions.

SFDPS route points are not guaranteed to be aerodromes. Never resolve them
through worldwide IATA codes or unrelated countries' local identifiers.
"""

import re

AIRPORT_TYPES = {"large_airport", "medium_airport", "small_airport"}


def resolve_faa_point(code, catalog):
    """Return (airport ident, provenance); ambiguous/non-airport points abstain.

    Four-letter ICAO/GPS airport codes and digit-containing US local airport
    identifiers are admissible. Bare three-letter points remain unverified:
    the string alone cannot distinguish airports from navaids or route fixes.
    Even admissible codes require separate evidence of a physical event.
    """
    code = (code or "").strip().upper()
    if not code:
        return "", "missing_point"
    if re.fullmatch(r"[A-Z]{4}", code):
        matches = {
            ident
            for ident, a in catalog.items()
            if a["type"] in AIRPORT_TYPES
            and code in (a.get("icao_code"), a.get("gps_code"))
        }
        kind = "icao_or_gps_airport_code"
    elif re.fullmatch(r"[A-Z0-9]{2,5}", code) and any(c.isdigit() for c in code):
        matches = {
            ident
            for ident, a in catalog.items()
            if a["type"] in AIRPORT_TYPES
            and a.get("iso_country") == "US"
            and code == a.get("local_code")
        }
        kind = "us_local_airport_code"
    else:
        return "", "unverified_route_point"
    if len(matches) == 1:
        return matches.pop(), kind
    return "", "ambiguous_airport_code" if matches else "unknown_airport_code"
