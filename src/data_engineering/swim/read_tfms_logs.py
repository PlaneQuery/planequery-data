from pathlib import Path
import gzip
from lxml import etree
import polars as pl

from data_engineering.utils import OUTPUT_DIR
OUTPUT_DIR = OUTPUT_DIR / "data"

# ── Container elements that wrap the payload inside a fltdMessage ─────────────
# When flattening, the content of the matched container is hoisted to the row
# *without* the container name as a prefix.  This means qualifiedAircraftId.*,
# position.*, etc. share the same flat key regardless of which message type
# (trackInformation, flightPlanInformation, …) the message is.
_FLTD_MSG_CONTAINERS = frozenset({
    "trackInformation",
    "flightPlanInformation",
    "flightPlanAmendmentInformation",
    "ncsmFlightModify",
    "ncsmFlightTimes",
    "ncsmFlightSectors",
    "ncsmFlightRoute",
    "departureInformation",
    "arrivalInformation",
    "boundaryCrossingUpdate",
    "flightPlanCancellation",
    "ncsmFlightCreate",
    "ncsmFlightScheduleActivate",
    "oceanicReport",
})

# Drop XML namespace boilerplate from row keys.
_SUPPRESS_SUFFIXES = (".uom", ".srsName", ".nil")
_SUPPRESS_PREFIXES = ("schemaLocation",)

# ── Schema: human-readable name → (element path list, attribute key | None) ───
# Paths are relative to the *container* element (not the fltdMessage root).
# key=None  → text content of the final element.
# key=str   → XML attribute on the final element.
#
# Message types that contribute each group:
#   trackInformation               → position_*, speed_knots, reported_altitude, track_*
#   flightPlanInformation          → filed_airspeed, requested_altitude, route_text,
#                                    coordination_*, route_*, flightStatusAndSpec.*
#   FlightRoute / FlightScheduleActivate
#                                  → assigned_altitude, ground_speed, star, dp, route_*
#   FlightModify / FlightCreate    → cdm_*, airline_*_time, flight_status
#   FlightTimes                    → times_*  (etd/eta sit directly under container)
#   departureInformation           → departure_time, fltd_*
#   arrivalInformation             → arrival_time, fltd_*
#   boundaryCrossingUpdate         → boundary_*, mach, route_text
#   flightPlanAmendmentInformation → amendment_*
#   flightPlanCancellation         → only qualifiedAircraftId fields
#   oceanicReport                  → position_*, track_*, speed_knots
#   fiMessage types (RSTR, GS, GDP, GADV, FXA, APTC, DICE, RAPT, TMI_FLIGHT_LIST)
#                                  → only the message-header fields (msg_type etc.)

TFMS_FLTD_SCHEMA: dict[str, tuple[list[str], str | None]] = {
    # ── Message header (fltdMessage / fiMessage attributes) ──────────────────
    "msg_type":                         ([], "msgType"),
    "source_timestamp":                 ([], "sourceTimeStamp"),
    "source_facility":                  ([], "sourceFacility"),
    # acid duplicates qualifiedAircraftId.aircraftId; kept for quick access.
    "acid":                             ([], "acid"),
    "dep_arpt":                         ([], "depArpt"),
    "arr_arpt":                         ([], "arrArpt"),
    "flight_ref":                       ([], "flightRef"),
    "airline":                          ([], "airline"),
    "major":                            ([], "major"),
    "fd_trigger":                       ([], "fdTrigger"),
    "cdm_participant":                  ([], "cdmPart"),
    "sensitivity":                      ([], "sensitivity"),

    # ── Core flight identity (qualifiedAircraftId — all fltdMessage types) ───
    "aircraft_category":                (["qualifiedAircraftId"], "aircraftCategory"),        # JET, PISTON, TURBO
    "user_category":                    (["qualifiedAircraftId"], "userCategory"),            # COMMERCIAL, AIR TAXI, OTHER
    "gufi":                             (["qualifiedAircraftId", "gufi"], None),
    "igtd":                             (["qualifiedAircraftId", "igtd"], None),             # initial gate time of departure
    "qualified_acid":                    (["qualifiedAircraftId", "aircraftId"], None),           # same as acid attr; suppresses unmapped-column noise
    "computer_facility":                (["qualifiedAircraftId", "computerId", "facilityIdentifier"], None),
    "computer_id_number":               (["qualifiedAircraftId", "computerId", "idNumber"], None),
    "departure_airport":                (["qualifiedAircraftId", "departurePoint", "airport"], None),
    "arrival_airport":                  (["qualifiedAircraftId", "arrivalPoint", "airport"], None),

    # ── Position (trackInformation, oceanicReport.reportedPositionData) ──────
    # Coordinates are DMS — convert to decimal with pos_lat/pos_lon helpers if needed.
    "position_lat_deg":                 (["position", "latitude", "latitudeDMS"], "degrees"),
    "position_lat_dir":                 (["position", "latitude", "latitudeDMS"], "direction"),   # NORTH | SOUTH
    "position_lat_min":                 (["position", "latitude", "latitudeDMS"], "minutes"),
    "position_lat_sec":                 (["position", "latitude", "latitudeDMS"], "seconds"),
    "position_lon_deg":                 (["position", "longitude", "longitudeDMS"], "degrees"),
    "position_lon_dir":                 (["position", "longitude", "longitudeDMS"], "direction"),  # EAST | WEST
    "position_lon_min":                 (["position", "longitude", "longitudeDMS"], "minutes"),
    "position_lon_sec":                 (["position", "longitude", "longitudeDMS"], "seconds"),
    "position_time":                    (["timeAtPosition"], None),

    # ── Speed ─────────────────────────────────────────────────────────────────
    "speed_knots":                      (["speed"], None),                             # trackInformation: direct text
    "filed_airspeed":                   (["speed", "filedTrueAirSpeed"], None),        # flightPlanInformation, FlightRoute
    "ground_speed":                     (["speed", "groundSpeed"], None),              # FlightRoute
    "mach":                             (["speed", "mach"], None),                     # boundaryCrossingUpdate

    # ── Altitude ──────────────────────────────────────────────────────────────
    "reported_altitude":                (["reportedAltitude", "assignedAltitude", "simpleAltitude"], None),  # trackInformation
    "assigned_altitude":                (["altitude", "assignedAltitude", "simpleAltitude"], None),          # FlightRoute, FlightScheduleActivate
    "requested_altitude":               (["altitude", "requestedAltitude", "simpleAltitude"], None),         # flightPlanInformation

    # ── Aircraft specs (flat form: flightPlanInformation, boundaryCrossing, dept) ──
    "flight_aircraft_specs":            (["flightAircraftSpecs"], None),               # ICAO type code as element text
    "equipment_qualifier":              (["flightAircraftSpecs"], "equipmentQualifier"),
    "special_aircraft_qualifier":       (["flightAircraftSpecs"], "specialAircraftQualifier"),

    # ── Aircraft specs (expanded form: FlightModify, FlightCreate, FlightTimes, et al.) ──
    "aircraft_model":                   (["flightStatusAndSpec", "aircraftModel"], None),
    "aircraft_spec_type":               (["flightStatusAndSpec", "aircraftSpecification"], None),  # ICAO type text
    "aircraft_engine_class":            (["flightStatusAndSpec", "aircraftSpecification"], "aircraftEngineClass"),
    "spec_equipment_qualifier":         (["flightStatusAndSpec", "aircraftSpecification"], "equipmentQualifier"),
    "flight_status":                    (["flightStatusAndSpec", "flightStatus"], None),

    # ── Track data (trackInformation, oceanicReport) ──────────────────────────
    "track_eta_type":                   (["ncsmTrackData", "eta"], "etaType"),
    "track_eta":                        (["ncsmTrackData", "eta"], "timeValue"),
    "track_rvsm_equipped":              (["ncsmTrackData", "rvsmData"], "equipped"),
    "track_rvsm_compliant":             (["ncsmTrackData", "rvsmData"], "currentCompliance"),
    "track_rvsm_future_compliant":      (["ncsmTrackData", "rvsmData"], "futureCompliance"),
    "track_arrival_fix":                (["ncsmTrackData", "arrivalFixAndTime"], "fixName"),
    "track_arrival_fix_time":           (["ncsmTrackData", "arrivalFixAndTime"], "arrTime"),
    "track_departure_fix":              (["ncsmTrackData", "departureFixAndTime"], "fixName"),
    "track_departure_fix_time":         (["ncsmTrackData", "departureFixAndTime"], "arrTime"),
    "next_event_lat":                   (["ncsmTrackData", "nextEvent"], "latitudeDecimal"),
    "next_event_lon":                   (["ncsmTrackData", "nextEvent"], "longitudeDecimal"),

    # ── Route data (flightPlanInformation, FlightRoute, FlightScheduleActivate) ──
    "route_etd_type":                   (["ncsmRouteData", "etd"], "etdType"),
    "route_etd":                        (["ncsmRouteData", "etd"], "timeValue"),
    "route_eta_type":                   (["ncsmRouteData", "eta"], "etaType"),
    "route_eta":                        (["ncsmRouteData", "eta"], "timeValue"),
    "diversion_indicator":              (["ncsmRouteData", "diversionIndicator"], None),
    "route_rvsm_equipped":              (["ncsmRouteData", "rvsmData"], "equipped"),
    "star":                             (["ncsmRouteData", "star"], "routeName"),
    "dp":                               (["ncsmRouteData", "dp"], "routeName"),
    "star_transition":                  (["ncsmRouteData", "starTransitionFix"], None),
    "dp_transition":                    (["ncsmRouteData", "dpTransitionFix"], None),
    "route_arrival_fix":                (["ncsmRouteData", "arrivalFixAndTime"], "fixName"),
    "route_arrival_fix_time":           (["ncsmRouteData", "arrivalFixAndTime"], "arrTime"),
    "route_departure_fix":              (["ncsmRouteData", "departureFixAndTime"], "fixName"),
    "next_position_lat":                (["ncsmRouteData", "nextPosition"], "latitudeDecimal"),
    "next_position_lon":                (["ncsmRouteData", "nextPosition"], "longitudeDecimal"),

    # ── Route text and coordination (flightPlanInformation, boundaryCrossingUpdate) ──
    "route_text":                       (["routeOfFlight"], "legacyFormat"),
    "coordination_fix":                 (["coordinationPoint", "namedFix"], None),
    "coordination_time":                (["coordinationTime"], None),
    "coordination_time_type":           (["coordinationTime"], "type"),

    # ── FlightTimes (ncsmFlightTimes): etd/eta sit directly under the container ──
    "times_etd_type":                   (["etd"], "etdType"),
    "times_etd":                        (["etd"], "timeValue"),
    "times_eta_type":                   (["eta"], "etaType"),
    "times_eta":                        (["eta"], "timeValue"),
    "times_rvsm_equipped":              (["rvsmData"], "equipped"),
    "times_arrival_fix":                (["arrivalFixAndTime"], "fixName"),
    "times_arrival_fix_time":           (["arrivalFixAndTime"], "arrTime"),
    "times_departure_fix":              (["departureFixAndTime"], "fixName"),

    # ── Airline / CDM data (FlightModify, FlightCreate) ──────────────────────
    "cdm_etd_type":                     (["airlineData", "etd"], "etdType"),
    "cdm_etd":                          (["airlineData", "etd"], "timeValue"),
    "cdm_eta_type":                     (["airlineData", "eta"], "etaType"),
    "cdm_eta":                          (["airlineData", "eta"], "timeValue"),
    "cdm_diversion":                    (["airlineData", "diversionIndicator"], None),
    "cdm_rvsm_equipped":                (["airlineData", "rvsmData"], "equipped"),
    "cdm_arrival_fix":                  (["airlineData", "arrivalFixAndTime"], "fixName"),
    "airline_out_time":                 (["airlineData", "flightTimeData"], "airlineOutTime"),
    "airline_off_time":                 (["airlineData", "flightTimeData"], "airlineOffTime"),
    "airline_on_time":                  (["airlineData", "flightTimeData"], "airlineOnTime"),
    "airline_in_time":                  (["airlineData", "flightTimeData"], "airlineInTime"),
    "runway_departure_time":            (["airlineData", "flightTimeData"], "runwayDeparture"),
    "runway_arrival_time":              (["airlineData", "flightTimeData"], "runwayArrival"),
    "gate_departure_time":              (["airlineData", "flightTimeData"], "gateDeparture"),
    "gate_arrival_time":                (["airlineData", "flightTimeData"], "gateArrival"),
    "original_departure_time":          (["airlineData", "flightTimeData"], "originalDeparture"),
    "original_arrival_time":            (["airlineData", "flightTimeData"], "originalArrival"),
    "flight_creation_time":             (["airlineData", "flightTimeData"], "flightCreation"),

    # ── departureInformation / arrivalInformation ─────────────────────────────
    "departure_time":                   (["timeOfDeparture"], None),
    "departure_estimated":              (["timeOfDeparture"], "estimated"),
    "arrival_time":                     (["timeOfArrival"], None),
    "arrival_estimated":                (["timeOfArrival"], "estimated"),
    # etd/eta inside ncsmFlightTimeData (both dept and arr messages share this path)
    "fltd_etd_type":                    (["ncsmFlightTimeData", "etd"], "etdType"),
    "fltd_etd":                         (["ncsmFlightTimeData", "etd"], "timeValue"),
    "fltd_eta_type":                    (["ncsmFlightTimeData", "eta"], "etaType"),
    "fltd_eta":                         (["ncsmFlightTimeData", "eta"], "timeValue"),

    # ── boundaryCrossingUpdate ────────────────────────────────────────────────
    "boundary_crossing_time":           (["boundaryPosition"], "boundaryCrossingTime"),
    "boundary_fix":                     (["boundaryPosition", "fixRadialDistance"], None),
    "boundary_fix_radial":              (["boundaryPosition", "fixRadialDistance"], "radial"),
    "boundary_fix_distance":            (["boundaryPosition", "fixRadialDistance"], "distance"),

    # ── flightPlanAmendmentInformation ────────────────────────────────────────
    "amendment_aircraft_specs":         (["amendmentData", "newFlightAircraftSpecs"], None),
    "amendment_equip_qual":             (["amendmentData", "newFlightAircraftSpecs"], "equipmentQualifier"),
    "amendment_route":                  (["amendmentData", "newRouteOfFlight"], "legacyFormat"),
    "amendment_altitude":               (["amendmentData", "newAltitude", "assignedAltitude", "simpleAltitude"], None),
    "amendment_airspeed":               (["amendmentData", "newSpeed", "filedTrueAirSpeed"], None),
    "amendment_coord_fix":              (["amendmentData", "newCoordinationPoint", "fixRadialDistance"], None),
    "amendment_coord_radial":           (["amendmentData", "newCoordinationPoint", "fixRadialDistance"], "radial"),
    "amendment_coord_distance":         (["amendmentData", "newCoordinationPoint", "fixRadialDistance"], "distance"),
    "amendment_coord_time":             (["amendmentData", "newCoordinationTime"], None),
    "amendment_coord_time_type":        (["amendmentData", "newCoordinationTime"], "type"),
}


def _local(tag: str) -> str:
    """Strip XML namespace from a tag, e.g. '{http://...}Foo' -> 'Foo'."""
    return tag.split("}")[-1] if "}" in tag else tag


_XSI_TYPE = "{http://www.w3.org/2001/XMLSchema-instance}type"


def _flatten_element(el: etree._Element, prefix: str, result: dict) -> None:
    """Recursively flatten an lxml element into a flat dict with dot-notation keys.

    Differences from the SFDPS version:
    - Does NOT skip plain `type` attributes (TFMS uses `type` as a data attribute,
      e.g. `<coordinationTime type="PROPOSED">`).
    - Still skips `xsi:type` by checking the fully-qualified name.
    """
    for attr_name, attr_val in el.attrib.items():
        if attr_name == _XSI_TYPE:
            continue
        local_attr = _local(attr_name)
        key = f"{prefix}.{local_attr}" if prefix else local_attr
        result[key] = attr_val

    text = (el.text or "").strip()
    if text and prefix:
        result[prefix] = text

    children_by_tag: dict[str, list] = {}
    for child in el:
        children_by_tag.setdefault(_local(child.tag), []).append(child)

    for tag, children in children_by_tag.items():
        if len(children) == 1:
            child_prefix = f"{prefix}.{tag}" if prefix else tag
            _flatten_element(children[0], child_prefix, result)
        else:
            for i, child in enumerate(children):
                child_prefix = f"{prefix}.{tag}[{i}]" if prefix else f"{tag}[{i}]"
                _flatten_element(child, child_prefix, result)


def parse_tfms_document(doc: str | bytes) -> list[dict]:
    """Parse a single TFMS XML document into a list of flat dicts (one per message).

    A single document may contain multiple fltdMessage elements (inside fltdOutput)
    or one fiMessage element (inside fiOutput).

    For fltdMessage: the payload container element (e.g. trackInformation) is
    stripped — its children are flattened without that container as a prefix, so
    qualifiedAircraftId.gufi etc. are consistent across all message types.

    For fiMessage: child elements are flattened with their tag name as prefix.
    """
    raw = doc.encode("utf-8") if isinstance(doc, str) else doc
    try:
        root = etree.fromstring(raw)
    except etree.XMLSyntaxError:
        return []

    rows = []
    for msg_el in root.iter():
        lname = _local(msg_el.tag)
        if lname not in ("fltdMessage", "fiMessage"):
            continue

        row: dict = {}
        # Message-level attributes (acid, msgType, sourceTimeStamp, etc.)
        for attr_name, attr_val in msg_el.attrib.items():
            if attr_name == _XSI_TYPE:
                continue
            row[_local(attr_name)] = attr_val

        if lname == "fltdMessage":
            # Locate the one payload container and flatten its content without prefix.
            for child in msg_el:
                if _local(child.tag) in _FLTD_MSG_CONTAINERS:
                    _flatten_element(child, "", row)
                    break
        else:
            # fiMessage: flatten each child with its tag as prefix.
            for child in msg_el:
                _flatten_element(child, _local(child.tag), row)

        row = {
            k: v for k, v in row.items()
            if not k.endswith(_SUPPRESS_SUFFIXES)
            and not any(k.startswith(p) for p in _SUPPRESS_PREFIXES)
        }

        if row:
            rows.append(row)

    return rows


def _schema_to_flat_key(attr_path: list[str], key: str | None) -> str:
    """Convert a schema (attr_path, key) entry to its dot-notation flat key."""
    parts = list(attr_path) + ([key] if key is not None else [])
    return ".".join(parts)


def remap_to_schema(
    df: pl.DataFrame,
    schema: dict[str, tuple[list[str], str | None]] = TFMS_FLTD_SCHEMA,
) -> pl.DataFrame:
    """Rename dynamically-discovered flat columns to human-readable schema names.

    Columns not matched by the schema are kept as-is.
    Prints a coverage summary (matched vs missing schema fields).
    """
    rename: dict[str, str] = {}
    for field_name, (attr_path, key) in schema.items():
        flat_key = _schema_to_flat_key(attr_path, key)
        if flat_key in df.columns:
            rename[flat_key] = field_name

    df = df.rename(rename)

    matched = [f for f in schema if f in df.columns]
    missing = [f for f in schema if f not in df.columns]
    print(f"Schema coverage: {len(matched)}/{len(schema)} fields matched")
    if missing:
        print(f"  Missing: {missing}")

    known = set(schema.keys())
    extra_multicol = [
        col for col in df.columns
        if col not in known and df[col].null_count() < len(df) - 1
    ]
    if extra_multicol:
        print(f"  Extra fields (not in schema, seen in >1 row):")

    return df


def split_xml_documents(text: str) -> list[str]:
    """Split concatenated XML documents from a TFMS log file."""
    docs, current = [], []
    for line in text.splitlines():
        if line.startswith("<?xml") and current:
            docs.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        docs.append("\n".join(current))
    return docs


def read_tfms_log(
    file_path,
    max_messages: int | None = None,
    apply_schema: bool = True,
) -> pl.DataFrame:
    """Read a TFMS messages log file into a Polars DataFrame.

    Each fltdMessage and fiMessage in the log becomes one row.  Fields are
    discovered dynamically — every element and attribute found in any message
    becomes a column; messages that lack a field get None.

    When apply_schema=True (default), columns are renamed to TFMS_FLTD_SCHEMA
    names where matched.

    Args:
        file_path:     Path to the .log file.
        max_messages:  Limit number of XML *documents* parsed (None = all).
        apply_schema:  Rename columns to schema names where matched.
    """
    file_path = Path(file_path)
    with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
        text = f.read()

    docs = split_xml_documents(text)
    if max_messages is not None:
        docs = docs[:max_messages]

    rows: list[dict] = []
    skipped_parse = 0

    for doc in docs:
        try:
            new_rows = parse_tfms_document(doc)
            rows.extend(new_rows)
        except Exception:
            skipped_parse += 1

    df = pl.DataFrame(rows, infer_schema_length=min(10_000, len(rows)) if rows else 0)
    print(f"Parsed {len(rows)} messages from {file_path}")
    if skipped_parse:
        print(f"  Skipped {skipped_parse} documents (parse errors)")

    if apply_schema:
        df = remap_to_schema(df)

    return df


RAW_DIR = OUTPUT_DIR / "raw"
OUTPUT_PARQUET_DIR = OUTPUT_DIR / "intermediate"


def process_day(
    d: "date",
    raw_dir: Path = RAW_DIR,
    output_dir: Path = OUTPUT_PARQUET_DIR,
) -> None:
    prefix = f"tfms-logs/year={d.year}/month={d.month:02d}/day={d.day:02d}"
    day_dir = raw_dir / prefix
    if not day_dir.exists():
        print(f"[{d}] Directory not found, skipping: {day_dir}")
        return

    log_files = sorted(p for p in day_dir.iterdir() if p.is_file() and p.name.endswith(".log.gz"))
    if not log_files:
        print(f"[{d}] No .log.gz files found in {day_dir}, skipping.")
        return

    print(f"[{d}] Processing {len(log_files)} file(s)...")
    out_path = output_dir / prefix / f"tfms-logs_{d.year}_{d.month:02d}_{d.day:02d}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    part_paths: list[Path] = []
    for path in log_files:
        df = read_tfms_log(path)
        if len(df) == 0:
            continue
        part_path = out_path.with_suffix(f".{path.name}.part.parquet")
        df.write_parquet(part_path)
        part_paths.append(part_path)
        del df

    if not part_paths:
        print(f"[{d}] No rows parsed, skipping output.")
        return

    combined = pl.scan_parquet(part_paths, extra_columns="ignore", missing_columns="insert")
    if "source_timestamp" in combined.collect_schema().names():
        combined = combined.sort("source_timestamp")
    combined.sink_parquet(out_path)

    for p in part_paths:
        p.unlink()

    result = pl.read_parquet(out_path)
    print(f"[{d}] Wrote {len(result)} rows → {out_path}  ({result.estimated_size('mb'):.1f} MB)")


if __name__ == "__main__":
    import argparse
    from datetime import date, timedelta

    parser = argparse.ArgumentParser(description="Process TFMS logs day by day into parquet.")
    parser.add_argument("start_date", type=date.fromisoformat, help="Start date inclusive (YYYY-MM-DD).")
    parser.add_argument("end_date", type=date.fromisoformat, help="End date exclusive (YYYY-MM-DD).")
    args = parser.parse_args()

    if args.end_date <= args.start_date:
        parser.error("end_date must be after start_date.")

    current = args.start_date
    while current < args.end_date:
        process_day(current)
        current += timedelta(days=1)

