from pathlib import Path
from lxml import etree
import polars as pl
from data_engineering.swim.load_xml_schema import load_nas_schema
import gzip
import argparse
from datetime import date, timedelta
from data_engineering.swim.swim_logs_download import download_swim_logs

from data_engineering.utils import OUTPUT_DIR
OUTPUT_DIR = OUTPUT_DIR / "data"

# Mapping from human-readable field name → (element path, attribute key).
# Key=None means text content of the final element; key=str means an XML attribute.
SFDPS_SCHEMA: dict[str, tuple[list[str], str | None]] = {
    # ── Core identifiers ──
    "timestamp":                        ([], "timestamp"),
    "gufi":                             (["gufi"], None),
    "aircraft_identification":          (["flightIdentification"], "aircraftIdentification"),
    "computer_id":                      (["flightIdentification"], "computerId"),
    "site_specific_plan_id":            (["flightIdentification"], "siteSpecificPlanId"),
    "flight_status":                    (["flightStatus"], "fdpsFlightStatus"),
    "flight_plan_id":                   (["flightPlan"], "identifier"),
    "flight_plan_remarks":              (["flightPlan"], "flightPlanRemarks"),
    # ── Flight metadata ──
    "centre":                           ([], "centre"),
    "source":                           ([], "source"),
    "system":                           ([], "system"),
    "flight_type":                      ([], "flightType"),
    "operator":                         (["operator", "operatingOrganization", "organization"], "name"),
    # ── Departure ──
    "departure_point":                  (["departure"], "departurePoint"),
    "departure_actual_time":            (["departure", "runwayPositionAndTime", "runwayTime", "actual"], "time"),
    "departure_estimated_time":         (["departure", "runwayPositionAndTime", "runwayTime", "estimated"], "time"),
    # ── Arrival ──
    "arrival_point":                    (["arrival"], "arrivalPoint"),
    "arrival_actual_time":              (["arrival", "runwayPositionAndTime", "runwayTime", "actual"], "time"),
    "arrival_estimated_time":           (["arrival", "runwayPositionAndTime", "runwayTime", "estimated"], "time"),
    # ── Assigned altitude ──
    "assigned_altitude_ft":             (["assignedAltitude", "simple"], None),
    "assigned_altitude_vfr_plus_ft":    (["assignedAltitude", "vfrPlus"], None),
    # ── Position (enRoute) ──
    "position_latlon":                  (["enRoute", "position", "position", "location", "pos"], None),
    "position_speed_knots":             (["enRoute", "position", "actualSpeed", "surveillance"], None),
    "position_altitude_ft":             (["enRoute", "position", "altitude"], None),
    "position_time":                    (["enRoute", "position"], "positionTime"),
    "position_target_altitude_ft":      (["enRoute", "position", "targetAltitude"], None),
    "position_report_source":           (["enRoute", "position"], "reportSource"),
    "position_coast_indicator":         (["enRoute", "position"], "coastIndicator"),
    # ── Controlling unit ──
    "controlling_unit":                 (["controllingUnit"], "unitIdentifier"),
    "controlling_sector":               (["controllingUnit"], "sectorIdentifier"),
    # ── Aircraft description ──
    "aircraft_type":                    (["aircraftDescription", "aircraftType", "icaoModelIdentifier"], None),
    "aircraft_address":                 (["aircraftDescription"], "aircraftAddress"),
    "aircraft_registration":            (["aircraftDescription"], "registration"),
    "wake_turbulence":                  (["aircraftDescription"], "wakeTurbulence"),
    # ── Interim / requested ──
    "interim_altitude_ft":              (["interimAltitude"], None),
    "requested_airspeed":               (["requestedAirspeed", "nasAirspeed"], None),
    # ── Beacon code ──
    "beacon_code":                              (["enRoute", "beaconCodeAssignment", "currentBeaconCode"], None),
    "reassigned_beacon_code":                   (["enRoute", "beaconCodeAssignment", "reassignedBeaconCode"], None),
    "previous_beacon_code":                     (["enRoute", "beaconCodeAssignment", "previousBeaconCode"], None),
    # ── Coordination ──
    "coordination_time":                        (["coordination"], "coordinationTime"),
    "coordination_time_handling":               (["coordination"], "coordinationTimeHandling"),
    "coordination_fix":                         (["coordination", "coordinationFix"], "fix"),
    # ── Route ──
    "route_text":                               (["agreed", "route"], "nasRouteText"),
    "initial_flight_rules":                     (["agreed", "route"], "initialFlightRules"),
    "local_intended_route":                     (["agreed", "route"], "localIntendedRoute"),
    "atc_intended_route":                       (["agreed", "route"], "atcIntendedRoute"),
    # ── enRoute extras ──
    "position_target_time":                     (["enRoute", "position"], "targetPositionTime"),
    "clearance_speed":                          (["enRoute", "cleared"], "clearanceSpeed"),
    "clearance_text":                           (["enRoute", "cleared"], "clearanceText"),
    "clearance_heading":                        (["enRoute", "cleared"], "clearanceHeading"),
    "handoff_event":                            (["enRoute", "boundaryCrossings", "handoff"], "event"),
    "handoff_receiving_unit":                   (["enRoute", "boundaryCrossings", "handoff", "receivingUnit"], "unitIdentifier"),
    "handoff_receiving_sector":                 (["enRoute", "boundaryCrossings", "handoff", "receivingUnit"], "sectorIdentifier"),
    "handoff_transferring_unit":                (["enRoute", "boundaryCrossings", "handoff", "transferringUnit"], "unitIdentifier"),
    "handoff_transferring_sector":              (["enRoute", "boundaryCrossings", "handoff", "transferringUnit"], "sectorIdentifier"),
    "handoff_accepting_unit":                   (["enRoute", "boundaryCrossings", "handoff", "acceptingUnit"], "unitIdentifier"),
    "handoff_accepting_sector":                 (["enRoute", "boundaryCrossings", "handoff", "acceptingUnit"], "sectorIdentifier"),
    "pointout_originating_unit":                (["enRoute", "pointout", "originatingUnit"], "unitIdentifier"),
    "pointout_originating_sector":              (["enRoute", "pointout", "originatingUnit"], "sectorIdentifier"),
    # ── Aircraft extras ──
    "equipment_qualifier":                      (["aircraftDescription"], "equipmentQualifier"),
    "standard_capabilities":                    (["aircraftDescription", "capabilities"], "standardCapabilities"),
    "aircraft_performance":                     (["aircraftDescription"], "aircraftPerformance"),
    "aircraft_quantity":                        (["aircraftDescription"], "aircraftQuantity"),
    "tfms_special_aircraft_qualifier":          (["aircraftDescription"], "tfmsSpecialAircraftQualifier"),
    # ── Requested altitude ──
    "requested_altitude_ft":                    (["requestedAltitude", "simple"], None),
    "requested_altitude_vfr_plus_ft":           (["requestedAltitude", "vfrPlus"], None),
    # ── Arrival / departure extras ──
    "arrival_alternate_code":                   (["arrival", "arrivalAerodromeAlternate"], "code"),
    "departure_alternate_code":                 (["departure", "takeoffAlternateAerodrome"], "code"),
    # ── Misc ──
    "special_handling":                         ([], "specialHandling"),
    "originator_aftn":                          (["originator"], "aftnAddress"),
    "airborne_hold":                            (["flightStatus"], "airborneHold"),
}

SUPPLEMENTAL_KEYS = [
    "MSG_SEQ_NO",
    "FDPS_GUFI",
    "FLIGHT_PLAN_SEQ_NO",
    "SOURCE_TIME_AND_SEQ",
    "SOURCE_TIME",
    # ADS-B position reports
    "ADSB_POS_174A",
    "ADSB_ALT_175A",
    "ADSB_VEL_176A",
    "ADSB_TIME_177A",
    "ADSB_02M_52B",
    # Other observed supplemental keys
    "FLIGHT_PLAN_REV_NO",
    "4TH_ADAPTED_FIELD",
    "TMI_IDS",
]

# Column suffixes/prefixes to discard at parse time — XML metadata noise.
# .uom / .srsName are unit-of-measure and coordinate-system annotations.
# supplementalData.* is already hoisted to sup_* columns.
# gufi.codeSpace is XML namespace boilerplate.
_SUPPRESS_SUFFIXES = (".uom", ".srsName", ".nil")
_SUPPRESS_PREFIXES = ("supplementalData.", "gufi.codeSpace")


def _local(tag: str) -> str:
    """Strip XML namespace from a tag, e.g. '{http://...}Foo' -> 'Foo'."""
    return tag.split("}")[-1] if "}" in tag else tag


def _flatten_element(el: etree._Element, prefix: str, result: dict) -> None:
    """Recursively flatten an lxml element into a flat dict with dot-notation keys."""
    # Attributes (skip xsi:type — already used for element selection)
    for attr_name, attr_val in el.attrib.items():
        local_attr = _local(attr_name)
        if local_attr == "type":
            continue
        key = f"{prefix}.{local_attr}" if prefix else local_attr
        result[key] = attr_val

    # Text content of leaf nodes
    text = (el.text or "").strip()
    if text and prefix:
        result[prefix] = text

    # Children — index repeated siblings of the same tag
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


XSI_TYPE = "{http://www.w3.org/2001/XMLSchema-instance}type"


def _find_nas_flight(root: etree._Element) -> etree._Element | None:
    for el in root.iter():
        xsi_type = el.get(XSI_TYPE)
        if xsi_type is not None and "NasFlightType" in xsi_type:
            return el
    return None


def parse_sfdps_message(root: etree._Element) -> dict | None:
    """
    Parse a single SFDPS XML document into a flat dict by dynamically
    discovering all elements and attributes present in the NasFlightType node.
    Supplemental nameValue pairs are lifted to top-level sup_* columns.
    Missing fields across messages will be None in the resulting DataFrame.
    """
    flight = _find_nas_flight(root)
    if flight is None:
        return None

    row: dict = {}
    _flatten_element(flight, "", row)

    # Lift supplementalData nameValue pairs to top-level sup_* columns
    for nv_el in flight.iter():
        if _local(nv_el.tag) == "nameValue":
            name = nv_el.get("name")
            if name:
                row[f"sup_{name.lower()}"] = nv_el.get("value")

    # Drop XML metadata noise (uom, srsName, nil, raw supplementalData, gufi namespace)
    row = {
        k: v for k, v in row.items()
        if not k.endswith(_SUPPRESS_SUFFIXES)
        and not any(k.startswith(p) for p in _SUPPRESS_PREFIXES)
    }

    return row



def _schema_to_flat_key(attr_path: list[str], key: str | None) -> str:
    """Convert a schema (attr_path, key) entry to its dot-notation flat key."""
    parts = list(attr_path) + ([key] if key is not None else [])
    return ".".join(parts)


def remap_to_schema(
    df: pl.DataFrame,
    schema: dict[str, tuple[list[str], str | None]] = SFDPS_SCHEMA,
    supplemental_keys: list[str] = SUPPLEMENTAL_KEYS,
) -> pl.DataFrame:
    """
    Rename dynamically-discovered flat columns to the human-readable names
    defined in SFDPS_SCHEMA where a match exists.  Unmatched columns are kept.
    """
    # Build flat_key → schema_name rename map
    rename: dict[str, str] = {}
    for field_name, (attr_path, key) in schema.items():
        flat_key = _schema_to_flat_key(attr_path, key)
        if flat_key in df.columns:
            rename[flat_key] = field_name

    # Supplemental keys: sup_MSG_SEQ_NO stored as sup_msg_seq_no already
    for sup_key in supplemental_keys:
        flat_key = f"sup_{sup_key.lower()}"
        if flat_key in df.columns:
            rename[flat_key] = flat_key  # already well-named, no rename needed

    df = df.rename(rename)

    return df


def split_xml_documents(text: str) -> list[str]:
    """Split concatenated XML documents from a log file."""
    docs, current = [], []
    for line in text.splitlines():
        if line.startswith("<?xml") and current:
            docs.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        docs.append("\n".join(current))
    return docs


def read_sfdps_log(
    file_path,
    xml_schema: etree.XMLSchema,
) -> pl.DataFrame:
    """
    Read an SFDPS messages log file into a Polars DataFrame.

    Fields are discovered dynamically from the XML — every element and attribute
    found in any message becomes a column; messages that lack a field get None.
    Columns are renamed to match SFDPS_SCHEMA where possible.

    Args:
        file_path: path to the .log file
        xml_schema: etree.XMLSchema returned by load_nas_schema()
    """
    file_path = Path(file_path)
    with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
        text = f.read()
    docs = split_xml_documents(text)
    
    rows = []
    skipped_validation = 0
    
    for doc in docs:
        try:
            root = etree.fromstring(doc.encode("utf-8"))
        except etree.XMLSyntaxError:
            skipped_validation += 1
            continue

        if not xml_schema.validate(root):
            skipped_validation += 1
            continue

        row = parse_sfdps_message(root)
        if row:
            rows.append(row)
    
    df = remap_to_schema(pl.DataFrame(rows, infer_schema_length=min(10_000, len(rows)) if rows else 0))
    print(f"Parsed {len(rows)} messages from {file_path}")
    if skipped_validation:
        print(f"  Skipped {skipped_validation} (failed XSD validation)")
    return df

RAW_DIR = OUTPUT_DIR / "raw"
OUTPUT_PARQUET_DIR = OUTPUT_DIR / "intermediate"


def process_day(
    d: date,
    raw_dir: Path = RAW_DIR,
    output_dir: Path = OUTPUT_PARQUET_DIR,
) -> None:
    xml_schema = load_nas_schema()
    prefix = f"sfdps-logs/year={d.year}/month={d.month:02d}/day={d.day:02d}"
    day_dir = raw_dir / prefix
    if not day_dir.exists():
        download_swim_logs("sfdps", d, d+timedelta(days=1))

    gz_files = sorted(p for p in day_dir.iterdir() if p.is_file() and p.suffix in (".gz", ".gzip"))
    if not gz_files:
        raise Exception(f"[{d}] No .gz files found in {day_dir}, skipping.")

    print(f"[{d}] Processing {len(gz_files)} file(s)...")
    out_path = output_dir / prefix / f"sfdps-logs_{d.year}_{d.month:02d}_{d.day:02d}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    part_paths: list[Path] = []
    for path in gz_files:
        df = read_sfdps_log(path, xml_schema)
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
    if "timestamp" in combined.collect_schema().names():
        combined = combined.sort("timestamp")
    combined.sink_parquet(out_path)

    for p in part_paths:
        p.unlink()

    print(f"[{d}] Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SFDPS logs day by day into parquet.")
    parser.add_argument("start_date", type=date.fromisoformat, help="Start date inclusive (YYYY-MM-DD).")
    parser.add_argument("end_date", type=date.fromisoformat, help="End date exclusive (YYYY-MM-DD).")
    args = parser.parse_args()

    if args.end_date <= args.start_date:
        parser.error("end_date must be after start_date.")

    current = args.start_date
    while current < args.end_date:
        process_day(current)
        current += timedelta(days=1)

# We leave out quite a few fields that are waypoints mostly and are in form of arrays connecetd to GUFI. Possibly investigate later but probably not important.
