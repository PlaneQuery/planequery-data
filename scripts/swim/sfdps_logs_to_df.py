from lxml import objectify
import pandas as pd
from pathlib import Path
import os
import gzip

aircraftIdentifications = []
NS = {
    "nas": "http://www.faa.aero/nas/3.0",
    "base": "http://www.fixm.aero/base/3.0",
    "flight": "http://www.fixm.aero/flight/3.0",
    "foundation": "http://www.fixm.aero/foundation/3.0",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}

def safe_get(obj, attr_path, key=None, default=None):
    """
    Safely traverse nested attributes and call .get(key) at the end if key is provided.
    attr_path: list of attribute names to traverse.
    key: key to pass to .get() at the final attribute.
    default: value to return if any attribute is missing.
    """
    try:
        for attr in attr_path:
            obj = getattr(obj, attr, None)
            if obj is None:
                return default
        if key is not None and hasattr(obj, "get"):
            return obj.get(key)
        return obj
    except Exception:
        return default

def parse_message(doc):
    
    root = objectify.fromstring(doc.encode("utf-8")) # TODO: NEED TO DEBUG THIS

    # You still typically access by navigation/XPath:
    flight = root.xpath(
        ".//*[@xsi:type and contains(@xsi:type, 'NasFlightType')]",
        namespaces=NS
    )[0]
    timestamp = safe_get(flight, [], "timestamp")
    aircraftIdentification = safe_get(flight, ["flightIdentification"], "aircraftIdentification")
    flightStatus = safe_get(flight, ["flightStatus"], "fdpsFlightStatus")
    # --- ADD GUFI (UUID) ---
    # <gufi codeSpace="urn:uuid">...</gufi>
    gufi = safe_get(flight, ["gufi"])
    if gufi is not None:
        # objectify element -> string value
        gufi = str(gufi)
    # ------------------------
    departure = safe_get(flight, ["departure"])
    departurePoint = safe_get(departure, [], "departurePoint") if departure is not None else None
    depature_time = safe_get(departure, ["runwayPositionAndTime", "runwayTime", "actual"], "time")
    estimated_depature_time = safe_get(departure, ["runwayPositionAndTime", "runwayTime", "estimated"], "time")

    arrival = safe_get(flight, ["arrival"])
    arrivalPoint = safe_get(arrival, [], "arrivalPoint") if arrival is not None else None
    arrival_time = safe_get(arrival, ["runwayPositionAndTime", "runwayTime", "actual"], "time")
    estimated_arrival_time = safe_get(arrival, ["runwayPositionAndTime", "runwayTime", "estimated"], "time")

    # arrival_time = arrival.runwayPositionAndTime.runwayTime.actual.get("time")
    return (timestamp, aircraftIdentification, flightStatus, gufi, departurePoint, depature_time, estimated_depature_time, arrivalPoint, arrival_time, estimated_arrival_time)

def split_xml_documents(text: str):
    docs = []
    current = []

    for line in text.splitlines():
        if line.startswith("<?xml") and current:
            docs.append("\n".join(current))
            current = []
        current.append(line)

    if current:
        docs.append("\n".join(current))

    return docs

sfdps_rows = []
def process_messages_file(file_path: str):
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        raw = f.read()
        docs = split_xml_documents(raw)
        for doc in docs:
            sfdps_row = parse_message(doc)
            sfdps_rows.append(sfdps_row)

OUTPUT_DIR = "/Volumes/T2-SSD/planequery/data/raw"
BUCKET = "planequery-swim-logs-prod"
PREFIX = "sfdps-logs/year=2026/month=02/day=01/"
DOWNLOAD_DIR = f"{OUTPUT_DIR}/{PREFIX}"
dir_path = Path(DOWNLOAD_DIR)
count = 0
for path in dir_path.iterdir():
    count+=1
    if path.is_file() and path.suffix == '.gz':
        print(path)
        process_messages_file(str(path))
#TODO: Either sort files first so in order or sort dataframe timestamp. 
df = pd.DataFrame(
    sfdps_rows,
    columns=[
        "timestamp",
        "aircraftIdentification",
        "flightStatus",
        "gufi",
        "departurePoint",
        "depature_time",
        "estimated_depature_time",
        "arrivalPoint",
        "arrival_time",
        "estimated_arrival_time"
    ]
)
pickle_output_dir = Path("/Volumes/T2-SSD/planequery/data/raw/pickle") / Path(PREFIX) / "output.pkl"
os.makedirs(pickle_output_dir.parent, exist_ok=True)
df.to_pickle(pickle_output_dir, compression="zstd")
