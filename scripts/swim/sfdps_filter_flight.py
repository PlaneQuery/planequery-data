from lxml import etree
from pathlib import Path
import os
import gzip

CALLSIGN = "UPS59"
OUTPUT_DIR = "/Volumes/T2-SSD/planequery/data/raw"
PREFIX = "sfdps-logs/year=2026/month=02/day=01/"
DOWNLOAD_DIR = f"{OUTPUT_DIR}/{PREFIX}"

NS = {
    "nas": "http://www.faa.aero/nas/3.0",
    "base": "http://www.fixm.aero/base/3.0",
    "flight": "http://www.fixm.aero/flight/3.0",
    "foundation": "http://www.fixm.aero/foundation/3.0",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}


def split_xml_documents(text: str) -> list[str]:
    docs = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("<?xml") and current:
            docs.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        docs.append("\n".join(current))
    return docs


def get_callsign(doc: str) -> str | None:
    try:
        root = etree.fromstring(doc.encode("utf-8"))
        results = root.xpath(
            ".//*[@xsi:type and contains(@xsi:type, 'NasFlightType')]"
            "/flight:flightIdentification/@aircraftIdentification",
            namespaces=NS,
        )
        return results[0] if results else None
    except Exception:
        return None


output_path = Path(OUTPUT_DIR) / "filtered" / PREFIX / f"{CALLSIGN}_messages.txt"
os.makedirs(output_path.parent, exist_ok=True)

match_count = 0
dir_path = Path(DOWNLOAD_DIR)
with open(output_path, "w", encoding="utf-8") as out_f:
    for path in sorted(dir_path.iterdir()):
        if path.is_file() and path.suffix == ".gz":
            print(path)
            with gzip.open(path, "rt", encoding="utf-8") as f:
                raw = f.read()
            for doc in split_xml_documents(raw):
                if get_callsign(doc) == CALLSIGN:
                    out_f.write(doc)
                    out_f.write("\n")
                    match_count += 1

print(f"Saved {match_count} messages for {CALLSIGN} → {output_path}")

