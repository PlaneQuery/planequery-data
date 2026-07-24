from lxml import etree
from pathlib import Path
import os
from data_engineering.utils import OUTPUT_DIR
OUTPUT_DIR = OUTPUT_DIR / "data" / "raw" / "xml-schemas"
FIXM_3_PATH = OUTPUT_DIR / "FIXM_Core_v3_0_full_release"
FIXM_NAS_3_PATH = OUTPUT_DIR / "FIXM_US_Extension_v3_0_full_archive"


def load_nas_schema(
    fixm_nas_path: Path = FIXM_NAS_3_PATH,
    fixm_core_path: Path = FIXM_3_PATH,
) -> etree.XMLSchema:
    # Step 1: The NAS XSDs reference ../../core/ relative to their location.
    # Symlink FIXM Core into the NAS extension's schemas/ dir so relative imports resolve.
    core_source = fixm_core_path / "schemas" / "core"
    core_link = fixm_nas_path / "schemas" / "core"

    if not core_link.exists():
        os.symlink(str(core_source), str(core_link))
        print(f"Created symlink: {core_link} -> {core_source}")
    else:
        print(f"Symlink already exists: {core_link}")

    # Step 2: Load Nas.xsd — the master schema that includes ALL NAS + Core types.
    # This is needed (not NasMessage.xsd alone) because the XML uses xsi:type polymorphism
    # e.g. xsi:type="ns5:NasFlightType" extends the base FlightType with NAS-specific elements.
    nas_xsd = fixm_nas_path / "schemas" / "extensions" / "nas" / "Nas.xsd"
    with open(nas_xsd, 'rb') as f:
        schema_doc = etree.parse(f)

    xml_schema = etree.XMLSchema(schema_doc)
    print(f"Schema loaded: {nas_xsd.name}")
    print(f"  Includes: NasMessage, NasFlightData, NasAircraft, NasArrival, NasDeparture,")
    print(f"            NasEnRouteData, NasRoute, NasPosition, NasStatus, etc.")
    return xml_schema
