from lxml import objectify
from collections import defaultdict

aircraftIdentifications = []
per_minute_ids = defaultdict(set)
last_minute = None

NS = {
    "nas": "http://www.faa.aero/nas/3.0",
    "base": "http://www.fixm.aero/base/3.0",
    "flight": "http://www.fixm.aero/flight/3.0",
    "foundation": "http://www.fixm.aero/foundation/3.0",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}

def parse_message(doc):
    global last_minute
    
    root = objectify.fromstring(doc.encode("utf-8")) # TODO: NEED TO DEBUG THIS

    # You still typically access by navigation/XPath:
    flight = root.xpath(
        ".//*[@xsi:type and contains(@xsi:type, 'NasFlightType')]",
        namespaces=NS
    )[0]
    try:
        aircraftIdentification = flight.flightIdentification.get("aircraftIdentification")
        flightStatus = flight.flightStatus.get("fdpsFlightStatus")
        
    except Exception as e:
        print(f"Error: {e}")
        print(doc)
