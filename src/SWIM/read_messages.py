from lxml import objectify, etree

aircraftIdentifications = []
NS = {
    "nas": "http://www.faa.aero/nas/3.0",
    "base": "http://www.fixm.aero/base/3.0",
    "flight": "http://www.fixm.aero/flight/3.0",
    "foundation": "http://www.fixm.aero/foundation/3.0",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}
def parse_message(doc):
    root = objectify.fromstring(doc.encode("utf-8"))

    # You still typically access by navigation/XPath:
    gufi = root.xpath("string(.//*[local-name()='gufi'])")
    flight = root.xpath(
        ".//*[@xsi:type and contains(@xsi:type, 'NasFlightType')]",
        namespaces=NS
    )[0]
    aircraftIdentification = flight.flightIdentification.get("aircraftIdentification")
    aircraftIdentifications.append(aircraftIdentification)
    print(aircraftIdentification)