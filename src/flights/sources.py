from flights.airport_nn import AlgorithmNNFlightSource, OpenSkyNNFlightSource
from flights.algorithm import AlgorithmNearestFlightSource
from flights.ground_truth.source import GroundTruthFlightSource
from flights.sun import SunFlightSource
from flights.openap import OpenAPFlightSource
from flights.opensky_reverse_eng import OpenSkyReverseEngFlightSource
from flights.opensky_reverse_eng_nearest import OpenSkyReverseEngNearestFlightSource
from flights.plane_notify import PlaneNotifyFlightSource
from flights.traffic import TrafficFlightSource
from flights.providers.adsbx import ADSBXPublishedFlightSource
from flights.providers.bts import BTSFlightSource
from flights.providers.eurocontrol import EurocontrolFlightSource
from flights.providers.mike import MikeFlightSource
from flights.providers.opensky import OpenSkyTrinoFlightSource
from flights.providers.sfdps import SFDPSFlightSource
from flights.verified.source import VerifiedFlightSource

# TODO: Probably a much better way of handling this, that makes imports easier etc..
ALGORITHM_NEAREST_SOURCE = AlgorithmNearestFlightSource()
ALGORITHM_NN_SOURCE = AlgorithmNNFlightSource()
ADSBX_SOURCE = ADSBXPublishedFlightSource()
BTS_SOURCE = BTSFlightSource()
EUROCONTROL_SOURCE = EurocontrolFlightSource()
EUROCONTROL_VERIFIED_SOURCE = VerifiedFlightSource(EUROCONTROL_SOURCE)
GROUND_TRUTH_SOURCE = GroundTruthFlightSource()
SUN_SOURCE = SunFlightSource()
BTS_VERIFIED_SOURCE = VerifiedFlightSource(BTS_SOURCE)
MIKE_SOURCE = MikeFlightSource()
OPENAP_SOURCE = OpenAPFlightSource()
OPENSKY_SOURCE = OpenSkyTrinoFlightSource()
OPENSKY_NN_SOURCE = OpenSkyNNFlightSource()
OPENSKY_REVERSE_ENG_SOURCE = OpenSkyReverseEngFlightSource()
OPENSKY_REVERSE_ENG_NEAREST_SOURCE = OpenSkyReverseEngNearestFlightSource()
PLANE_NOTIFY_SOURCE = PlaneNotifyFlightSource()
TRAFFIC_SOURCE = TrafficFlightSource()
SFDPS_SOURCE = SFDPSFlightSource()
SFDPS_VERIFIED_SOURCE = VerifiedFlightSource(SFDPS_SOURCE)

FLIGHT_SOURCES = (
    ALGORITHM_NEAREST_SOURCE,
    ALGORITHM_NN_SOURCE,
    ADSBX_SOURCE,
    BTS_SOURCE,
    EUROCONTROL_SOURCE,
    EUROCONTROL_VERIFIED_SOURCE,
    BTS_VERIFIED_SOURCE,
    GROUND_TRUTH_SOURCE,
    SUN_SOURCE,
    MIKE_SOURCE,
    OPENAP_SOURCE,
    OPENSKY_SOURCE,
    OPENSKY_NN_SOURCE,
    OPENSKY_REVERSE_ENG_SOURCE,
    OPENSKY_REVERSE_ENG_NEAREST_SOURCE,
    PLANE_NOTIFY_SOURCE,
    TRAFFIC_SOURCE,
    SFDPS_SOURCE,
    SFDPS_VERIFIED_SOURCE,
)
FLIGHT_SOURCES_BY_NAME = {source.name: source for source in FLIGHT_SOURCES}
assert len(FLIGHT_SOURCES_BY_NAME) == len(FLIGHT_SOURCES)
