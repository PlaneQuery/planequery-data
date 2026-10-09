from flights.algorithm import AlgorithmNearestFlightSource
from flights.airport_nn import AlgorithmNNFlightSource
from flights.ground_truth.source import GroundTruthFlightSource
from flights.providers.adsbx import ADSBXPublishedFlightSource
from flights.providers.bts import BTSFlightSource
from flights.providers.eurocontrol import EurocontrolFlightSource
from flights.providers.mike import MikeFlightSource
from flights.providers.opensky import OpenSkyTrinoFlightSource
from flights.providers.sfdps import SFDPSFlightSource
from flights.verified.bts_verified import BTSVerifiedFlightSource


ALGORITHM_NEAREST_SOURCE = AlgorithmNearestFlightSource()
ALGORITHM_NN_SOURCE = AlgorithmNNFlightSource()
ADSBX_SOURCE = ADSBXPublishedFlightSource()
BTS_SOURCE = BTSFlightSource()
EUROCONTROL_SOURCE = EurocontrolFlightSource()
GROUND_TRUTH_SOURCE = GroundTruthFlightSource()
BTS_VERIFIED_SOURCE = BTSVerifiedFlightSource()
MIKE_SOURCE = MikeFlightSource()
OPENSKY_SOURCE = OpenSkyTrinoFlightSource()
SFDPS_SOURCE = SFDPSFlightSource()

FLIGHT_SOURCES = (
    ALGORITHM_NEAREST_SOURCE,
    ALGORITHM_NN_SOURCE,
    ADSBX_SOURCE,
    BTS_SOURCE,
    EUROCONTROL_SOURCE,
    BTS_VERIFIED_SOURCE,
    GROUND_TRUTH_SOURCE,
    MIKE_SOURCE,
    OPENSKY_SOURCE,
    SFDPS_SOURCE,
)
FLIGHT_SOURCES_BY_NAME = {source.name: source for source in FLIGHT_SOURCES}
assert len(FLIGHT_SOURCES_BY_NAME) == len(FLIGHT_SOURCES)
