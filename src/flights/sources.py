from flights.algorithm import AlgorithmNearestFlightSource
from flights.airport_nn import AlgorithmNNFlightSource
from flights.providers.bts import BTSFlightSource
from flights.providers.eurocontrol import EurocontrolFlightSource
from flights.providers.mike import MikeFlightSource
from flights.providers.sfdps import SFDPSFlightSource


ALGORITHM_NEAREST_SOURCE = AlgorithmNearestFlightSource()
ALGORITHM_NN_SOURCE = AlgorithmNNFlightSource()
BTS_SOURCE = BTSFlightSource()
EUROCONTROL_SOURCE = EurocontrolFlightSource()
MIKE_SOURCE = MikeFlightSource()
SFDPS_SOURCE = SFDPSFlightSource()

FLIGHT_SOURCES = (
    ALGORITHM_NEAREST_SOURCE,
    ALGORITHM_NN_SOURCE,
    BTS_SOURCE,
    EUROCONTROL_SOURCE,
    MIKE_SOURCE,
    SFDPS_SOURCE,
)
FLIGHT_SOURCES_BY_NAME = {source.name: source for source in FLIGHT_SOURCES}
assert len(FLIGHT_SOURCES_BY_NAME) == len(FLIGHT_SOURCES)
