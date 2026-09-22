from enum import StrEnum

class FlightSource(StrEnum):
    ALGORITHM = "algorithm"
    ALGORITHM_NN = "algorithm-nn"
    ALGORITHM_NEAREST = "algorithm-nearest"
    OPENSKY = "opensky"
    SFDPS = "sfdps"
    BTS = "bts"
    EUROCONTROL = "eurocontrol"
