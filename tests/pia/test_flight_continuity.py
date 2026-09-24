from airports.models import AirportType
from pia.flight_continuity import AIRPORT_SIZE_RANK


def test_airport_size_rank_covers_every_airport_type():
    assert set(AIRPORT_SIZE_RANK) == set(AirportType)
