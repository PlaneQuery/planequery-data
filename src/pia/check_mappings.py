from data_engineering.flights.flight_type import Flight, FlightIndex


def flight_overlaps(
    flight_a,
    flight_b,
) -> bool:
    """Return True if the two flights overlap in time."""
    return not (
        flight_a.landing_time <= flight_b.takeoff_time
        or flight_b.landing_time <= flight_a.takeoff_time
    )

def icao_mapping_from_flight_mapping(mapping: dict[Flight, str]) -> dict[str, str]:
    '''pia icao: real icao'''
    return {flight.icao: real_icao for flight, real_icao in mapping.items()}

def real_icao_to_pia_icaos_from_flight_mapping(mapping: dict[Flight, str]) -> dict[str, set[str]]:
    '''real icao: set of pia icaos'''
    real_icao_to_pia_icaos: dict[str, set[str]] = dict()
    for flight, icao in mapping.items():
        if flight.pia:
            real_icao_to_pia_icaos.setdefault(icao, set()).add(flight.icao)
    return real_icao_to_pia_icaos

def check_overlap_in_mappings(
    flight_index: FlightIndex,
    mapping: dict[Flight, str],
) -> set[str]:
    """Find overlapping flights assigned to the same real aircraft."""
    real_icao_to_pia_icaos: dict[str, set[str]] = dict()
    failed_icaos = set()
    for flight, icao in mapping.items():
        if flight.pia:
            real_icao_to_pia_icaos.setdefault(icao, set()).add(flight.icao)

    for real_icao, pia_icaos in real_icao_to_pia_icaos.items():
        icaos = [real_icao] + list(pia_icaos)
        flights = []
        for icao in icaos:
            flights.extend(flight_index.flights_by_icao[icao])
        for i in range(len(flights)):
            flight_a = flights[i]
            other_flights = flights.copy()
            other_flights.pop(i)
            for flight_b in other_flights:
                if flight_overlaps(flight_a, flight_b):
                    print(
                        f"Flights {flight_a.flight_id} and {flight_b.flight_id} "
                        f"overlap in time but are assigned to the same real aircraft "
                        f"{real_icao}"
                    )
                    failed_icaos.add(real_icao)
                    failed_icaos.update(pia_icaos)
    # Logic to populate overlaps goes here.
    return failed_icaos

def check_flights_are_continuous(flight_index: FlightIndex, mapping: dict[Flight, str]):
    failed_icaos: set[str] = set()
    icao_mapping = icao_mapping_from_flight_mapping(mapping)
    # real icao -> pia icao
    real_icao_to_pia_icaos: dict[str, set[str]] = dict()
    for pia_icao, real_icao in icao_mapping.items():
        real_icao_to_pia_icaos.setdefault(real_icao, set()).add(pia_icao)
    for real_icao, pia_icaos in real_icao_to_pia_icaos.items():
        icaos = [real_icao] + list(pia_icaos)
        all_flights = []
        for icao in icaos:
            flights = flight_index.flights_by_icao[icao]
            all_flights.extend(flights)
        all_flights.sort(key=lambda flight: flight.takeoff_time)
        for i in range(len(all_flights) - 1):
            flight_a = all_flights[i]
            flight_b = all_flights[i + 1]
            if flight_a.icao == flight_b.icao:
                continue
            if flight_a.landing_airport_ident != flight_b.takeoff_airport_ident:
                print(
                    f"Flights {flight_a.flight_id} and {flight_b.flight_id} "
                    f"have a discontinuity in airports: "
                    f"{flight_a.landing_airport_ident} -> {flight_b.takeoff_airport_ident} "
                )
                failed_icaos.add(real_icao)
                failed_icaos.update(pia_icaos)
    return failed_icaos

def remove_failed_icaos(real_icao_to_pia_icaos: dict[str, set[str]], failed_icaos: set[str]):
    for failed_icao in failed_icaos:
        if failed_icao in real_icao_to_pia_icaos:
            del real_icao_to_pia_icaos[failed_icao]
        for pia_icaos in real_icao_to_pia_icaos.values():
            if failed_icao in pia_icaos:
                pia_icaos.remove(failed_icao)

# Need to make length of flights and number of transistions. 
def stats_on_mappings(flight_index: FlightIndex, real_icao_to_pia_icaos):
    real_icao_stats = {}
    for real_icao, pia_icaos in real_icao_to_pia_icaos.items():
        icaos = [real_icao] + list(pia_icaos)
        all_flights = []
        for icao in icaos:
            flights = flight_index.flights_by_icao[icao]
            all_flights.extend(flights)
        all_flights.sort(key=lambda flight: flight.takeoff_time)
        num_transitions = 0
        for i in range(len(all_flights) - 1):
            flight_a = all_flights[i]
            flight_b = all_flights[i + 1]
            if flight_a.icao == flight_b.icao:
                continue
            else:
                num_transitions += 1
        real_icao_stats[real_icao] = {
            "num_transitions": num_transitions,
            "num_flights": len(all_flights),
        }
    return real_icao_stats

def check_mappings(
    flight_index: FlightIndex,
    mapping: dict[Flight, str],
):
    failed_icaos = check_overlap_in_mappings(flight_index, mapping)
    failed_icaos.update(check_flights_are_continuous(flight_index, mapping))
    print(failed_icaos)
    return failed_icaos