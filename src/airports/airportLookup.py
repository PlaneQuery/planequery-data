import csv
import logging
import numpy as np
import math
from dataclasses import dataclass
from sklearn.neighbors import BallTree
from scipy.spatial import KDTree
from enum import StrEnum
from pathlib import Path
import os
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
Airport_Size_Ranking = { # TODO: move this inside class but python is being annoying with thinking this should be another enum value when I try to make it a member variable
    "SMALL_AIRPORT": 1,
    "MEDIUM_AIRPORT": 2,
    "LARGE_AIRPORT": 3
}

# "small_airport, medium_airport, large_airport, seaplane_base, heliport, closed, balloonport"
class Airport_Types(StrEnum):
    SMALL_AIRPORT = "small_airport"
    MEDIUM_AIRPORT = "medium_airport"
    LARGE_AIRPORT = "large_airport"
    
    # Define custom ordering using a ranking

    def __lt__(self, other):
        if isinstance(other, Airport_Types):
            return Airport_Size_Ranking[self.name] < Airport_Size_Ranking[other.name]
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Airport_Types):
            return Airport_Size_Ranking[self.name] <= Airport_Size_Ranking[other.name]
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Airport_Types):
            return Airport_Size_Ranking[self.name] > Airport_Size_Ranking[other.name]
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Airport_Types):
            return Airport_Size_Ranking[self.name] >= Airport_Size_Ranking[other.name]
        return NotImplemented
@dataclass
class Airport:
    ident: str
    iata: str
    coordinates: tuple[float, float]
    elevation_ft: int
    type: Airport_Types
    distance_from_plane_point_in_km: float = 0.0

class AirportLookup:
    _instance = None

    def __new__(cls):  # Singleton
        if cls._instance is None:
            cls._instance = super(AirportLookup, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.load_airports_from_csv()

    def load_airports_from_csv(self):
        self._coordinates = []
        self._coordinates_rad = []
        self._airports: list[Airport] = []
        self.iata_index: dict[str, Airport] = {} # TODO: Make this readonly.
        self.iata_to_ident_index: dict[str, str] = {}
        self.ident_index: dict[str, Airport] = {}
        with open(OUTPUT_DIR / 'data' / 'interim' / 'airports.csv', 'r', encoding='utf-8') as airport_csv:
            airport_csv_reader = csv.DictReader(filter(lambda row: row[0] != '#', airport_csv))
            for airport in airport_csv_reader:
                if airport['type'] in Airport_Types:
                    latitude_deg = float(airport['latitude_deg'])
                    longitude_deg = float(airport['longitude_deg'])
                    airport_coord_deg = (latitude_deg, longitude_deg)
                    airport_coord_rad = np.radians(airport_coord_deg)
                    ident: str = airport['ident'] # TODO: rename this to ident which is correct name. 
                    elevation_ft = airport["elevation_ft"]
                    elevation_ft = int(elevation_ft) if elevation_ft else 0
                    iata: str = airport['iata_code'] # can be '' empty
                    add_airport = Airport(
                        ident=ident,
                        iata=iata,
                        coordinates=airport_coord_deg,
                        elevation_ft=elevation_ft,
                        type = Airport_Types(airport['type'])
                    )
                    self._airports.append(add_airport)
                    self._coordinates.append(airport_coord_deg)
                    self._coordinates_rad.append(airport_coord_rad)
                    if iata:
                        self.iata_index[iata] = add_airport
                        self.iata_to_ident_index[iata] = ident
                    if ident:
                        self.ident_index[ident] = add_airport
            self._ball_tree = BallTree(np.array(self._coordinates_rad), metric='haversine')
            coords_rad = np.array(self._coordinates_rad)  # (N, 2) lat/lon in radians
            xyz = np.column_stack([
                np.cos(coords_rad[:, 0]) * np.cos(coords_rad[:, 1]),
                np.cos(coords_rad[:, 0]) * np.sin(coords_rad[:, 1]),
                np.sin(coords_rad[:, 0]),
            ])
            self._kd_tree = KDTree(xyz)

    def getClosestAirport(self, latitude, longitude) -> Airport:
        query_point_rad = np.radians((latitude, longitude))
        distance, index = self._ball_tree.query([query_point_rad], k=1)
        airport = self._airports[index[0][0]]
        distance_in_km = distance[0][0]  * 6371.0 # Earth's radius in kilometers
        airport.distance_from_plane_point_in_km = float(distance_in_km)
        return airport

    def getAirportsWithinRadius(self, latitude: float, longitude: float, radius_km: float = 15.0) -> list[Airport]:
        if not getattr(self, "_ball_tree", None):
            return []
        query_point_rad = np.radians((latitude, longitude))
        radius_rad = radius_km / 6371.0
        indices, distances = self._ball_tree.query_radius(
            [query_point_rad],
            r=radius_rad,
            return_distance=True,
            sort_results=True,
        )
        results: list[Airport] = []
        for idx, dist in zip(indices[0], distances[0]):
            airport = self._airports[idx]
            airport.distance_from_plane_point_in_km = float(dist * 6371.0)
            results.append(airport)
        return results

    def get_Airport_from_airport_icao(self, airport_icao: str) -> Airport | None:
        return self.ident_index.get(airport_icao)

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface specified in decimal degrees.
    """
    R = 6371.0  # Earth radius in kilometers

    # Convert decimal degrees to radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(delta_lambda / 2.0) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance in kilometers

def flight_time_makes_sense(lat1, lon1, lat2, lon2, takeoff_time, landing_time):
    """
    Determine if the flight time between two locations makes sense for a private jet.
    
    Parameters:
    - lat1, lon1: Latitude and longitude of the departure airport.
    - lat2, lon2: Latitude and longitude of the arrival airport.
    - takeoff_time: Takeoff time in seconds.
    - landing_time: Landing time in seconds.
    
    Returns:
    - True if the flight time makes sense, False otherwise.
    """
    distance_km = haversine(lat1, lon1, lat2, lon2)
    time_seconds = landing_time - takeoff_time

    if time_seconds <= 0:
        return False  # Landing time must be after takeoff time

    time_hours = time_seconds / 3600.0
    required_speed = distance_km / time_hours  # Speed in km/h

    max_private_jet_speed = 950  # Maximum speed in km/h
    min_private_jet_speed = 300  # Minimum cruising speed in km/h

    if required_speed > max_private_jet_speed:
        logging.error("Error create_json_flight: " + f"required_speed > max_private_jet_speed")
        return False  # Required speed exceeds maximum private jet speed
    if required_speed < min_private_jet_speed/5: # Allow for circling. 
        logging.error("Error create_json_flight: " + f"required_speed < min_private_jet_speed")
        return False  # Required speed is too low; flight time seems excessively long
    return True  # Flight time makes 

if __name__ == "__main__":
    airport_lookup = AirportLookup()
    airport = airport_lookup.getClosestAirport(37.62131, -122.37896)  # Coordinates for San Francisco International Airport
    print(f"Closest airport ICAO: {airport.ident}, Type: {airport.type}, Distance: {airport.distance_from_plane_point_in_km:.2f} km")