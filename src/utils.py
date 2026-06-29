import math
import subprocess

def haversine(lat1, lon1, lat2, lon2):
    '''Returned distance in km'''
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(delta_lambda / 2.0) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def current_commit_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
