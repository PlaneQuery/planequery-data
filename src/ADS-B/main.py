from datetime import datetime, timedelta, timezone
import sys
import os
import requests

import time
import logging
import clickhouse_connect
logging.basicConfig(level=logging.DEBUG)
environment = os.getenv('ENVIRONMENT', 'development')

client = clickhouse_connect.get_client(
    host=os.environ['CLICKHOUSE_HOST'],
    username='default',
    password=os.environ['CLICKHOUSE_PASSWORD'],
    secure=True,
    connect_timeout=30,        # TCP/TLS handshake
    send_receive_timeout=120, 
)
class FetchError:
    def __init__(self, url, error):
        self.url = url
        self.error = error

def get_current_utc_time_in_milliseconds() -> int:
    return int(time.time() * 1000)
import math
def is_epoch_utc_milliseconds(x, min_year=2000, max_year=2200):
    """
    Return True if x is a Unix epoch timestamp in *milliseconds* (UTC),
    plausibly between Jan 1, min_year and Jan 1, max_year.
    Accepts int or float. Rejects seconds/us/ns by range.
    """
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return False

    # Convert assuming milliseconds
    try:
        dt = datetime.fromtimestamp(x / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return False

    # Year gate ensures we don't accidentally accept seconds (too small)
    # or micro/nanoseconds (too large)
    return (min_year <= dt.year < max_year)

START_TIME_OF_PROGRAM = get_current_utc_time_in_milliseconds()
START_TIME_OF_PROGRAM_MINUS_DAY= START_TIME_OF_PROGRAM - 24*3600*1000

last_message_time_by_icao: dict[str, int] = dict()

json_output_path = "output/adsb_updating/adsb.lol"
os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
os.makedirs(json_output_path, exist_ok=True)  # Ensure leaf directory exists

def process_adsb_message_stream(save_readsb_to_all_to_json: bool = False, use_opensky: bool = True) -> str:
    start = time.perf_counter()
    now = time.time()
    fallback_until = getattr(process_adsb_message_stream, "_fallback_until", 0.0)
    data_source = "adsblol"
    if now >= fallback_until:
        try:
            print("adsblol")
            adsb_messages = process_adsblol_reapi_states()
            process_adsb_message_stream._fallback_until = 0.0
            data_source = "adsblol"
        except Exception as exc:
            logging.warning("adsblol failed, using opensky for 10 minutes: %s", exc)
            process_adsb_message_stream._fallback_until = now + 10 * 60
            adsb_messages = process_opensky_all_states()
            data_source = "opensky"
    else:
        print("adsblol fallback active, using opensky")
        adsb_messages = process_opensky_all_states()
        data_source = "opensky"

    # process_takeoff_landing_events(adsb_messages)
    print(len(adsb_messages), "new messages to insert")
    # insert_adsb_messages(client, adsb_messages)
    print("Inserted messages")
    end = time.perf_counter()
    print(f"Elapsed process_adsb_message_stream: {_format_duration(end - start)}")
    return data_source

def process_adsblol_reapi_states():
    data = get_adsblol_reapi_request()
    # adsb_messages: list[Adsb_Message] = []

    # Assume API returns {"ac": [...]} as in example
    messages = data["aircraft"]
    now = int(data["now"] * 1000)
    print(len(messages), "messages received from adsb.lol")
    for message in messages:
        if "seen_pos" not in message:
            continue
        seen_pos = int(message["seen_pos"] * 1000)
        message_time = now - seen_pos
        icao: str = message["hex"]
        last_message_time = get_last_message_time(icao)
        if message_time <= last_message_time: # I think broken should be inverse on last_message_time = 0.
                # This message is older or same as the last processed message for this ICAO
                continue
        from my_types import Adsb_Message_reapi
        message_obj = Adsb_Message_reapi.from_json(message)
        print(message_obj)
        # print(message)
        # print(type(message))
        
    # return adsb_messages

def process_pia_planes():
    pass
def get_adsblol_reapi_request():
    url = "https://re-api.adsb.lol?all"
    attempts = 2
    delay = 1.0
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 429:
                logging.error("Rate limited by adsb.lol API on attempt %d/%d", attempt, attempts)
                # get Retry-After:
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    retry_after = int(retry_after)
                    delay = float(retry_after+5)
                else:
                    delay = 60.0
                continue

            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            logging.warning("get_adsblol_reapi_request attempt %d/%d failed: %s", attempt, attempts, e)
            if attempt < attempts:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"Failed to fetch {url} after {attempts} attempts") from last_err

def get_opensky_all_states_request():
    username = os.environ.get('OPENSKY_USERNAME')
    password = os.environ.get('OPENSKY_PASSWORD')
    url = "https://opensky-network.org/api/states/all"

    attempts = 2
    delay = 120.0
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            if username and password:
                resp = requests.get(url, auth=(username, password), timeout=10)
            else:
                resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            logging.warning("get_opensky_all_states_request attempt %d/%d failed: %s", attempt, attempts, e)
            if attempt < attempts:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"Failed to fetch {url} after {attempts} attempts") from last_err

def process_opensky_all_states():
    adsb_messages: list[Adsb_Message] = []
    data = get_opensky_all_states_request()
    # https://openskynetwork.github.io/opensky-api/rest.html#all-state-vectors
    states = data.get("states", [])
    print(len(states), "messages received from OpenSky")
    for state in states:
        # state is a list with fixed indices
        # 0: icao24, 1: callsign, 2: origin_country, 3: time_position, 4: last_contact
        # 5: longitude, 6: latitude, 7: baro_altitude, 8: on_ground
        # 9: velocity, 10: true_track, 11: vertical_rate
        # 12: sensors, 13: geo_altitude, 14: squawk, 15: spi, 16: position_source
        icao24 = state[0]
        callsign = state[1]
        origin_country = state[2]
        time_position = state[3] # seconds since epoch
        last_contact = state[4]
        longitude = state[5]
        latitude = state[6]
        baro_altitude = state[7]
        on_ground = state[8]
        velocity = state[9]
        true_track = state[10]
        vertical_rate = state[11]
        sensors = state[12]
        geo_altitude = state[13]
        squawk = state[14]
        spi = state[15]
        position_source = state[16]
        if time_position is None or latitude is None or longitude is None:
            continue
        last_message_time = get_last_message_time(icao24)
        time_position_ms = int(time_position * 1000)
        assert(is_epoch_utc_milliseconds(time_position_ms))
        if time_position_ms <= last_message_time:
                # This message is older or same as the last processed message for this ICAO
                print("continue here", icao24, time_position_ms)
                continue
        print("insertion", icao24, time_position_ms)
        adsb_message = Adsb_Message(
            time=datetime.fromtimestamp(time_position, tz=timezone.utc),
            icao24=icao24,
            lat=latitude,
            lon=longitude,
            on_ground=on_ground,
            alt_baro=int(baro_altitude) if baro_altitude is not None else None,
            baro_rate=int(vertical_rate) if vertical_rate is not None else None,
            gs=float(velocity) if velocity is not None else None,
            track=float(true_track) if true_track is not None else None,
            alt_geom=int(geo_altitude) if geo_altitude is not None else None,
            geom_rate=None,
            squawk=squawk,
            dbFlags=None,
            category=None
        )
        adsb_messages.append(adsb_message)
        last_message_time_by_icao[icao24] = time_position_ms

    return adsb_messages
def get_last_message_time(icao: str) -> int:
    if icao in last_message_time_by_icao:
        return last_message_time_by_icao[icao]
    last_message_time_by_icao[icao] = 0
    return last_message_time_by_icao[icao]

# We can create dict of the last time that a legs & 2 was there. 
# I could just load all adsb_messages in past 1 hours and get by icao

def get_last_message_time_for_icaos():
    start_dt = datetime.now() - timedelta(hours=1)
    start_dt_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    query = f'''
        Select icao, MAX(time) as last_time from adsb_messages_realtime Where time >= '{start_dt_str}' GROUP BY icao
    '''
    df = client.query_df(query)
    for _, row in df.iterrows():
        icao = row['icao']
        last_time = row['last_time']
        last_timestamp_ms = int(last_time.timestamp() * 1000)
        last_message_time_by_icao[icao] = last_timestamp_ms

def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes):02d}:{sec:05.2f} (mm:ss)"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}:{int(minutes):02d}:{int(sec):02d} (hh:mm:ss)"

if __name__ == '__main__':
    # python -m src.adsb_updating.main
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    print(f"Starting adsb_updating in {environment} environment")
    logging.info("Logging at Info level")
    # get_last_message_time_for_icaos()
    interval = 60*2  # 2 minute
    while True:
        loop_start = time.perf_counter()
        try:
            data_source = process_adsb_message_stream()
        except Exception as e:
            logging.error("Error in process_adsb_message_stream: %s", e)
            client = clickhouse_connect.get_client(
            host=os.environ['CLICKHOUSE_HOST'],
            username='default',
            password=os.environ['CLICKHOUSE_PASSWORD'],
            secure=True,
            connect_timeout=30,        # TCP/TLS handshake
            send_receive_timeout=120, 
           )
        elapsed = time.perf_counter() - loop_start
        interval = 30 if data_source == "adsblol" else 60 * 5
        sleep_time = interval - elapsed
        print(f"Sleeping for {sleep_time:.2f} seconds before next fetch...")
        if sleep_time > 0:
            time.sleep(sleep_time)
        # if sleep_time <= 0, loop continues immediately

