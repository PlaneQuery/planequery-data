from datetime import datetime, timedelta, timezone
import sys
import os
import requests
import argparse

import time
import logging
import clickhouse_connect
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

def process_adsb_message_stream(primary_source: str = "adsb.lol") -> str:
    start = time.perf_counter()
    now = time.time()
    fallback_until = getattr(process_adsb_message_stream, "_fallback_until", 0.0)
    fallback_source = "adsb.fi" if primary_source == "adsb.lol" else "adsb.lol"
    
    if now >= fallback_until:
        try:
            print(f"Using primary source: {primary_source}")
            adsb_messages = process_reapi_states(primary_source)
            process_adsb_message_stream._fallback_until = 0.0
            data_source = primary_source
        except Exception as exc:
            logging.warning(f"{primary_source} failed, using {fallback_source} for 10 minutes: %s", exc)
            process_adsb_message_stream._fallback_until = now + 10 * 60
            adsb_messages = process_reapi_states(fallback_source)
            data_source = fallback_source
    else:
        print(f"Primary source fallback active, using {fallback_source}")
        adsb_messages = process_reapi_states(fallback_source)
        data_source = fallback_source

    # process_takeoff_landing_events(adsb_messages)
    print(len(adsb_messages), "new messages to insert")
    insert_adsb_messages(client, adsb_messages)
    print("Inserted messages")
    end = time.perf_counter()
    print(f"Elapsed process_adsb_message_stream: {_format_duration(end - start)}")
    return data_source

from adsb_messages_types import reapi_to_row, Adsb_Message_reapi, AdsbMessageRow, insert_adsb_messages
def process_reapi_states(source: str):
    """Generic function to process ADS-B data from either adsb.lol or adsb.fi"""
    data = get_reapi_request(source)
    adsb_message_rows: list[AdsbMessageRow] = []
    
    # adsb.lol uses "aircraft", adsb.fi uses "ac"
    messages_key = "aircraft" if source == "adsb.lol" else "ac"
    messages = data[messages_key]
    
    # Handle different timestamp formats: adsb.lol uses seconds, adsb.fi uses milliseconds
    now_raw = data["now"]
    if is_epoch_utc_milliseconds(now_raw):
        now = int(now_raw)  # Already in milliseconds
    else:
        now = int(now_raw * 1000)  # Convert from seconds to milliseconds
    
    print(len(messages), f"messages received from {source}")
    for message in messages:
        print(message)
        adsb_message_reapi = Adsb_Message_reapi.from_json(message)
        if adsb_message_reapi.seen_pos is None:
            continue
        seen_pos = adsb_message_reapi.seen_pos * 1000
        message_time = now - seen_pos
        icao: str = message["hex"]
        last_message_time = get_last_message_time(icao)
        if message_time <= last_message_time:
            continue
                
        time = datetime.fromtimestamp(message_time / 1000)
        db_source = "adsb.lol" if source == "adsb.lol" else "other"
        print(f"db_source: {db_source}")
        adsb_message_row = reapi_to_row(adsb_message_reapi,
            time=time, data_source=db_source)
        adsb_message_rows.append(adsb_message_row)
        
    return adsb_message_rows

def get_reapi_request(source: str):
    """Generic function to fetch ADS-B data from either adsb.lol or adsb.fi"""
    if source == "adsb.lol":
        url = "https://re-api.adsb.lol?all"
    elif source == "adsb.fi":
        url = "https://opendata.adsb.fi/api/v2/snapshot"
    else:
        raise ValueError(f"Unknown source: {source}")
    
    attempts = 2
    delay = 1.0
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 429:
                logging.error("Rate limited by %s API on attempt %d/%d", source, attempt, attempts)
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
            logging.warning("get_reapi_request for %s attempt %d/%d failed: %s", source, attempt, attempts, e)
            if attempt < attempts:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"Failed to fetch {url} after {attempts} attempts") from last_err

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
    parser = argparse.ArgumentParser(description='ADS-B message processor')
    parser.add_argument('--primary-source', choices=['adsb.lol', 'adsb.fi'], 
                        default='adsb.lol', help='Primary ADS-B data source to try first')
    args = parser.parse_args()
    
    # python -m src.adsb_updating.main
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    print(f"Starting adsb_updating in {environment} environment")
    print(f"Primary source: {args.primary_source}")
    logging.info("Logging at Info level")
    # get_last_message_time_for_icaos()
    interval = 60*2  # 2 minute
    while True:
        loop_start = time.perf_counter()
        try:
            data_source = process_adsb_message_stream(args.primary_source)
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
        interval = 30 if data_source == "adsb.lol" else 60 * 2
        sleep_time = interval - elapsed
        print(f"Sleeping for {sleep_time:.2f} seconds before next fetch...")
        if sleep_time > 0:
            time.sleep(sleep_time)
        # if sleep_time <= 0, loop continues immediately

