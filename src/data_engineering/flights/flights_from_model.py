import time

import polars as pl
from argparse import ArgumentParser
from datetime import date, datetime
from pathlib import Path
from flights_ai_model.inference import run_inference
from data_engineering.utils import MODEL_PATH, delete_from_s3, does_s3_key_exist, download_from_s3, s3_key_exists
from datetime import timedelta
from data_engineering.flights.flight_type import flights_parquet_path
from data_engineering.utils import upload_to_s3
import tempfile
import os
import sys
import threading
import urllib.request
import json
import urllib.error
from data_engineering.utils import OUTPUT_DIR, S3_BUCKET
from data_engineering.flights.flight_type import flights_parquet_path

BOOLEAN_COLUMNS = ["pia", "ladd", "military", "interesting"]

def _format_time(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"

def ingest_flights_to_iceberg(df_flights: pl.DataFrame, target_date: date) -> None:
    from data_engineering.spark import get_spark_session

    spark = get_spark_session()
    next_day = target_date + timedelta(days=1)
    existing_row_count = spark.sql(
        f"""
        SELECT COUNT(*) AS row_count
        FROM glue.{Glue_Database}.flights
        WHERE takeoff_time >= TIMESTAMP '{target_date:%Y-%m-%d} 00:00:00'
          AND takeoff_time < TIMESTAMP '{next_day:%Y-%m-%d} 00:00:00'
        """
    ).collect()[0]["row_count"]
    print(f"Rows already in table on {target_date}: {existing_row_count}")
    if existing_row_count > 0:
        raise Exception(f"Data for {target_date} already exists in Iceberg. Ingestion skipped.")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "data.parquet")
        df_flights.write_parquet(tmp_path)
        df = spark.read.parquet(tmp_path)
        df = df.fillna(False, subset=BOOLEAN_COLUMNS)
        df = df.coalesce(2)
        df.writeTo(f"glue.{Glue_Database}.flights").append()
    spark.stop()
    print(f"Ingested {len(df_flights)} flights into Iceberg")

def check_for_aws_spot_reclaim_signal(target_date, tmpdir, stop_event):
    URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"
    INSTANCE_ID_URL = "http://169.254.169.254/latest/meta-data/instance-id"
    start_time = time.time()
    def terminate_current_instance():
        import boto3

        with urllib.request.urlopen(INSTANCE_ID_URL, timeout=1) as response:
            instance_id = response.read().decode()
        print(f"Terminating interrupted EC2 instance {instance_id} before Batch retry.")
        ec2 = boto3.client("ec2", region_name=os.environ.get("AWS_REGION"))
        ec2.terminate_instances(InstanceIds=[instance_id])
        time.sleep(20)

    def exit_logic():
        parquet_path = flights_parquet_path(target_date)
        tmpdir_path = Path(tmpdir.name)
        print(f"SPOT INTERRUPTION SIGNAL RECEIVED after {_format_time(timedelta(seconds=time.time() - start_time))}.")
        if len(list(tmpdir_path.glob("flights_batch_*.parquet"))) == 0:
            print("No batch files found in tmpdir, skipping upload to S3.")
        else:
            pl.scan_parquet(tmpdir_path / "flights_batch_*.parquet").sink_parquet(parquet_path)
            upload_to_s3(parquet_path)
            txt_file = parquet_path.parent / "metadata.json"
            upload_to_s3(txt_file)
        try:
            terminate_current_instance()
        except Exception as e:
            print(f"Failed to terminate interrupted EC2 instance before retry: {e}")
        os._exit(99)
    while not stop_event.is_set():
        try:
            with urllib.request.urlopen(URL, timeout=1) as response:
                if response.status == 200:
                    print("SPOT INTERRUPTION WARNING!")
                    print(response.read().decode())
                    exit_logic()
        except urllib.error.URLError:
            pass
        time.sleep(5)
def main():
    parser = ArgumentParser(description="Run flight inference for a given day and ingest to Iceberg")
    parser.add_argument('start_date',help='Start date (inclusive) in YYYY-MM-DD format' ) # These are positional args. 
    parser.add_argument('end_date', help='End date (exclusive) in YYYY-MM-DD format.')
    parser.add_argument("--model",      type=Path, default=MODEL_PATH, help="Path to model .pth")
    parser.add_argument(
        '--ingest-to-iceberg',
        action='store_true',
        help='Ingest the created parquet files into Iceberg'
    )
    parser.add_argument(
        '--write-to-s3',
        action='store_true',
        help='Write the created parquet files to S3 (in addition to local filesystem)'
    )
    parser.add_argument(
        '--fresh',
        action='store_true'
    )
    parser.add_argument("--prod", action='store_true', help="Run in production mode") # TODO: This should really be envionrment variaible.
    args = parser.parse_args()
    global Glue_Database
    if args.prod:
        print("Running in PRODUCTION mode.")
        Glue_Database = "planequery_lake_prod"
    else:
        print("Running in DEV mode.")
        Glue_Database = "planequery_lake_dev"
    fresh = args.fresh if args.fresh else False
    ingest_to_iceberg = args.ingest_to_iceberg
    write_to_s3 = args.write_to_s3
    model_path = args.model
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    array_index = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX")
    current_date = start_date + timedelta(days=int(array_index or 0))
    if array_index is not None:
        end_date = current_date + timedelta(days=1)
        print(f"Running AWS Batch array child {array_index} for {current_date}")
    while current_date < end_date:
        print(f"Running inference for {current_date}")
        icaos_stop = 0
        existing_flights_df = None
        json_file_path = flights_parquet_path(current_date).parent / "metadata.json"
        if not fresh and does_s3_key_exist(flights_parquet_path(current_date)):
            print(f"Existing flights parquet found for {current_date}. Downloading from S3.")
            download_from_s3(flights_parquet_path(current_date))
            existing_flights_df = pl.read_parquet(flights_parquet_path(current_date))
            if does_s3_key_exist(json_file_path):
                download_from_s3(json_file_path)
                with open(json_file_path, "r") as f:
                    data = json.load(f)
                icaos_stop = data["icaos_stop"]
                print(f"Resuming from ICAO index {icaos_stop}")
        tmpdir = tempfile.TemporaryDirectory()
        stop_event = threading.Event()
        thread = threading.Thread(target=check_for_aws_spot_reclaim_signal, args=(current_date, tmpdir, stop_event))
        thread.start()
        df_flights = run_inference(current_date, model_path=model_path, tmpdir=tmpdir, icao_start=icaos_stop, existing_flights_df=existing_flights_df)
        stop_event.set()
        thread.join()
        # Potentially batch ec2 could get stop event and it's killed before flights inserted but unlikely. 
        if write_to_s3:
            upload_to_s3(flights_parquet_path(current_date))
            if does_s3_key_exist(json_file_path):
                delete_from_s3(json_file_path)
        if ingest_to_iceberg:
            ingest_flights_to_iceberg(df_flights, current_date)
        tmpdir.cleanup()
        current_date += timedelta(days=1)

if __name__ == "__main__":
    main()
