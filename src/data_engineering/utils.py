from datetime import date
import os
from pathlib import Path
from botocore.exceptions import ClientError
FLIGHTS_VERSION = "v2"
S3_BUCKET = "planequery-dev-planequery-data"
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
ATHENA_DATABASE = "planequery_lake_dev"
ATHENA_WORKGROUP ="planequery-workgroup-dev"
MODEL_PATH = OUTPUT_DIR / "data"/ "models"/ "2026-04-21/2026-04-21_20-08_1771d1f_30days_32epochs/model.pth"

def lookup_parquet_version(target_date: date): # TODO: I can just check if a v5 exists and then if not use v4.
    if target_date >= date(2026,2, 17):
        return "v5"
    elif target_date < date(2024,1,1):
        return "v5"
    else:
        return "v4"
def is_ecs():
    return (
        "ECS_CONTAINER_METADATA_URI" in os.environ
        or "ECS_CONTAINER_METADATA_URI_V4" in os.environ
    )
def get_s3_key(path: Path):
    return str(path.relative_to(OUTPUT_DIR))
def does_s3_key_exist(path: Path) -> bool:
    import boto3
    s3 = boto3.client("s3")
    s3_key = get_s3_key(path)
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise  # permissions or other errors
def download_from_s3(path: Path):
    import boto3
    s3 = boto3.client("s3")
    s3_key = get_s3_key(path)
    local_path = OUTPUT_DIR / path
    os.makedirs(local_path.parent, exist_ok=True)
    s3.download_file(S3_BUCKET, s3_key, str(local_path))
def s3_key_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise  # permissions or other errors

def upload_to_s3(path: Path):
    import boto3
    s3 = boto3.client('s3')
    s3_key = str(path.relative_to(OUTPUT_DIR))
    s3.upload_file(str(path), S3_BUCKET, s3_key)

def delete_from_s3(path: Path):
    import boto3
    s3 = boto3.client('s3')
    s3_key = str(path.relative_to(OUTPUT_DIR))
    s3.delete_object(Bucket=S3_BUCKET, Key=s3_key)