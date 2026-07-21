import os
from pathlib import Path

from botocore.exceptions import ClientError

FLIGHTS_VERSION = "v2"
S3_BUCKET = "planequery-dev-planequery-data"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./"))


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
        raise


def download_from_s3(path: Path):
    import boto3
    s3 = boto3.client("s3")
    s3_key = get_s3_key(path)
    local_path = OUTPUT_DIR / path
    os.makedirs(local_path.parent, exist_ok=True)
    s3.download_file(S3_BUCKET, s3_key, str(local_path))


def download_prefix_from_s3(path: Path) -> int:
    """Download every object below an OUTPUT_DIR-relative S3 prefix."""

    import boto3

    s3 = boto3.client("s3")
    prefix = get_s3_key(path).rstrip("/") + "/"
    downloaded = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            local_path = OUTPUT_DIR / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(S3_BUCKET, key, str(local_path))
            downloaded += 1
    return downloaded


def upload_to_s3(path: Path):
    import boto3

    s3 = boto3.client("s3")
    s3_key = get_s3_key(path)
    s3.upload_file(str(path), S3_BUCKET, s3_key)
