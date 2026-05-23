"""Create the Iceberg tables (adsb_messages, flights) from sql/iceberg/*.sql
using PySpark + the Iceberg Glue catalog, then register Glue table optimizers.

Run locally:
    python scripts/create_iceberg_table.py
    python scripts/create_iceberg_table.py --prod

Requires:
    pip install pyspark boto3
    Java 11+ on PATH
    AWS credentials (env vars / aws-vault / SSO) with Glue + S3 + STS access
"""

import argparse
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from data_engineering.spark import get_spark_session

REPO_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = REPO_ROOT / "sql" / "iceberg"
REGION = "us-east-2"
ADSB_MESSAGES_BUCKET = "planequery-dev-lake"

# Tables to create.  Each entry: (table_name, sql_filename).
TABLES = [
    ("adsb_messages", "adsb_messages.sql"),
    ("flights", "flights.sql"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Iceberg tables and Glue table optimizers."
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Create tables in planequery_lake_prod. Flights use planequery-prod-lake; adsb_messages still uses planequery-dev-lake.",
    )
    return parser.parse_args()


def table_bucket(table_name: str, environment: str) -> str:
    if table_name == "adsb_messages":
        return ADSB_MESSAGES_BUCKET
    return f"planequery-{environment}-lake"


def main() -> None:
    args = parse_args()
    environment = "prod" if args.prod else "dev"
    database = f"planequery_lake_{environment}"
    optimizer_role_name = f"planequery-{environment}-glue-table-optimizer"
    warehouse = f"s3://planequery-{environment}-lake/iceberg/"

    spark = get_spark_session(
        warehouse=warehouse,
        app_name="create_iceberg_tables",
    )

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")

    for table_name, sql_filename in TABLES:
        sql_file = SQL_DIR / sql_filename
        bucket = table_bucket(table_name, environment)
        table_location = f"s3://{bucket}/iceberg/{table_name}/"
        if spark.catalog.tableExists(f"{database}.{table_name}"):
            print(f"Table {database}.{table_name} already exists; skipping creation.")
            continue

        ddl = sql_file.read_text().format(database=database, location=table_location)
        print(f"Running DDL from {sql_file} with location {table_location}")
        spark.sql(ddl)

    print(f"Tables in {database}:")
    spark.sql(f"SHOW TABLES IN {database}").show(truncate=False)

    spark.stop()

    # Register Glue managed table optimizers.  These must be created after the
    # Iceberg tables exist, so they live here rather than in the CDK stack.
    sts = boto3.client("sts", region_name=REGION)
    account_id = sts.get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{account_id}:role/{optimizer_role_name}"

    glue_client = boto3.client("glue", region_name=REGION)

    for table_name, _ in TABLES:
        for optimizer_type in ("compaction", "retention", "orphan_file_deletion"):
            kwargs = dict(
                CatalogId=account_id,
                DatabaseName=database,
                TableName=table_name,
                Type=optimizer_type,
                TableOptimizerConfiguration={"roleArn": role_arn, "enabled": True},
            )
            try:
                glue_client.create_table_optimizer(**kwargs)
                print(f"Created {optimizer_type} optimizer on {table_name}")
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "AlreadyExistsException":
                    glue_client.update_table_optimizer(**kwargs)
                    print(f"Updated {optimizer_type} optimizer on {table_name}")
                elif error_code == "AccessDeniedException":
                    raise RuntimeError(
                        "Access denied while registering Glue table optimizers. "
                        "Rerun with credentials that allow glue:CreateTableOptimizer, "
                        "glue:UpdateTableOptimizer, and iam:PassRole for "
                        f"{role_arn}. Alternatively, after the Iceberg tables exist, "
                        "deploy the CDK stack with PLANEQUERY_CREATE_TABLE_OPTIMIZERS_IN_CDK=1."
                    ) from e
                else:
                    raise

    print("Table optimizers configured.")


if __name__ == "__main__":
    main()
