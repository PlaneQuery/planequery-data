from pyspark.sql import SparkSession

DATABASE = "planequery_lake_dev"
TABLE = "adsb_messages"
BASE_INPUT = (
    "s3://planequery-dev-planequery-data/data/processed/adsb/parquet_output/"
    "source=adsblol/version=6"
)
TARGET = f"glue_catalog.{DATABASE}.{TABLE}"

spark = SparkSession.builder.getOrCreate()

spark.conf.set("spark.sql.shuffle.partitions", "2000")
spark.conf.set("spark.sql.iceberg.handle-timestamp-without-timezone", "true")

# Iceberg commit retry settings
spark.conf.set("spark.sql.catalog.glue_catalog.commit.retry.num-retries", "10")
spark.conf.set("spark.sql.catalog.glue_catalog.commit.retry.min-wait-ms", "1000")
spark.conf.set("spark.sql.catalog.glue_catalog.commit.retry.max-wait-ms", "60000")
spark.conf.set("spark.sql.catalog.glue_catalog.commit.retry.total-timeout-ms", "1800000")

iceberg_cols = spark.table(TARGET).columns

for year in [2026]:
    for month in range(1, 13):
        path = f"{BASE_INPUT}/year={year}/month={month:02d}/"

        print(f"Appending {path} to {TARGET}")

        try:
            df = spark.read.parquet(path).select(*iceberg_cols)

            if df.rdd.isEmpty():
                print(f"No rows found for {path}, skipping")
                continue

            df.writeTo(TARGET).append()

            print(f"Finished {path}")

        except Exception as e:
            message = str(e)

            # Skip missing future/month folders
            if "Path does not exist" in message or "No such file or directory" in message:
                print(f"Missing path, skipping: {path}")
                continue

            raise
