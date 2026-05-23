from pyspark.sql import SparkSession
import os
import math
def _round_down_to_mb_boundary(value_mb: int, boundary_mb: int = 256) -> int:
    """Round down to a predictable Spark-friendly memory boundary."""
    return max(boundary_mb, (value_mb // boundary_mb) * boundary_mb)


def _format_spark_memory(memory_mb: int) -> str:
    """Format Spark memory settings using g where possible, otherwise m."""
    if memory_mb % 1024 == 0:
        return f"{memory_mb // 1024}g"
    return f"{memory_mb}m"


def _get_total_system_memory_mb() -> int:
    """Return total physical system memory in MB using the standard library."""
    page_size_name = "SC_PAGE_SIZE" if "SC_PAGE_SIZE" in os.sysconf_names else "SC_PAGESIZE"
    page_count_name = "SC_PHYS_PAGES"

    if page_size_name not in os.sysconf_names or page_count_name not in os.sysconf_names:
        return 4096

    total_bytes = os.sysconf(page_size_name) * os.sysconf(page_count_name)
    return max(1024, int(total_bytes / (1024**2)))


def get_local_spark_resource_config() -> dict[str, str]:
    """Size a local Spark session based on available CPU and RAM.

    The goal is to stay conservative enough for small machines while still
    using the available resources. On a 4 GB / 2 vCPU system this produces a
    config close to local[2], 1792m driver memory, and 512m overhead.
    """
    total_memory_mb = _get_total_system_memory_mb()
    logical_cpu_count = max(1, os.cpu_count() or 1)
    worker_cores = max(1, min(logical_cpu_count, 8))

    reserved_system_mb = 1024 if total_memory_mb >= 4096 else 768
    spark_budget_mb = max(1536, total_memory_mb - reserved_system_mb)

    driver_memory_mb = _round_down_to_mb_boundary(int(spark_budget_mb * 0.6))
    driver_memory_mb = min(driver_memory_mb, max(1024, total_memory_mb - 768))

    overhead_target_mb = max(384, int(driver_memory_mb * 0.2))
    driver_overhead_mb = int(math.ceil(overhead_target_mb / 128) * 128)

    max_spark_memory_mb = max(1408, total_memory_mb - 512)
    if driver_memory_mb + driver_overhead_mb > max_spark_memory_mb:
        driver_memory_mb = _round_down_to_mb_boundary(
            int(max_spark_memory_mb * 0.8),
            boundary_mb=256,
        )
        driver_overhead_mb = max(384, max_spark_memory_mb - driver_memory_mb)

    partitions = max(2, min(32, worker_cores * 2))

    return {
        "master": f"local[{worker_cores}]",
        "spark.driver.memory": _format_spark_memory(driver_memory_mb),
        "spark.driver.memoryOverhead": _format_spark_memory(driver_overhead_mb),
        "spark.sql.shuffle.partitions": str(partitions),
        "spark.default.parallelism": str(partitions),
    }

def get_spark_session(
    warehouse: str = "s3://planequery-dev-lake/iceberg/",
    app_name: str = "planequery_spark",
) -> SparkSession:
    from pyspark.sql import SparkSession

    REGION = "us-east-2"

    ICEBERG_PKG = "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.0"
    ICEBERG_AWS_BUNDLE = "org.apache.iceberg:iceberg-aws-bundle:1.10.0"
    spark_resource_config = get_local_spark_resource_config()

    print(
        "Using local Spark config "
        f"master={spark_resource_config['master']}, "
        f"driver_memory={spark_resource_config['spark.driver.memory']}, "
        f"driver_overhead={spark_resource_config['spark.driver.memoryOverhead']}, "
        f"shuffle_partitions={spark_resource_config['spark.sql.shuffle.partitions']}"
    )

    spark = (
        SparkSession.builder.appName(app_name)
        .master(spark_resource_config["master"])
        .config("spark.jars.packages", f"{ICEBERG_PKG},{ICEBERG_AWS_BUNDLE}")
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.glue", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.glue.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog")
        .config("spark.sql.catalog.glue.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        .config("spark.sql.catalog.glue.warehouse", warehouse)
        .config("spark.sql.catalog.glue.client.region", REGION)
        .config("spark.sql.defaultCatalog", "glue")
        .config("spark.sql.session.timeZone", "UTC")

        # Keep local Spark from binding to an unroutable hostname on macOS/VPNs.
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.blockManager.port", "0")

        # memory
        .config("spark.driver.memory", spark_resource_config["spark.driver.memory"])
        .config("spark.driver.memoryOverhead", spark_resource_config["spark.driver.memoryOverhead"])
        .config("spark.sql.shuffle.partitions", spark_resource_config["spark.sql.shuffle.partitions"])
        .config("spark.default.parallelism", spark_resource_config["spark.default.parallelism"])
        .config("spark.sql.adaptive.enabled", "true")

        # Iceberg write behavior
        .config("spark.sql.catalog.glue.write.distribution-mode", "none")
        .config("spark.sql.catalog.glue.write.target-file-size-bytes", str(128 * 1024 * 1024))

        .getOrCreate()
    )
    return spark
