-- Quickly find missing or low-volume ADS-B days from Iceberg metadata.
--
-- The range is inclusive of both 2024-01-01 and 2026-08-26. A missing day
-- has zero current Iceberg records; a low-volume day has fewer than 100,000.
-- This reads partition metadata rather than scanning the ADS-B Parquet rows.

WITH expected_days AS (
    SELECT day
    FROM UNNEST(
        SEQUENCE(
            DATE '2024-01-01',
            DATE '2026-08-26',
            INTERVAL '1' DAY
        )
    ) AS dates(day)
),
daily_partition_counts AS (
    SELECT
        CAST(
            DATE_ADD(
                'hour',
                CAST(partitions."partition".time_hour AS BIGINT),
                TIMESTAMP '1970-01-01 00:00:00'
            ) AS DATE
        ) AS day,
        SUM(partitions.record_count) AS message_count
    FROM "planequery_lake_prod"."adsb_messages$partitions" AS partitions
    -- Iceberg exposes hours(time) here as integer hours since Unix epoch.
    WHERE partitions."partition".time_hour >= 473352 -- 2024-01-01 00:00 UTC
      AND partitions."partition".time_hour < 496608  -- 2026-08-27 00:00 UTC
    GROUP BY 1
)
SELECT
    expected_days.day,
    COALESCE(daily_partition_counts.message_count, 0) AS message_count,
    CASE
        WHEN daily_partition_counts.day IS NULL THEN 'missing'
        ELSE 'below_threshold'
    END AS status
FROM expected_days
LEFT JOIN daily_partition_counts
    ON expected_days.day = daily_partition_counts.day
WHERE COALESCE(daily_partition_counts.message_count, 0) < 100000
ORDER BY expected_days.day;
