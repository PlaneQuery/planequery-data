CREATE TABLE IF NOT EXISTS {database}.flights (
    -- identity
    icao                         STRING NOT NULL,
    callsign                     STRING,
    registration                 STRING,

    -- takeoff
    takeoff_time                 TIMESTAMP NOT NULL,
    takeoff_airport_id           BIGINT,
    takeoff_airport_ident        STRING,

    -- landing
    landing_time                 TIMESTAMP,
    landing_airport_id           BIGINT,
    landing_airport_ident        STRING,

    -- aircraft metadata
    pia                          BOOLEAN NOT NULL,
    ladd                         BOOLEAN NOT NULL,
    military                     BOOLEAN NOT NULL,
    interesting                  BOOLEAN NOT NULL,
    aircraft_type                STRING,
    owner                        STRING,
    aircraft_description         STRING,
    category                     STRING,

    -- airport catalog provenance
    airport_catalog_observed_commit_sha    STRING NOT NULL,
    airport_catalog_knowledge_cutoff_date  DATE NOT NULL,
    airport_catalog_knowledge_commit_sha   STRING NOT NULL,
    airport_catalog_available_date         DATE NOT NULL,
    airport_catalog_policy_version         STRING NOT NULL
)
USING iceberg
PARTITIONED BY (days(takeoff_time))
LOCATION '{location}'
TBLPROPERTIES (
    'write.format.default'='parquet',
    'write.parquet.compression-codec'='zstd'
);
