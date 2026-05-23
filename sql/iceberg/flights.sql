CREATE TABLE IF NOT EXISTS {database}.flights (
    -- identity
    icao                         STRING NOT NULL,
    callsign                     STRING,
    registration                 STRING,

    -- takeoff
    takeoff_time                 TIMESTAMP NOT NULL,
    takeoff_airport_ident        STRING,

    -- landing
    landing_time                 TIMESTAMP,
    landing_airport_ident        STRING,

    -- aircraft metadata
    pia                          BOOLEAN NOT NULL,
    ladd                         BOOLEAN NOT NULL,
    military                     BOOLEAN NOT NULL,
    interesting                  BOOLEAN NOT NULL,
    aircraft_type                STRING,
    owner                        STRING,
    aircraft_description         STRING,
    category                     STRING
)
USING iceberg
PARTITIONED BY (days(takeoff_time))
LOCATION '{location}'
TBLPROPERTIES (
    'write.format.default'='parquet',
    'write.parquet.compression-codec'='zstd'
);
