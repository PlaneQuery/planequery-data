CREATE TABLE IF NOT EXISTS {database}.adsb_messages (
    -- identity / time
    time                         TIMESTAMP NOT NULL,
    icao                         STRING NOT NULL,
    callsign                     STRING,
    registration                 STRING,
    aircraft_type                STRING,

    -- position / state
    lat                          DOUBLE NOT NULL,
    lon                          DOUBLE NOT NULL,
    radius_of_containment_m      INT,
    on_ground                    BOOLEAN,

    -- altitude / vertical motion
    baro_altitude_ft             INT,
    geom_altitude_ft             INT,
    baro_rate_fpm                INT,
    geom_rate_fpm                INT,

    -- speed / horizontal motion
    ground_speed_kt              FLOAT,
    indicated_airspeed_kt        FLOAT,
    true_airspeed_kt             FLOAT,
    mach                         FLOAT,

    -- heading / orientation
    track_deg                    FLOAT,
    track_rate_deg_s             FLOAT,
    roll_deg                     FLOAT,
    magnetic_heading_deg         FLOAT,
    true_heading_deg             FLOAT,

    -- navigation
    nav_qnh_hpa                  FLOAT,
    nav_altitude_mcp_ft          INT,
    nav_altitude_fms_ft          INT,
    nav_heading_deg              FLOAT,
    nav_modes                    ARRAY<STRING>,

    -- transponder / status
    squawk                       STRING,
    emergency                    STRING,
    alert                        BOOLEAN,
    spi                          BOOLEAN,

    -- ADS-B integrity / accuracy
    version                      INT,
    nic                          INT,
    nic_baro                     INT,
    nac_p                        INT,
    nac_v                        INT,
    sil                          INT,
    sil_type                     STRING,
    gva                          INT,
    sda                          INT,

    -- source / message metadata
    type                         STRING,
    flags                        INT,
    rssi_dbfs                    FLOAT,
    source                       STRING,

    -- aircraft metadata
    no_reg_data                  BOOLEAN, -- this seems like useless column. 
    pia                          BOOLEAN,
    ladd                         BOOLEAN,
    military                     BOOLEAN,
    interesting                  BOOLEAN,
    owner                        STRING,
    aircraft_description         STRING,
    category                     STRING,

    -- derived weather / air data
    wind_direction_deg           FLOAT,
    wind_speed_kt                FLOAT,
    outside_air_temp_c           FLOAT,
    total_air_temp_c             FLOAT,

    -- data provenance
    mlat                         ARRAY<STRING>,
    tisb                         ARRAY<STRING>
)
USING iceberg
PARTITIONED BY (hours(time))
LOCATION '{location}'
TBLPROPERTIES (
    'write.format.default'='parquet',
    'write.parquet.compression-codec'='zstd'
);
