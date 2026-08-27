-- Find PIA and LADD aircraft that broadcast the same non-registration
-- callsign within 200 minutes of one another.
--
-- Date range:
--   start_date: 2025-01-01 (inclusive)
--   end_date:   2025-02-01 (exclusive)
--
-- This reproduces the ADS-B matching portion of
-- src/pia/identify_ladd_planes/pia_callsign_switch.py. The final registration
-- lookup remains separate because OpenAirframes data is not an Athena table
-- in this repository.

WITH normalized_observations AS (
    SELECT DISTINCT
        time,
        LOWER(TRIM(icao)) AS icao,
        UPPER(TRIM(callsign)) AS callsign,
        COALESCE(pia, FALSE) AS pia,
        COALESCE(ladd, FALSE) AS ladd
    FROM planequery_lake_prod.adsb_messages
    WHERE time >= TIMESTAMP '2025-01-01 00:00:00'
      AND time < TIMESTAMP '2025-02-01 00:00:00'
      AND (COALESCE(pia, FALSE) OR COALESCE(ladd, FALSE))
      AND icao IS NOT NULL
      AND callsign IS NOT NULL
),
valid_observations AS (
    SELECT *
    FROM normalized_observations
    WHERE callsign <> ''
      -- ADS-B flight identification: 1-8 alphanumeric characters beginning
      -- with a letter.
      AND REGEXP_LIKE(callsign, '^[A-Z][A-Z0-9]{0,7}$')
      -- Exclude valid FAA N-numbers, including the prohibited I/O letters.
      AND NOT REGEXP_LIKE(
          callsign,
          '^N(?:[1-9][0-9]{0,4}|[1-9][0-9]{0,3}[A-HJ-NP-Z]|[1-9][0-9]{0,2}[A-HJ-NP-Z]{2})$'
      )
),
pia_observations AS (
    SELECT DISTINCT
        icao AS pia_icao,
        callsign,
        time AS pia_time
    FROM valid_observations
    WHERE pia
),
ladd_observations AS (
    SELECT DISTINCT
        icao AS ladd_icao,
        callsign,
        time AS ladd_time
    FROM valid_observations
    WHERE ladd
),
candidate_matches AS (
    SELECT
        p.pia_icao,
        l.ladd_icao,
        p.callsign,
        p.pia_time,
        l.ladd_time,
        ABS(DATE_DIFF('millisecond', p.pia_time, l.ladd_time))
            AS time_difference,
        ROW_NUMBER() OVER (
            PARTITION BY p.pia_icao, l.ladd_icao, p.callsign
            ORDER BY
                ABS(DATE_DIFF('millisecond', p.pia_time, l.ladd_time)),
                p.pia_time,
                l.ladd_time
        ) AS match_rank
    FROM pia_observations AS p
    INNER JOIN ladd_observations AS l
        ON p.callsign = l.callsign
       AND p.pia_icao <> l.ladd_icao
       AND l.ladd_time BETWEEN
           p.pia_time - INTERVAL '200' MINUTE
           AND p.pia_time + INTERVAL '200' MINUTE
)
SELECT
    pia_icao,
    ladd_icao,
    callsign,
    pia_time,
    ladd_time,
    time_difference -- milliseconds
FROM candidate_matches
WHERE match_rank = 1
ORDER BY
    pia_icao,
    ladd_icao,
    callsign;
