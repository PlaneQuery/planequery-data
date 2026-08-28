-- Find PIA ICAOs that broadcast exactly one N-number during a date range.
--
-- Date range:
--   start_date: 2025-01-01 (inclusive)
--   end_date:   2025-02-01 (exclusive)
--
-- This reproduces the ADS-B portion of
-- src/pia/identify_ladd_planes/pia_broadcast_adsb.py. The final FAA ICAO
-- lookup remains separate because OpenAirframes FAA data is not an Athena
-- table in this repository.

WITH pia_n_number_broadcasts AS (
    SELECT DISTINCT
        LOWER(TRIM(icao)) AS pia_icao,
        UPPER(TRIM(callsign)) AS registration
    FROM planequery_lake_prod.adsb_messages
    WHERE time >= TIMESTAMP '2025-01-01 00:00:00'
      AND time < TIMESTAMP '2025-02-01 00:00:00'
      AND COALESCE(pia, FALSE)
      AND callsign IS NOT NULL
      AND STARTS_WITH(UPPER(TRIM(callsign)), 'N')
      AND UPPER(TRIM(callsign)) NOT IN ('N12345', 'N123AB')
)
SELECT
    pia_icao,
    MIN(registration) AS registration
FROM pia_n_number_broadcasts
GROUP BY pia_icao
HAVING COUNT(DISTINCT registration) = 1
ORDER BY pia_icao;
