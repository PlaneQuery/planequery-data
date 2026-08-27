-- Return every unique PIA ICAO observed from 2024-01-01 through 2026-08-26.
-- The start and end dates are both inclusive.
--
-- This reproduces the ADS-B query in
-- src/pia/ground_truth.py:get_unique_pia_icaos_from_adsblol. Its separate FAA
-- overlap assertion is not included because OpenAirframes FAA data is not an
-- Athena table in this repository.

SELECT DISTINCT
    LOWER(icao) AS icao
FROM planequery_lake_prod.adsb_messages
WHERE time >= TIMESTAMP '2024-01-01 00:00:00'
  AND time < TIMESTAMP '2026-08-27 00:00:00'
  AND pia = TRUE
ORDER BY icao;
