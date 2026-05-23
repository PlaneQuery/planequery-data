#!/bin/bash
set -e
python src/data_engineering/adsb/download_adsb_to_parquet.py 2026-05-22 2026-05-23 --write-to-s3 --ingest-to-iceberg
python src/flights_ai_model/features.py --features inference --start-date 2026-05-21 --end-date 2026-05-22 --write-to-s3
python src/data_engineering/flights/flights_from_model.py 2026-05-21 2026-05-22 --write-to-s3 --ingest-to-iceberg --prod
