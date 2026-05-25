#!/bin/bash
set -e
python src/data_engineering/adsb/download_adsb_to_parquet.py 2026-05-24 2026-05-25 --write-to-s3 --ingest-to-iceberg
python src/flights_ai_model/features.py --features inference --start-date 2026-05-23 --end-date 2026-05-24 --write-to-s3
python src/data_engineering/flights/flights_from_model.py 2026-05-23 2026-05-24 --write-to-s3 --ingest-to-iceberg --prod
