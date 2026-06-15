from datetime import date, timedelta

from data_engineering.utils import OUTPUT_DIR
TRAINING_FEATURE_PATHS = [
    OUTPUT_DIR / "data/features/training/v18/bts/year=2025/month=09/day=01/part-0.parquet",
    OUTPUT_DIR / "data/features/training/v18/bts/year=2025/month=09/day=02/part-0.parquet",
    OUTPUT_DIR / "data/features/training/v18/bts/year=2025/month=09/day=03/part-0.parquet",
    OUTPUT_DIR / "data/features/training/v18/bts/year=2025/month=09/day=04/part-0.parquet",
    OUTPUT_DIR / "data/features/training/v18/bts/year=2025/month=09/day=05/part-0.parquet",
    OUTPUT_DIR / "data/features/training/v18/sfdps/year=2026/month=02/day=01/part-0.parquet",
    OUTPUT_DIR / "data/features/training/v18/sfdps/year=2026/month=02/day=02/part-0.parquet",
    OUTPUT_DIR / "data/features/training/v18/sfdps/year=2026/month=02/day=03/part-0.parquet",
    OUTPUT_DIR / "data/features/training/v18/sfdps/year=2026/month=02/day=04/part-0.parquet",
    OUTPUT_DIR / "data/features/training/v18/sfdps/year=2026/month=02/day=05/part-0.parquet",
]

def sequence_training_days(bts_days=1, bts_start_date=date(2025, 9, 1), sfdps_days=1, sfdps_start_date=date(2026, 2, 1)):
    def training_features_path(source, current_date):
        return (
            OUTPUT_DIR
            / "data/features/training/v18"
            / source
            / f"year={current_date:%Y}"
            / f"month={current_date:%m}"
            / f"day={current_date:%d}"
            / "part-0.parquet"
        )

    training_features_paths = []
    for i in range(bts_days):
        current_date = bts_start_date + timedelta(days=i)
        training_features_paths.append(training_features_path("bts", current_date))

    for i in range(sfdps_days):
        current_date = sfdps_start_date + timedelta(days=i)
        training_features_paths.append(training_features_path("sfdps", current_date))

    return training_features_paths
