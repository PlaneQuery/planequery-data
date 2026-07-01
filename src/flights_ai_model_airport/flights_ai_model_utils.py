import subprocess
from datetime import datetime
from pathlib import Path


def get_model_save_path(num_days: int, dt: datetime | None = None) -> Path:
    from data_engineering.utils import OUTPUT_DIR

    dt = dt or datetime.now()
    date_str = dt.strftime("%Y-%m-%d_%H-%M")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
    folder = f"{date_str}_{commit}_{num_days}days"
    return OUTPUT_DIR / "data" / "models" / "flights_ai_model_airport" / "runs" / folder / "model.pkl"
