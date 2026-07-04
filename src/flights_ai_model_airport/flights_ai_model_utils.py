import subprocess
from datetime import datetime
from pathlib import Path

AIRPORT_MODEL_ENDPOINTS = ("takeoff", "landing")

def get_model_runs_dir() -> Path:
    from data_engineering.utils import OUTPUT_DIR

    return OUTPUT_DIR / "data" / "models" / "flights_ai_model_airport" / "runs"

def get_model_path(num_days: int, dt: datetime | None = None) -> Path:
    dt = dt or datetime.now()
    date_str = dt.strftime("%Y-%m-%d_%H-%M")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
    folder = f"{date_str}_{commit}_{num_days}days"
    return get_model_runs_dir() / folder

def get_latest_airport_model_path() -> Path:
    model_paths = sorted(get_model_runs_dir().glob("*/model.pkl"))
    if not model_paths:
        raise FileNotFoundError(f"No model.pkl files found in {get_model_runs_dir()}")
    return model_paths[-1]
