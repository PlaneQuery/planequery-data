import torch
from datetime import datetime
from pathlib import Path
import subprocess
from data_engineering.utils import OUTPUT_DIR

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

MODELS_DIR = OUTPUT_DIR / "data" / "models" / "flights_airport_nn"

def model_folder_path(num_days: int | None = 0, dt: datetime | None = None, epochs: int = 0) -> Path:
    dt = dt or datetime.now()
    date_str = dt.strftime("%Y-%m-%d_%H-%M")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
    runs_dir = MODELS_DIR / "runs"
    folder = runs_dir / f"{date_str}_{commit}_{num_days}days_{epochs}epochs"
    return folder
