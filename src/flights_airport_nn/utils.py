import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

from data_engineering.utils import OUTPUT_DIR

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
MODELS_DIR = OUTPUT_DIR / "data/models/flights_airport_nn"


def model_folder_path(num_days=0, dt: datetime | None = None, epochs=0) -> Path:
    timestamp = (dt or datetime.now(UTC)).strftime("%Y-%m-%d_%H-%M-%S-%f")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "nogit"
    return MODELS_DIR / "runs" / f"{timestamp}_{commit}_{num_days}days_{epochs}epochs"
