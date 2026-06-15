import argparse
import math
import os
import random
import subprocess
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from flights_ai_model.inference import inference
from flights_ai_model.validation import flights_df_comparison, get_validation_flights
from flights_ai_model.features import _training_features_path, create_training_features_for_day, apply_zscore
from data_engineering.bts.read_bts_ontime_to_flights import get_bts_flights_for_day
from data_engineering.adsb.read_adsb import read_adsb

from flights_ai_model.model import (
    FEATURE_COLS,
    DailyFlightIterableDataset,
    FlightEventBiGRU,
    get_device
)

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./"))
MODELS_DIR = OUTPUT_DIR / "data" / "models" / "runs"
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


def get_model_save_path(num_days: int, epochs: int, dt: datetime | None = None) -> Path:
    """Return a save path: MODELS_DIR/YYYY-MM-DD/{file_str}-{commit}_{num_days}days_{epochs}epochs/model.pth"""
    dt = dt or datetime.now()
    date_str = dt.strftime("%Y-%m-%d_%H-%M")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
    folder = f"{date_str}_{commit}_{num_days}days_{epochs}epochs"
    return MODELS_DIR / folder / "checkpoints" / f"epoch_{epochs:03d}.pt"


def _training_source_for_date(current_date: date) -> str:
    if current_date < date(2026, 1, 1):
        return "bts"
    if current_date > date(2026, 1, 20):
        return "sfdps"
    raise ValueError("No training data source available for this date.")


def train( # this will simply take paths not start_days, end_days.. 
    train_paths,
    epochs: int | None = None,
    val_date: date | None = None,
    evaluate_performance = False,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
):
    device = get_device()
    print(f"Using device: {device}")
    output_paths = []
    if evaluate_performance == True:
        assert val_date is not None
    if evaluate_performance and val_date:
        val_source = _training_source_for_date(val_date)
        df_gold_flights = get_validation_flights(val_date, read_adsb(val_date))
        df_val_features = apply_zscore(create_training_features_for_day(val_date, val_source)) # Make this if statement so I can not always do this. 
        df_val_features = df_val_features.filter(pl.col("icao").is_in(df_gold_flights.get_column("icao").unique().to_list()))
        val_cache = _training_features_path(val_date, val_source)
        val_paths = [val_cache]

    # Auto-scale epochs: sqrt scaling from reference point (3 days / 10 epochs)
    if epochs is None:
        epochs = max(10, round(10 * math.sqrt(len(train_paths) / 3)))
    print(f"Train days: {len(train_paths)} |  Epochs: {epochs}  |  {len(FEATURE_COLS)} features")

    # Estimate class weights from all training days
    label_counts = (
        pl.concat([pl.read_parquet(p).select("label") for p in train_paths])
        .get_column("label")
        .value_counts()
        .sort("label")
    )
    counts = label_counts.get_column("count").to_list()
    max_count = max(counts)
    weights = [(max_count / c) ** 0.5 for c in counts]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"class_weights  outside={class_weights[0]:.1f}  first={class_weights[1]:.1f}  middle={class_weights[2]:.1f}  last={class_weights[3]:.1f}")

    train_loader = DataLoader(
        DailyFlightIterableDataset(train_paths, shuffle=True),
        batch_size=batch_size, num_workers=0,
    )

    model     = FlightEventBiGRU(input_dim=len(FEATURE_COLS), hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_inf_f1 = -1.0
    best_df_flights_output = None
    best_stats = None
    consecutive_nan_losses = 0
    MAX_CONSECUTIVE_NANS = 2
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(x, y, class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        avg_loss = train_loss / max(train_batches, 1)
        if math.isnan(avg_loss):
            consecutive_nan_losses += 1
            print(f"  -> NaN loss detected ({consecutive_nan_losses}/{MAX_CONSECUTIVE_NANS})")
            if consecutive_nan_losses >= MAX_CONSECUTIVE_NANS:
                print(f"  -> Stopping early: {MAX_CONSECUTIVE_NANS} consecutive NaN losses.")
                break
        else:
            consecutive_nan_losses = 0

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if evaluate_performance:
            model.eval()
            # do a brief inference to get actual performance on flights. 
            df_inferernce_flights = inference(df_val_features, model)
            df_flights_output, stats = flights_df_comparison(df_inferernce_flights, df_gold_flights) # TODO: I should not be wasting cuda time on evalauation. 
            f1 = stats["f1"]
            recall = stats["recall"]
            precision = stats["precision"]

            print(f"epoch={epoch:>3}  train_loss={avg_loss:.4f} precision={precision:.3f}  recall={recall:.3f}  inference_f1={f1:.4f}  (n_correctly_matched={stats['n_correctly_matched']}  n_predicted={stats['n_predicted']}  n_gold={stats['n_gold']})")

            if f1 > best_inf_f1:
                best_inf_f1 = f1
                best_df_flights_output = df_flights_output
                best_stats = stats
        print(f"train loss: {train_loss}")
        output_path = get_model_save_path(num_days=len(train_paths), epochs=epoch)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        print(f"  -> saved to {output_path}")
        output_paths.append(output_path)
    print(f"Training complete. Best F1: {best_inf_f1:.4f}")
    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FlightEventBiGRU")
    parser.add_argument("--start-date", default="2025-09-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--num-days",   type=int, default=3,   help="Number of days to load")
    parser.add_argument("--epochs",     type=int, default=None, help="Epochs (default: auto-scale with data size)")
    parser.add_argument("--val-date",   default="2025-08-01", help="Fixed validation date YYYY-MM-DD (default: last training day)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    args = parser.parse_args()

# nohup python -u src/flights_ai_model/train.py --start-date 2026-02-01 --num-days 15 --val-date 2026-03-01 &
