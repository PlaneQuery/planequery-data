import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import polars as pl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import IterableDataset
from flights_ai_model.features import NULL_INDICATOR_COLS, apply_zscore

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class FlightEventBiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, num_classes=4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out)

    def forward_batched(self, x_padded, lengths):
        """Batched forward using packed sequences for variable-length inputs.

        Produces identical results to calling forward() on each sequence
        individually, but runs the GRU on the full batch at once.
        """
        packed = pack_padded_sequence(
            x_padded, lengths, batch_first=True, enforce_sorted=True
        )
        out_packed, _ = self.gru(packed)
        out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)
        return self.head(out_padded)

    def compute_loss(self, x, y, class_weights):
        logits = self.forward(x)
        B, T, C = logits.shape
        return F.cross_entropy(
            logits.reshape(B * T, C),
            y.reshape(B * T),
            weight=class_weights,
        )


FEATURE_COLS = [
    "baro_altitude_ft",
    "ground_speed_kt",
    "track_deg",
    "on_ground",
    "time_delta_s",
    "callsign_changed_from_prev",
    "above_ground_level_closest_airport",
    *NULL_INDICATOR_COLS,
] # TODO: Make this NULLABLE_COLS + ZSCORE_COLS # TODO: I could run on a subset of these to compare performance.
MAX_SEQ_LEN = 2048
WINDOW_STRIDE = 1536

class DailyFlightIterableDataset(IterableDataset):
    """Streams fixed-length flight windows from per-day parquet files."""

    def __init__(self, parquet_paths: list, max_len: int = MAX_SEQ_LEN, stride: int = WINDOW_STRIDE, shuffle: bool = False):
        self.paths = list(parquet_paths)
        self.max_len = max_len
        self.stride = stride
        self.shuffle = shuffle

    def __iter__(self):
        paths = self.paths.copy()
        if self.shuffle:
            random.shuffle(paths)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]

        for path in paths:
            df = pl.read_parquet(path)
            df = apply_zscore(df)
            groups = df.sort(["icao", "time"]).partition_by("icao", maintain_order=True)
            if self.shuffle:
                random.shuffle(groups)
            for group in groups:
                x = torch.tensor(group.select(FEATURE_COLS).to_numpy(), dtype=torch.float32)
                y = torch.tensor(group["label"].to_numpy(), dtype=torch.long)
                # Sliding windows, zero-padded to fixed length
                for start in range(0, len(x), self.stride):
                    x_win = x[start : start + self.max_len]
                    y_win = y[start : start + self.max_len]
                    pad = self.max_len - x_win.shape[0]
                    if pad > 0:
                        x_win = F.pad(x_win, (0, 0, 0, pad))
                        y_win = F.pad(y_win, (0, pad))
                    yield x_win, y_win 

