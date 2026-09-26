"""A shared trajectory ranker; airport identities are never model parameters."""

import torch
from torch import nn

from flights.airport_nn.features import DISTANCE, INPUT_SIZE, VERSION


def default_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, and otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AirportRanker(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.project = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU()
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2 + INPUT_SIZE * 2, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.distance_weight = nn.Parameter(torch.tensor(5.0))

    def forward(self, values, mask):
        b, c, t, f = values.shape
        x = values.reshape(b * c, t, f)
        valid = mask.reshape(b * c, t)
        x = x.masked_fill(~valid[..., None], 0)
        encoded, _ = self.gru(self.project(x))
        lengths = valid.sum(1).clamp_min(1)
        last = encoded[torch.arange(len(x), device=x.device), lengths - 1]
        mean = x.sum(1) / lengths[:, None]
        score = self.output(
            torch.cat([encoded[:, 0], last, x[:, 0], mean], dim=1)
        ).squeeze(1)
        score = score - self.distance_weight * x[:, 0, DISTANCE]
        return score.reshape(b, c).masked_fill(~mask.any(-1), -1e4)


def load_model(path, device: str | torch.device | None = None):
    device = default_device() if device is None else torch.device(device)
    print(f"Loading airport NN on {device}", flush=True)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["feature_version"] != VERSION:
        raise ValueError("Incompatible checkpoint; train a new airport model")
    model = AirportRanker(checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["weights"])
    return model.to(device).eval()
