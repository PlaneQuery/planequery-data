"""A shared trajectory ranker; airport identities are never model parameters."""

import torch
from torch import nn

from flights_airport_nn.features import DISTANCE, INPUT_SIZE, VERSION


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


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["feature_version"] != VERSION:
        raise ValueError("Incompatible checkpoint; train a new airport model")
    model = AirportRanker(checkpoint["hidden_size"]).to(device)
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    return model
