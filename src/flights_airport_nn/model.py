"""Candidate ranker combining endpoint geometry and recurrent trajectory evidence."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class AirportGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        dropout: float = 0.15,
        architecture: str = "ranker",
        distance_index: int | None = 4,
    ):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.architecture, self.dropout = architecture, dropout
        self.distance_index = distance_index
        if architecture == "legacy":
            self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, 1)
            return
        self.project = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU()
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 3 + input_size * 2, hidden_size * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        # Start near the nearest-airport baseline; learn corrections from data.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.distance_weight = nn.Parameter(torch.tensor(5.0))

    def forward(self, x, mask):
        if x.ndim != 3 or mask.shape != x.shape[:2]:
            raise ValueError(
                "Expected x [candidate,time,feature] and mask [candidate,time]"
            )
        mask = mask.to(x.device, dtype=torch.bool)
        if self.architecture == "legacy":
            packed = pack_padded_sequence(
                x,
                mask.sum(1).cpu().clamp_min(1),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
            return self.output(hidden[-1]).squeeze(-1)
        # Zero padded inputs; gathering the last valid state makes right-padding inert.
        x = x.masked_fill(~mask[..., None], 0)
        encoded, _ = self.gru(self.project(x))
        lengths = mask.sum(1).clamp_min(1)
        last = encoded[torch.arange(len(x), device=x.device), lengths - 1]
        weights = (
            self.attention(encoded).squeeze(-1).masked_fill(~mask, -1e4).softmax(1)
        )
        pooled = (encoded * weights[..., None]).sum(1)
        mean = x.sum(1) / lengths[:, None]
        summary = torch.cat([encoded[:, 0], last, pooled, x[:, 0], mean], dim=-1)
        score = self.output(summary).squeeze(-1)
        if self.distance_index is not None:
            score = score - self.distance_weight * x[:, 0, self.distance_index]
        return score

    def rank(self, x, mask):
        b, c, t, f = x.shape
        present = mask.any(-1)
        scores = self(x.reshape(b * c, t, f), mask.reshape(b * c, t)).reshape(b, c)
        return scores.masked_fill(~present, -1e4)
