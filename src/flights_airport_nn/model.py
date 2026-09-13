import torch
import torch.nn as nn

from flights_airport_nn.features.features import NUMBER_OF_FEATURES


class AirportGRU(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()

        self.gru = nn.GRU(
            input_size=NUMBER_OF_FEATURES,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape:
        # (batch_size, sequence_length, input_size)

        _, hidden = self.gru(x)

        # hidden shape:
        # (num_layers, batch_size, hidden_size)

        last_hidden = hidden[-1]

        # (batch_size, hidden_size) -> (batch_size, 1)
        score = self.output(last_hidden)

        return score.squeeze(-1)