import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
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

    def forward(self, x, mask):
        # x:    (batch_size, sequence_length, input_size)
        # mask: (batch_size, sequence_length)

        lengths = mask.sum(dim=1).cpu()

        packed_x = pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.gru(packed_x)

        last_hidden = hidden[-1]

        score = self.output(last_hidden)

        return score.squeeze(-1)