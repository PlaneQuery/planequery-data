"""Assign airports without changing flight boundaries, metadata or schema."""

import numpy as np
import polars as pl
import torch

from flights.airport_nn.features import pack, trajectories
from flights.flight_type import with_flight_schema_columns


def predict(endpoints, model, batch_size=64):
    device = next(model.parameters()).device
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(endpoints), batch_size):
            group = endpoints[start : start + batch_size]
            x, mask = pack(group)
            scores = (
                model(torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device))
                .cpu()
                .numpy()
            )
            for endpoint, logits in zip(group, scores):
                row = {
                    "row": endpoint.row,
                    "endpoint": endpoint.endpoint,
                    "nn": None,
                    "confidence": 0.0,
                    "nearest": endpoint.nearest,
                }
                if endpoint.airports:
                    logits = logits[: len(endpoint.airports)]
                    probability = np.exp(logits - logits.max())
                    probability /= probability.sum()
                    chosen = int(probability.argmax())
                    row.update(
                        nn=endpoint.airports[chosen],
                        confidence=float(probability[chosen]),
                    )
                result.append(row)
    return result


def predict_flights(flights, messages, model):
    flights = with_flight_schema_columns(flights)
    predictions = predict(list(trajectories(flights, messages)), model)
    airports = {name: [None] * flights.height for name in ("takeoff", "landing")}
    for row in predictions:
        airports[row["endpoint"]][row["row"]] = row["nn"]
    return flights.with_columns(
        *[
            pl.Series(f"{name}_airport_ident", values, dtype=pl.String)
            for name, values in airports.items()
        ]
    )
