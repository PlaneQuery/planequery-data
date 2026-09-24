"""Assign airports without changing flight boundaries, metadata or schema."""

from itertools import islice

import numpy as np
import polars as pl
import torch

from airports.catalog import AirportCatalogView
from flights.airport_nn.features import pack, trajectories
from flights.flight_type import with_flight_schema_columns


def predict(endpoints, model, batch_size=64):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = next(model.parameters()).device
    result = []
    model.eval()
    endpoints = iter(endpoints)
    with torch.inference_mode():
        while group := list(islice(endpoints, batch_size)):
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
                    "airport_id": None,
                    "airport_ident": None,
                    "confidence": 0.0,
                    "nearest_id": endpoint.nearest_id,
                    "nearest_ident": endpoint.nearest_ident,
                }
                if endpoint.candidate_ids:
                    logits = logits[: len(endpoint.candidate_ids)]
                    probability = np.exp(logits - logits.max())
                    probability /= probability.sum()
                    chosen = int(probability.argmax())
                    row.update(
                        airport_id=endpoint.candidate_ids[chosen],
                        airport_ident=endpoint.candidate_idents[chosen],
                        confidence=float(probability[chosen]),
                    )
                result.append(row)
    return result


def predict_flights(
    flights,
    messages,
    model,
    airport_catalog: AirportCatalogView,
):
    flights = with_flight_schema_columns(flights)
    predictions = predict(
        trajectories(flights, messages, airport_catalog),
        model,
    )
    airport_ids = {name: [None] * flights.height for name in ("takeoff", "landing")}
    airport_idents = {name: [None] * flights.height for name in ("takeoff", "landing")}
    for row in predictions:
        endpoint = row["endpoint"]
        airport_ids[endpoint][row["row"]] = row["airport_id"]
        airport_idents[endpoint][row["row"]] = row["airport_ident"]
    return flights.with_columns(
        *[
            expression
            for name in ("takeoff", "landing")
            for expression in (
                pl.Series(f"{name}_airport_id", airport_ids[name], dtype=pl.Int64),
                pl.Series(
                    f"{name}_airport_ident",
                    airport_idents[name],
                    dtype=pl.String,
                ),
            )
        ]
    )
