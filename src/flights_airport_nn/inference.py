from datetime import date
from pathlib import Path

import numpy as np
import torch

from flights_airport_nn.features.features import (
    ALL_FEATURE_CONFIG,
    FEATURE_VERSION,
    FeatureConfig,
    FeatureName,
)
from flights_airport_nn.model import AirportGRU
from flights_airport_nn.utils import device


def load_model(model_path: str | Path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    config = FeatureConfig(
        tuple(FeatureName(name) for name in checkpoint["feature_names"])
    )
    if checkpoint.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            "Checkpoint uses incompatible feature preprocessing; retrain with feature version 2"
        )
    model = AirportGRU(
        config.input_size,
        checkpoint["hidden_size"],
        checkpoint.get("dropout", 0.15),
        distance_index=checkpoint.get("distance_index"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.temperature = checkpoint.get("temperature", 1.0)
    model.training_dates = tuple(
        date.fromisoformat(d) for d in checkpoint.get("train_dates", [])
    )
    model.validation_dates = tuple(
        date.fromisoformat(d) for d in checkpoint.get("validation_dates", [])
    )
    model.eval()
    return model, config


def predict(model, features, mask, batch_size=512):
    """Return ranking logits. Use a softmax within each endpoint for probabilities."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(features) == 0:
        return np.empty(0, np.float32)
    model.eval()
    target_device = next(model.parameters()).device
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            x = torch.as_tensor(
                features[start : start + batch_size],
                dtype=torch.float32,
                device=target_device,
            )
            m = torch.as_tensor(
                mask[start : start + batch_size], dtype=torch.bool, device=target_device
            )
            outputs.append(model(x, m).cpu().numpy())
    return np.concatenate(outputs)


def predict_endpoints(
    features, model, feature_config=ALL_FEATURE_CONFIG, top_k=3, endpoint_batch_size=16
):
    """No labels required. Confidence is conditional on the retrieved candidates."""
    if top_k < 1 or endpoint_batch_size < 1:
        raise ValueError("top_k and endpoint_batch_size must be positive")
    predictions = []
    endpoints = features.endpoints
    for start in range(0, len(endpoints), endpoint_batch_size):
        group = endpoints[start : start + endpoint_batch_size]
        candidates = [c for endpoint in group for c in endpoint.candidates]
        logits = np.empty(0, dtype=np.float32)
        if candidates:
            n = max(len(c.features) for c in candidates)
            x, masks, _ = zip(
                *(c.features_to_numpy(n, feature_config) for c in candidates)
            )
            logits = predict(model, np.asarray(x), np.asarray(masks)) / getattr(
                model, "temperature", 1.0
            )
        offset = 0
        for endpoint in group:
            row = {
                "flight_id": endpoint.flight_id,
                "endpoint": endpoint.endpoint,
                "airport_ident": None,
                "confidence": 0.0,
                "alternatives": [],
            }
            count = len(endpoint.candidates)
            if count:
                scores = logits[offset : offset + count]
                probabilities = np.exp(scores - scores.max())
                probabilities /= probabilities.sum()
                order = np.argsort(-probabilities)[:top_k]
                row["alternatives"] = [
                    {
                        "airport_ident": endpoint.candidates[i].predicted_airport.ident,
                        "confidence": float(probabilities[i]),
                    }
                    for i in order
                ]
                row.update(row["alternatives"][0])
            offset += count
            predictions.append(row)
    return predictions


def inference(features, model, feature_config=ALL_FEATURE_CONFIG):
    for row in predict_endpoints(features, model, feature_config):
        # Clear unknown endpoints so a supplied label or previous guess cannot leak through.
        setattr(
            features.flights_by_id[row["flight_id"]],
            row["endpoint"] + "_airport_ident",
            row["airport_ident"],
        )
    return features.flights_by_id


def run_inference(
    target_dt,
    model,
    feature_config=ALL_FEATURE_CONFIG,
    aircraft_batch_size=128,
    adsb_src="adsblol",
    flights=None,
):
    """Process aircraft batches so a worldwide day need not fit in RAM."""
    import polars as pl

    from data_engineering.adsb.read_adsb import read_adsb
    from data_engineering.flights.flight_type import get_flights
    from flights_airport_nn.features.features import create_features

    if aircraft_batch_size < 1:
        raise ValueError("aircraft_batch_size must be positive")
    if flights is None:
        flights = get_flights(target_dt, no_airports_model=True, adsb_src=adsb_src)
    icaos = sorted(flights["icao"].unique().to_list())
    output = {}
    for start in range(0, len(icaos), aircraft_batch_size):
        batch_icaos = icaos[start : start + aircraft_batch_size]
        batch = flights.filter(pl.col("icao").is_in(batch_icaos))
        adsb = read_adsb(
            target_dt,
            icaos=batch_icaos,
            source=adsb_src,
            columns=[
                "time",
                "icao",
                "lat",
                "lon",
                "baro_altitude_ft",
                "ground_speed_kt",
                "track_deg",
                "baro_rate_fpm",
                "on_ground",
            ],
        )
        output.update(inference(create_features(batch, adsb), model, feature_config))
    return output
