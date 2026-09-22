from datetime import date
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from airports.airport_lookup import Airport
from flights.airport_nn.features import (
    ALL_FEATURE_CONFIG,
    FeatureConfig,
    FeatureName,
    Features,
    create_features_for_day,
)
from flights.airport_nn.utils import device
from flights.airport_nn.model import AirportGRU


def load_model(
    model_path: str | Path,
) -> tuple[AirportGRU, FeatureConfig]:
    """Load a trained airport GRU on the best available device."""
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    feature_config = FeatureConfig(
        enabled=tuple(
            FeatureName(name)
            for name in checkpoint["feature_names"]
        )
    )
    model = AirportGRU(
        input_size=feature_config.input_size,
        hidden_size=checkpoint["hidden_size"],
    ).to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()
    return model, feature_config


def predict(
    model: AirportGRU,
    features: np.ndarray,
    mask: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    outputs = []

    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            end = start + batch_size

            batch = torch.as_tensor(
                features[start:end],
                dtype=torch.float32,
                device=device,
            )
            mask_batch = torch.as_tensor(
                mask[start:end],
                dtype=torch.bool,
            )

            probabilities = torch.sigmoid(
                model(batch, mask_batch)
            )
            outputs.append(probabilities.cpu().numpy())

    return np.concatenate(outputs)
def inference(
    features: Features,
    model: AirportGRU,
    feature_config: FeatureConfig = ALL_FEATURE_CONFIG,
):
    X, mask, y, candidate_rows = features.numpy_features(feature_config)
    flights = features.flights_by_id
    output = predict(model, X, mask)

    scores_by_flight_id: defaultdict[str, dict[Airport, float]] = defaultdict(dict)
    for i in range(len(output)):
        score = output[i]
        candidate = candidate_rows[i]
        flight_id = candidate.flight_id
        scores_by_flight_id[flight_id][candidate.predicted_airport] = score

    for flight_id, flight_scores in scores_by_flight_id.items():
        airport = max(
            flight_scores,
            key=lambda candidate: flight_scores[candidate],
        )
        score = flight_scores[airport]
        flights[flight_id].takeoff_airport_ident = airport.ident
        
    return flights


def run_inference(
    target_dt: date,
    model: AirportGRU,
    feature_config: FeatureConfig = ALL_FEATURE_CONFIG,
):
    features = create_features_for_day(target_dt)
    flights = inference(features, model, feature_config)
    return flights

if __name__ == "__main__":
    from datetime import date
    model, feature_config = load_model(
        model_path="/Volumes/T2-SSD/planequery/data/models/flights_airport_nn/runs/2026-09-13_17-39_9a999b3_0days_0epochs/model.pt",
    )
    run_inference(date(2026, 3, 2), model, feature_config)
