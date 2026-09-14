from datetime import date
from pathlib import Path

import numpy as np
import torch

from flights_airport_nn.features.features import FlightFeatures, create_features_for_day
from flights_airport_nn.utils import device
from flights_airport_nn.model import AirportGRU


def load_model(
    model_path: str | Path,
) -> AirportGRU:
    """Load a trained airport GRU on the best available device."""
    model = AirportGRU().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def predict(
    model: AirportGRU,
    features: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    outputs = []

    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(
                features[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            probabilities = torch.sigmoid(model(batch))
            outputs.append(probabilities.cpu().numpy())

    return np.concatenate(outputs)
def inference():
    # we batch input it. But maybe to start we go one by one?
    model = load_model(
        model_path="/Volumes/T2-SSD/planequery/data/models/flights_airport_nn/runs/2026-09-13_17-39_9a999b3_0days_0epochs/model.pt",
    )
    features = create_features_for_day(date(2026, 3, 1))
    X, mask, y = features.numpy_features()
    mapping = features.mapping
    output = predict(model, X) # yes, this is scaler.
    print(output.shape)


def run_inference_day(target_dt: date):
    pass

if __name__ == "__main__":
    inference()